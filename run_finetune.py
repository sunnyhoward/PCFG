"""
Finetuning + reverse experiment script.

Loads existing pretrained models (pretrain_corr_X.XX.pth) and runs:
  - Finetuning at each concentration value
  - Reverse training after each finetune

Pretrained models are expected at: pretrain_corr_{correlation:.2f}.pth
(relative to the script directory, i.e. /workspace/PCFG/)
"""

import os
import json
import torch
from typing import Dict

from config import CFG
from config_utils import (
    build_optimizer,
    build_task_registry,
    get_device,
    get_warmup_steps,
    resolve_task_weights,
    set_seed,
)
from mingpt import GPT, GPTConfig
from pcfg_gen import CharTokenizer, PCFGDataset, PCFGGenerator, generate_dataset, build_pools, format_example
from train_help import train

cfg = CFG
cfg["device"] = "cuda:0"

set_seed(cfg["seed"])
device = get_device(cfg["device"])
print(f"Using device: {device}")

# Create result directories
results_dir = cfg["paths"]["results_dir"]
models_dir = cfg["paths"]["models_dir"]
histories_dir = cfg["paths"]["histories_dir"]
os.makedirs(models_dir, exist_ok=True)
os.makedirs(histories_dir, exist_ok=True)

# Build task registry
task_registry = build_task_registry(cfg["task_definitions"])
task_sets = cfg["task_sets"]
pretrain_tasks = task_sets["pretrain"]

chunk_size = cfg["pcfg"]["chunk_size"]
experiment_cfg = cfg["experiment"]
correlation_values = experiment_cfg["correlation_values"]   # [0.0, 0.25, 0.5, 0.75, 1.0]
concentration_values = experiment_cfg["concentration_values"]  # [0.1, 0.3, 0.5, 0.7, 0.9]

# Build shared pools (needed for finetuning batches)
pool_cfg = cfg.get("pool", {})
pcfg = PCFGGenerator()
pools = build_pools(
    pcfg_gen=pcfg,
    n_correlated=pool_cfg.get("n_correlated", 100000),
    n_uncorrelated=pool_cfg.get("n_uncorrelated", 100000),
    chunk_size=chunk_size,
    verbose=True,
)

model_cfg = cfg["model"]
tokenizer = CharTokenizer()

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=model_cfg["block_size"],
    n_layer=model_cfg["n_layer"],
    n_head=model_cfg["n_head"],
    n_embd=model_cfg["n_embd"],
    embd_pdrop=model_cfg["embd_pdrop"],
    resid_pdrop=model_cfg["resid_pdrop"],
    attn_pdrop=model_cfg["attn_pdrop"],
)

print(f"\n{'='*80}")
print(f"Finetune + Reverse Experiment")
print(f"Correlation values: {correlation_values}")
print(f"Concentration values: {concentration_values}")
print(f"{'='*80}\n")


def generate_dataset_from_pool(pool, n_examples, task_names, task_reg):
    """Build examples by sampling strings from a pre-built pool."""
    import random
    examples = []
    for _ in range(n_examples):
        pcfg_string = random.choice(pool)
        task_name = random.choice(task_names)
        task_def, answer = task_reg.apply_task(task_name, pcfg_string)
        examples.append(format_example(pcfg_string, task_def, answer))
    return examples


def build_eval_datasets(pcfg, tokenizer, task_registry, chunk_size, cfg_dict, pools):
    all_tasks = cfg_dict["task_sets"]["all"]
    other_tasks = [t for t in all_tasks if t not in ["count_a", "count_b"]]
    n_val = cfg_dict["data"]["val_examples"]
    max_len = cfg_dict["tokenizer"]["max_length"]
    mask_answer_only_val = cfg_dict["tokenizer"]["mask_answer_only_val"]

    # count_a — correlated strings (count_a == count_b + 1)
    count_a_corr_examples = generate_dataset_from_pool(
        pools["correlated"], n_val, ["count_a"], task_registry,
    )
    # count_a — uncorrelated strings (natural PCFG)
    count_a_uncorr_examples = generate_dataset_from_pool(
        pools["uncorrelated"], n_val, ["count_a"], task_registry,
    )

    # count_b — correlated strings (count_a == count_b + 1)
    count_b_corr_examples = generate_dataset_from_pool(
        pools["correlated"], n_val, ["count_b"], task_registry,
    )
    # count_b — uncorrelated strings (natural PCFG)
    count_b_uncorr_examples = generate_dataset_from_pool(
        pools["uncorrelated"], n_val, ["count_b"], task_registry,
    )

    per_other = cfg_dict["data"].get("eval_per_other_task", 500)
    other_corr_examples = generate_dataset_from_pool(
        pools["correlated"], per_other * len(other_tasks), other_tasks, task_registry,
    )
    other_uncorr_examples = generate_dataset_from_pool(
        pools["uncorrelated"], per_other * len(other_tasks), other_tasks, task_registry,
    )

    return {
        "count_a_corr": PCFGDataset(count_a_corr_examples, tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_a_uncorr": PCFGDataset(count_a_uncorr_examples, tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_b_corr": PCFGDataset(count_b_corr_examples, tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_b_uncorr": PCFGDataset(count_b_uncorr_examples, tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "all_other_corr": PCFGDataset(other_corr_examples, tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "all_other_uncorr": PCFGDataset(other_uncorr_examples, tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
    }


def get_final_metric(history: Dict, split_name: str, metric_name: str):
    vals = history.get("val", {}).get(split_name, {}).get(metric_name, [])
    return vals[-1] if vals else None


# Pre-build eval datasets (same for all correlations since eval_use_correlation=False)
eval_datasets = build_eval_datasets(pcfg, tokenizer, task_registry, chunk_size, cfg, pools)

pretrain_task_weights = resolve_task_weights(
    pretrain_tasks,
    "operand_probs",
    cfg["operand_probs"],
)

finetune_train_tasks = [
    "count_a", "count_b", "count_c",
    "count_aa", "count_bb", "count_cc",
    "index_a", "index_b", "index_c",
    "index_aa", "index_bb", "index_cc",
    "token_at_40",
]

all_results = {}

for correlation in correlation_values:
    print(f"\n{'='*80}")
    print(f"Correlation: {correlation}")
    print(f"{'='*80}")

    correlation_key = f"corr_{correlation:.2f}"
    all_results[correlation_key] = {}

    # Load pretrained model — try local dir first, then results/models/
    pretrain_paths = [
        f"pretrain_corr_{correlation:.2f}.pth",
        os.path.join(models_dir, f"pretrain_corr_{correlation:.2f}.pth"),
    ]
    pretrain_model_path = None
    for p in pretrain_paths:
        if os.path.exists(p):
            pretrain_model_path = p
            break
    if pretrain_model_path is None:
        print(f"  [SKIP] No pretrained model found for correlation={correlation:.2f}. Tried: {pretrain_paths}")
        continue
    print(f"  Loading pretrained model: {pretrain_model_path}")

    for concentration in concentration_values:
        print(f"\n  Concentration: {concentration:.2f}")

        conc_key = f"conc_{concentration:.2f}"
        all_results[correlation_key][conc_key] = {}

        # Load fresh copy of pretrained model
        model = GPT(config).to(device)
        ckpt = torch.load(pretrain_model_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        # ---- Finetuning ----
        finetune_optimizer = build_optimizer(
            model.parameters(),
            cfg["optimizer"],
            experiment_cfg["finetune_lr"],
        )

        other_task_weight = (1 - concentration) / len(finetune_train_tasks[1:])
        finetune_weights = [concentration, *[other_task_weight] * (len(finetune_train_tasks) - 1)]

        history_finetune = train(
            model=model,
            tokenizer=tokenizer,
            device=device,
            steps=experiment_cfg["finetune_steps"],
            batch_size=experiment_cfg["finetune_batch_size"],
            lr=experiment_cfg["finetune_lr"],
            min_lr=experiment_cfg["finetune_min_lr"],
            warmup_steps=get_warmup_steps(
                experiment_cfg["finetune_steps"],
                warmup_ratio=experiment_cfg["finetune_warmup_ratio"],
            ),
            log_interval=experiment_cfg["finetune_log_interval"],
            task_names=finetune_train_tasks,
            task_weights=finetune_weights,
            pcfg_gen=pcfg,
            task_reg=task_registry,
            chunk_size=chunk_size,
            mask_answer_only=False,
            max_grad_norm=experiment_cfg["max_grad_norm"],
            optimizer=finetune_optimizer,
            val_datasets=eval_datasets,
            val_batch_size=experiment_cfg["finetune_batch_size"],
            log_prefix=f"Finetune[corr={correlation:.2f},conc={concentration:.2f}]",
            use_lr_schedule=True,
            metrics=cfg.get("metrics"),
            data_pools=pools,
            correlation=correlation,
        )

        finetune_model_path = os.path.join(models_dir, f"finetune_corr_{correlation:.2f}_conc_{concentration:.2f}.pth")
        finetune_history_path = os.path.join(histories_dir, f"finetune_corr_{correlation:.2f}_conc_{concentration:.2f}_history.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': finetune_optimizer.state_dict(),
            'correlation': correlation,
            'concentration': concentration,
            'stage': 'finetune',
        }, finetune_model_path)
        torch.save(history_finetune, finetune_history_path)
        print(f"    Saved finetuned model: {finetune_model_path}")

        # ---- Reverse training ----
        reverse_optimizer = build_optimizer(
            model.parameters(),
            cfg["optimizer"],
            experiment_cfg["reverse_lr"],
        )

        reverse_history = train(
            model=model,
            tokenizer=tokenizer,
            device=device,
            steps=experiment_cfg["reverse_steps"],
            batch_size=experiment_cfg["reverse_batch_size"],
            lr=experiment_cfg["reverse_lr"],
            min_lr=experiment_cfg["reverse_min_lr"],
            warmup_steps=get_warmup_steps(
                experiment_cfg["reverse_steps"],
                warmup_ratio=experiment_cfg["reverse_warmup_ratio"],
            ),
            log_interval=experiment_cfg["reverse_log_interval"],
            task_names=pretrain_tasks,
            task_weights=pretrain_task_weights,
            pcfg_gen=pcfg,
            task_reg=task_registry,
            chunk_size=chunk_size,
            mask_answer_only=False,
            max_grad_norm=experiment_cfg["max_grad_norm"],
            optimizer=reverse_optimizer,
            val_datasets=eval_datasets,
            val_batch_size=experiment_cfg["reverse_batch_size"],
            log_prefix=f"Reverse[corr={correlation:.2f},conc={concentration:.2f}]",
            use_lr_schedule=False,
            metrics=cfg.get("metrics"),
            data_pools=pools,
            correlation=correlation,
        )

        reverse_model_path = os.path.join(models_dir, f"reverse_corr_{correlation:.2f}_conc_{concentration:.2f}.pth")
        reverse_history_path = os.path.join(histories_dir, f"reverse_corr_{correlation:.2f}_conc_{concentration:.2f}_history.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': reverse_optimizer.state_dict(),
            'correlation': correlation,
            'concentration': concentration,
            'stage': 'reverse',
        }, reverse_model_path)
        torch.save(reverse_history, reverse_history_path)
        print(f"    Saved reverse model: {reverse_model_path}")

        def _record_phase(hist):
            splits = ["count_a_corr", "count_a_uncorr",
                       "count_b_corr", "count_b_uncorr",
                       "all_other_corr", "all_other_uncorr"]
            d = {}
            for s in splits:
                d[f"{s}_acc"]  = get_final_metric(hist, s, "answer_acc")
                d[f"{s}_loss"] = get_final_metric(hist, s, "loss")
            return d

        all_results[correlation_key][conc_key]["finetune_final"] = _record_phase(history_finetune)
        all_results[correlation_key][conc_key]["reverse_final"]  = _record_phase(reverse_history)

# Save summary
summary_path = os.path.join(results_dir, "finetune_experiment_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved experiment summary: {summary_path}")
print(f"\n{'='*80}")
print(f"Experiment completed! Results saved to: {results_dir}")
print(f"{'='*80}")
