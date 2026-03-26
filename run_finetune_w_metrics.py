"""
Finetuning + reverse experiment with per-step gradient metrics.

Same as run_finetune.py, but at each eval step also computes:
  - Gradient projection (dot product, cosine sim) between task pairs
  - Layerwise drift (weight change since last eval step)

These are stored in the history alongside the standard val metrics.
"""

import os
import json
import torch
from typing import Dict
from torch.utils.data import DataLoader

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
from pcfg_gen import (
    CharTokenizer, PCFGDataset, PCFGGenerator,
    generate_dataset, build_pools, format_example, collate_fn,
)
from train_help import train
from gradient_metrics import (
    compute_gradient_projection,
    compute_layerwise_drift,
    snapshot_state,
    build_task_loaders_weighted,
)

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
correlation_values = experiment_cfg["correlation_values"]
concentration_values = experiment_cfg["concentration_values"]

# Build shared pools
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
n_layers = model_cfg["n_layer"]

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
print(f"Finetune + Reverse Experiment (with gradient metrics)")
print(f"Correlation values: {correlation_values}")
print(f"Concentration values: {concentration_values}")
print(f"{'='*80}\n")


# ---- Helpers ----

def generate_dataset_from_pool(pool, n_examples, task_names, task_reg):
    import random
    examples = []
    for _ in range(n_examples):
        pcfg_string = random.choice(pool)
        task_name = random.choice(task_names)
        task_def, answer = task_reg.apply_task(task_name, pcfg_string)
        examples.append(format_example(pcfg_string, task_def, answer))
    return examples


def build_eval_datasets(pools, cfg_dict):
    all_tasks = cfg_dict["task_sets"]["all"]
    other_tasks = [t for t in all_tasks if t not in ["count_a", "count_b"]]
    n_val = cfg_dict["data"]["val_examples"]
    max_len = cfg_dict["tokenizer"]["max_length"]
    mask_answer_only_val = cfg_dict["tokenizer"]["mask_answer_only_val"]
    per_other = cfg_dict["data"].get("eval_per_other_task", 500)

    return {
        "count_a_corr": PCFGDataset(
            generate_dataset_from_pool(pools["correlated"], n_val, ["count_a"], task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_a_uncorr": PCFGDataset(
            generate_dataset_from_pool(pools["uncorrelated"], n_val, ["count_a"], task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_b_corr": PCFGDataset(
            generate_dataset_from_pool(pools["correlated"], n_val, ["count_b"], task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "count_b_uncorr": PCFGDataset(
            generate_dataset_from_pool(pools["uncorrelated"], n_val, ["count_b"], task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "all_other_corr": PCFGDataset(
            generate_dataset_from_pool(pools["correlated"], per_other * len(other_tasks), other_tasks, task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
        "all_other_uncorr": PCFGDataset(
            generate_dataset_from_pool(pools["uncorrelated"], per_other * len(other_tasks), other_tasks, task_registry),
            tokenizer, max_length=max_len, mask_answer_only=mask_answer_only_val),
    }


def get_final_metric(history: Dict, split_name: str, metric_name: str):
    vals = history.get("val", {}).get(split_name, {}).get(metric_name, [])
    return vals[-1] if vals else None


# ---- Build eval datasets and metric loaders ----
eval_datasets = build_eval_datasets(pools, cfg)

# Build DataLoaders for gradient metrics (smaller batch size, just for gradient computation)
METRIC_BATCH_SIZE = 64
metric_loaders = {
    name: DataLoader(ds, batch_size=METRIC_BATCH_SIZE, shuffle=False,
                     collate_fn=lambda b, tok=tokenizer: collate_fn(b, tok))
    for name, ds in eval_datasets.items()
}

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

# Task pairs to compute gradient projection for
TASK_PAIRS = [
    ("count_a", "count_b"),
    ("count_a", "all_other"),
]


def make_on_eval(correlation, prev_state_holder):
    """
    Create an on_eval callback that computes gradient metrics at each eval step.

    prev_state_holder: dict with key 'state' holding the previous model snapshot.
    """
    task_loaders = build_task_loaders_weighted(metric_loaders, correlation)

    def on_eval(model, step, history):
        # ---- Gradient projection ----
        for task_a, task_b in TASK_PAIRS:
            if task_a not in task_loaders or task_b not in task_loaders:
                continue
            proj = compute_gradient_projection(
                model, task_loaders[task_a], task_loaders[task_b],
                device, n_layers,
            )
            pair_key = f"grad_{task_a}_vs_{task_b}"
            if pair_key not in history:
                history[pair_key] = {
                    "dot_product": {l: [] for l in range(n_layers)},
                    "cosine_sim": {l: [] for l in range(n_layers)},
                    "norm_a": {l: [] for l in range(n_layers)},
                    "norm_b": {l: [] for l in range(n_layers)},
                }
            for l in range(n_layers):
                if l in proj:
                    for metric_name in ["dot_product", "cosine_sim", "norm_a", "norm_b"]:
                        history[pair_key][metric_name][l].append(proj[l][metric_name])

            cos_str = ", ".join(f"L{l}:{proj[l]['cosine_sim']:.3f}" for l in sorted(proj))
            print(f"      {pair_key} cos: [{cos_str}]")

        # ---- Layerwise drift ----
        drift = compute_layerwise_drift(model, prev_state_holder["state"], n_layers)
        if "layerwise_drift" not in history:
            history["layerwise_drift"] = {l: [] for l in range(n_layers)}
            for extra_key in ["embedding", "pos_embedding", "head"]:
                history["layerwise_drift"][extra_key] = []

        for key, val in drift.items():
            if key not in history["layerwise_drift"]:
                history["layerwise_drift"][key] = []
            history["layerwise_drift"][key].append(val)

        drift_str = ", ".join(f"L{l}:{drift[l]:.4f}" for l in range(n_layers))
        print(f"      drift: [{drift_str}]")

        # Update snapshot for next drift computation
        prev_state_holder["state"] = snapshot_state(model)

    return on_eval


# ---- Main experiment loop ----
all_results = {}

for correlation in correlation_values:
    print(f"\n{'='*80}")
    print(f"Correlation: {correlation}")
    print(f"{'='*80}")

    correlation_key = f"corr_{correlation:.2f}"
    all_results[correlation_key] = {}

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

        # Snapshot for drift tracking (starts from pretrained weights)
        ft_prev_state = {"state": snapshot_state(model)}

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
            on_eval=make_on_eval(correlation, ft_prev_state),
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

        # Fresh snapshot for reverse drift tracking
        rv_prev_state = {"state": snapshot_state(model)}

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
            on_eval=make_on_eval(correlation, rv_prev_state),
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
