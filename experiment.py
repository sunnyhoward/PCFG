"""
Master experiment script for PCFG 2: Correlation-based learning.

This script trains models at various correlation levels, then finetunes them 
at different data concentrations.

Workflow:
1. For each correlation value:
   a. Pretrain model on all tasks except count_a
   b. Save pretrained model
   c. For each concentration value:
      i. Load pretrained model
      ii. Finetune on count_a (some %) + count_b (rest %)
      iii. Save results and model
"""

import os
import json
import torch
from typing import Dict, List

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
from pcfg_gen import CharTokenizer, PCFGDataset, PCFGGenerator, generate_dataset, build_pools
from train_help import train

cfg = CFG
cfg["device"] = "cuda:0"

set_seed(cfg["seed"])
device = get_device(cfg["device"])
if cfg.get("debug", {}).get("print_device", True):
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
finetune_tasks = task_sets["finetune"]

chunk_size = cfg["pcfg"]["chunk_size"]

# Get experiment parameters
experiment_cfg = cfg["experiment"]
correlation_values = [0]#experiment_cfg["correlation_values"]
concentration_values = experiment_cfg["concentration_values"]

# Build shared PCFG generator and pools (one-time cost for the whole experiment)
pool_cfg = cfg.get("pool", {})
pcfg = PCFGGenerator()
pools = build_pools(
    pcfg_gen=pcfg,
    n_correlated=pool_cfg.get("n_correlated", 100000),
    n_uncorrelated=pool_cfg.get("n_uncorrelated", 1010000),
    chunk_size=chunk_size,
    verbose=True,
)

# Model and training configs
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
print(f"Starting PCFG 2 Experiment")
print(f"Correlation values: {correlation_values}")
print(f"Concentration values: {concentration_values}")
print(f"{'='*80}\n")


def build_eval_datasets(
    pcfg: PCFGGenerator,
    tokenizer: CharTokenizer,
    task_registry,
    chunk_size: int,
    cfg_dict: Dict,
    eval_use_correlation: bool,
):
    all_tasks = cfg_dict["task_sets"]["all"]
    other_tasks = [t for t in all_tasks if t not in ["count_a", "count_b"]]

    count_a_examples = generate_dataset(
        n_examples=cfg_dict["data"]["val_examples"],
        task_names=["count_a"],
        pcfg_gen=pcfg,
        task_reg=task_registry,
        chunk_size=chunk_size,
        use_correlation=eval_use_correlation,
    )
    count_b_examples = generate_dataset(
        n_examples=cfg_dict["data"]["val_examples"],
        task_names=["count_b"],
        pcfg_gen=pcfg,
        task_reg=task_registry,
        chunk_size=chunk_size,
        use_correlation=eval_use_correlation,
    )

    per_other = cfg_dict["data"].get("eval_per_other_task", 500)
    other_examples = []
    for task_name in other_tasks:
        other_examples.extend(
            generate_dataset(
                n_examples=per_other,
                task_names=[task_name],
                pcfg_gen=pcfg,
                task_reg=task_registry,
                chunk_size=chunk_size,
                use_correlation=eval_use_correlation,
            )
        )

    max_len = cfg_dict["tokenizer"]["max_length"]
    mask_answer_only_val = cfg_dict["tokenizer"]["mask_answer_only_val"]

    return {
        "count_a": PCFGDataset(
            count_a_examples,
            tokenizer,
            max_length=max_len,
            mask_answer_only=mask_answer_only_val,
        ),
        "count_b": PCFGDataset(
            count_b_examples,
            tokenizer,
            max_length=max_len,
            mask_answer_only=mask_answer_only_val,
        ),
        "all_other_avg": PCFGDataset(
            other_examples,
            tokenizer,
            max_length=max_len,
            mask_answer_only=mask_answer_only_val,
        ),
    }


def get_final_metric(history: Dict, split_name: str, metric_name: str):
    vals = history.get("val", {}).get(split_name, {}).get(metric_name, [])
    return vals[-1] if vals else None


# Main experiment loop
all_results = {}

for correlation_idx, correlation in enumerate(correlation_values):
    print(f"\n{'='*80}")
    print(f"Correlation {correlation_idx+1}/{len(correlation_values)}: {correlation}")
    print(f"{'='*80}")
    
    correlation_key = f"corr_{correlation:.2f}"
    all_results[correlation_key] = {}
    
    eval_datasets = build_eval_datasets(
        pcfg=pcfg,
        tokenizer=tokenizer,
        task_registry=task_registry,
        chunk_size=chunk_size,
        cfg_dict=cfg,
        eval_use_correlation=False,
    )
    
    # ==================== PRETRAINING ====================
    print(f"\n--- Pretraining at correlation={correlation} ---")
    
    # Create fresh model for this correlation
    model = GPT(config).to(device)
    
    pretrain_cfg = experiment_cfg
    pretrain_optimizer = build_optimizer(
        model.parameters(),
        cfg["optimizer"],
        pretrain_cfg["pretrain_lr"],
    )
    
    pretrain_task_weights = resolve_task_weights(
        pretrain_tasks,
        "operand_probs",
        cfg["operand_probs"],
    )
    
    history_pretrain = train(
        model=model,
        tokenizer=tokenizer,
        device=device,
        steps=pretrain_cfg["pretrain_steps"],
        batch_size=pretrain_cfg["pretrain_batch_size"],
        lr=pretrain_cfg["pretrain_lr"],
        min_lr=pretrain_cfg["pretrain_min_lr"],
        warmup_steps=get_warmup_steps(
            pretrain_cfg["pretrain_steps"],
            warmup_ratio=pretrain_cfg["pretrain_warmup_ratio"],
        ),
        log_interval=pretrain_cfg["pretrain_log_interval"],
        task_names=pretrain_tasks,
        task_weights=pretrain_task_weights,
        pcfg_gen=pcfg,
        task_reg=task_registry,
        chunk_size=chunk_size,
        mask_answer_only=False,
        max_grad_norm=pretrain_cfg["max_grad_norm"],
        optimizer=pretrain_optimizer,
        val_datasets=eval_datasets,
        val_batch_size=pretrain_cfg["pretrain_batch_size"],
        log_prefix=f"Pretrain[corr={correlation:.2f}]",
        use_lr_schedule=True,
        metrics=cfg.get("metrics"),
        data_pools=pools,
        correlation=correlation,
    )
    
    # Save pretrained model
    pretrain_model_path = os.path.join(
        models_dir,
        f"pretrain_corr_{correlation:.2f}.pth",
    )
    pretrain_history_path = os.path.join(
        histories_dir,
        f"pretrain_corr_{correlation:.2f}_history.pth",
    )
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': pretrain_optimizer.state_dict(),
        'correlation': correlation,
    }
    torch.save(checkpoint, pretrain_model_path)
    torch.save(history_pretrain, pretrain_history_path)
    print(f"Saved pretrained model: {pretrain_model_path}")

    all_results[correlation_key]["pretrain_final"] = {
        "count_a_acc": get_final_metric(history_pretrain, "count_a", "answer_acc"),
        "count_a_loss": get_final_metric(history_pretrain, "count_a", "loss"),
        "count_b_acc": get_final_metric(history_pretrain, "count_b", "answer_acc"),
        "count_b_loss": get_final_metric(history_pretrain, "count_b", "loss"),
        "all_other_avg_acc": get_final_metric(history_pretrain, "all_other_avg", "answer_acc"),
        "all_other_avg_loss": get_final_metric(history_pretrain, "all_other_avg", "loss"),
    }
    
    # ==================== FINETUNING ====================
    print(f"\n--- Finetuning at correlation={correlation} ---")
    
    for conc_idx, concentration in enumerate(concentration_values):
        print(f"\n  Concentration {conc_idx+1}/{len(concentration_values)}: {concentration:.2f}")
        
        conc_key = f"conc_{concentration:.2f}"
        all_results[correlation_key][conc_key] = {}
        
        # Load pretrained model
        model = GPT(config).to(device)
        checkpoint = torch.load(pretrain_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        finetune_optimizer = build_optimizer(
            model.parameters(),
            cfg["optimizer"],
            pretrain_cfg["finetune_lr"],
        )
        
        # concentration is fraction of count_a, rest is count_b
        finetune_weights = [concentration, 1.0 - concentration]  # [count_a, count_b]
        finetune_train_tasks = ["count_a", "count_b"]
        
        history_finetune = train(
            model=model,
            tokenizer=tokenizer,
            device=device,
            steps=pretrain_cfg["finetune_steps"],
            batch_size=pretrain_cfg["finetune_batch_size"],
            lr=pretrain_cfg["finetune_lr"],
            min_lr=pretrain_cfg["finetune_min_lr"],
            warmup_steps=get_warmup_steps(
                pretrain_cfg["finetune_steps"],
                warmup_ratio=pretrain_cfg["finetune_warmup_ratio"],
            ),
            log_interval=pretrain_cfg["finetune_log_interval"],
            task_names=finetune_train_tasks,
            task_weights=finetune_weights,
            pcfg_gen=pcfg,
            task_reg=task_registry,
            chunk_size=chunk_size,
            mask_answer_only=False,
            max_grad_norm=pretrain_cfg["max_grad_norm"],
            optimizer=finetune_optimizer,
            val_datasets=eval_datasets,
            val_batch_size=pretrain_cfg["finetune_batch_size"],
            log_prefix=f"Finetune[corr={correlation:.2f},conc={concentration:.2f}]",
            use_lr_schedule=True,
            metrics=cfg.get("metrics"),
            data_pools=pools,
            correlation=correlation,
        )
        
        # Save finetuned model
        finetune_model_path = os.path.join(
            models_dir,
            f"finetune_corr_{correlation:.2f}_conc_{concentration:.2f}.pth",
        )
        finetune_history_path = os.path.join(
            histories_dir,
            f"finetune_corr_{correlation:.2f}_conc_{concentration:.2f}_history.pth",
        )
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': finetune_optimizer.state_dict(),
            'correlation': correlation,
            'concentration': concentration,
            'stage': 'finetune',
        }
        torch.save(checkpoint, finetune_model_path)
        torch.save(history_finetune, finetune_history_path)

        reverse_optimizer = build_optimizer(
            model.parameters(),
            cfg["optimizer"],
            pretrain_cfg["reverse_lr"],
        )
        reverse_history = train(
            model=model,
            tokenizer=tokenizer,
            device=device,
            steps=pretrain_cfg["reverse_steps"],
            batch_size=pretrain_cfg["reverse_batch_size"],
            lr=pretrain_cfg["reverse_lr"],
            min_lr=pretrain_cfg["reverse_min_lr"],
            warmup_steps=get_warmup_steps(
                pretrain_cfg["reverse_steps"],
                warmup_ratio=pretrain_cfg["reverse_warmup_ratio"],
            ),
            log_interval=pretrain_cfg["reverse_log_interval"],
            task_names=pretrain_tasks,
            task_weights=pretrain_task_weights,
            pcfg_gen=pcfg,
            task_reg=task_registry,
            chunk_size=chunk_size,
            mask_answer_only=False,
            max_grad_norm=pretrain_cfg["max_grad_norm"],
            optimizer=reverse_optimizer,
            val_datasets=eval_datasets,
            val_batch_size=pretrain_cfg["reverse_batch_size"],
            log_prefix=f"Reverse[corr={correlation:.2f},conc={concentration:.2f}]",
            use_lr_schedule=False,
            metrics=cfg.get("metrics"),
            data_pools=pools,
            correlation=correlation,
        )

        reverse_model_path = os.path.join(
            models_dir,
            f"reverse_corr_{correlation:.2f}_conc_{concentration:.2f}.pth",
        )
        reverse_history_path = os.path.join(
            histories_dir,
            f"reverse_corr_{correlation:.2f}_conc_{concentration:.2f}_history.pth",
        )
        reverse_checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': reverse_optimizer.state_dict(),
            'correlation': correlation,
            'concentration': concentration,
            'stage': 'reverse',
        }
        torch.save(reverse_checkpoint, reverse_model_path)
        torch.save(reverse_history, reverse_history_path)
        
        all_results[correlation_key][conc_key]["finetune_final"] = {
            "count_a_acc": get_final_metric(history_finetune, "count_a", "answer_acc"),
            "count_a_loss": get_final_metric(history_finetune, "count_a", "loss"),
            "count_b_acc": get_final_metric(history_finetune, "count_b", "answer_acc"),
            "count_b_loss": get_final_metric(history_finetune, "count_b", "loss"),
            "all_other_avg_acc": get_final_metric(history_finetune, "all_other_avg", "answer_acc"),
            "all_other_avg_loss": get_final_metric(history_finetune, "all_other_avg", "loss"),
        }
        all_results[correlation_key][conc_key]["reverse_final"] = {
            "count_a_acc": get_final_metric(reverse_history, "count_a", "answer_acc"),
            "count_a_loss": get_final_metric(reverse_history, "count_a", "loss"),
            "count_b_acc": get_final_metric(reverse_history, "count_b", "answer_acc"),
            "count_b_loss": get_final_metric(reverse_history, "count_b", "loss"),
            "all_other_avg_acc": get_final_metric(reverse_history, "all_other_avg", "answer_acc"),
            "all_other_avg_loss": get_final_metric(reverse_history, "all_other_avg", "loss"),
        }
        
        print(f"    Saved finetuned model: {finetune_model_path}")
        print(f"    Saved reverse model: {reverse_model_path}")

# Save summary results
summary_path = os.path.join(results_dir, "experiment_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved experiment summary: {summary_path}")

print(f"\n{'='*80}")
print(f"Experiment completed!")
print(f"Results saved to: {results_dir}")
print(f"{'='*80}")
