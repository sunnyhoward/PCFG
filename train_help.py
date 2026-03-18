from pcfg_gen import CharTokenizer, PCFGGenerator, TaskRegistry, format_example, PCFGDataset, collate_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
from collections import defaultdict
import math


def calculate_answer_accuracy(logits, targets, answer_positions_batch):
    """Calculate accuracy on answer tokens only.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        targets: (batch, seq_len) - target token IDs (may include -100)
        answer_positions_batch: List of lists containing answer positions for each example
    
    Returns:
        accuracy: Fraction of correctly predicted answer tokens
        total: Total number of answer tokens
    """
    predictions = torch.argmax(logits, dim=-1)  # (batch, seq_len)
    
    correct = 0
    total = 0
    
    # Iterate through batch
    for i in range(targets.size(0)):
        positions = answer_positions_batch[i]
        if not positions:
            continue
        
        # Get predictions and targets at answer positions
        answer_preds = predictions[i][positions]
        answer_targets = targets[i][positions]
        
        # Filter out any masked targets if present
        valid_mask = answer_targets != -100
        if valid_mask.sum() == 0:
            continue
        
        answer_preds = answer_preds[valid_mask]
        answer_targets = answer_targets[valid_mask]
        
        # Count correct predictions
        correct += (answer_preds == answer_targets).sum().item()
        total += valid_mask.sum().item()
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total

# ----------------------------
# Online data sampling helpers
# ----------------------------

def sample_batch(batch_size: int,
                 task_names: List[str],
                 task_weights: List[float],
                 pcfg_gen: PCFGGenerator,
                 task_reg: TaskRegistry,
                 tokenizer: CharTokenizer,
                 chunk_size: int = 250,
                 mask_answer_only: bool = True,
                 use_correlation: bool = False,
                 data_pools: Dict = None,
                 correlation: float = 0.0):
    """Sample a batch online.

    Args:
        data_pools: Optional dict with keys ``'correlated'`` and
            ``'uncorrelated'``, each a list of pre-generated PCFG strings
            (built with ``pcfg_gen.build_pools``).  When provided, strings are
            sampled from the pools rather than generated on the fly.
        correlation: Fraction of examples drawn from the correlated pool.
            ``0.0`` → all uncorrelated, ``1.0`` → all correlated.
            Only used when ``data_pools`` is not None.
        use_correlation: Deprecated – has no effect when ``data_pools`` is
            provided.  Kept for backwards compatibility.
    """
    examples = []
    for _ in range(batch_size):
        if data_pools is not None:
            if correlation > 0.0 and random.random() < correlation:
                pcfg_string = random.choice(data_pools['correlated'])
            else:
                pcfg_string = random.choice(data_pools['uncorrelated'])
        else:
            pcfg_string = pcfg_gen.generate_chunk(chunk_size)
        task_name = random.choices(task_names, weights=task_weights, k=1)[0]
        task_def, answer = task_reg.apply_task(task_name, pcfg_string)
        examples.append(format_example(pcfg_string, task_def, answer))

    batch_dataset = PCFGDataset(examples, tokenizer, max_length=512, mask_answer_only=mask_answer_only)
    batch = collate_fn([batch_dataset[i] for i in range(len(batch_dataset))], tokenizer)
    return batch

# ----------------------------
# LR schedule helpers
# ----------------------------

def get_cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# ----------------------------
# Training helpers
# ----------------------------

def _build_val_loaders(
    val_datasets,
    batch_size: int,
    tokenizer: CharTokenizer,
    collate_fn_override: Callable = None,
):
    if val_datasets is None:
        return {}

    collate = collate_fn_override or collate_fn

    if isinstance(val_datasets, DataLoader):
        return {"val": val_datasets}

    if isinstance(val_datasets, dict):
        val_items = val_datasets.items()
    elif isinstance(val_datasets, (list, tuple)):
        val_items = [(f"val_{i}", ds) for i, ds in enumerate(val_datasets)]
    else:
        val_items = [("val", val_datasets)]

    val_loaders = {}
    for name, ds in val_items:
        if isinstance(ds, DataLoader):
            val_loaders[name] = ds
        else:
            val_loaders[name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch, tok=tokenizer: collate(batch, tok),
            )
    return val_loaders


def _evaluate_loader(model, loader, device, metrics_set):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            answer_positions = batch['answer_positions']
            logits, loss = model(input_ids, target_ids)
            total_loss += loss.item()
            if "answer_acc" in metrics_set:
                acc, total = calculate_answer_accuracy(logits, target_ids, answer_positions)
                total_correct += acc * total
                total_tokens += total

    avg_loss = total_loss / max(1, len(loader))
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc


def train(
    model,
    tokenizer: CharTokenizer,
    device,
    steps: int,
    batch_size: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    log_interval: int,
    task_names: List[str],
    task_weights: List[float],
    pcfg_gen: PCFGGenerator,
    task_reg: TaskRegistry,
    chunk_size: int = 250,
    mask_answer_only: bool = True,
    use_correlation: bool = False,
    max_grad_norm: float = 1.0,
    optimizer=None,
    val_datasets=None,
    val_batch_size=None,
    collate_fn_override=None,
    log_prefix: str = "Train",
    use_lr_schedule: bool = True,
    lr_schedule_total_steps=None,
    lr_schedule_start_step: int = 0,
    metrics=None,
    data_pools=None,
    correlation: float = 0.0,
):
    """Generic training loop with optional multi-set validation.

    Args:
        mask_answer_only: If True, loss is computed only on answer tokens.
        data_pools: Optional dict with ``'correlated'`` and ``'uncorrelated'``
            PCFG string lists (from ``build_pools``).  When provided, training
            strings are sampled from the pools rather than generated on the fly.
        correlation: Fraction of training examples drawn from the correlated
            pool.  ``0.0`` = all natural, ``1.0`` = all correlated.
        use_correlation: Deprecated – ignored when ``data_pools`` is provided.
        val_datasets: Optional validation datasets/loaders.
    """
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    val_bs = val_batch_size or batch_size
    val_loaders = _build_val_loaders(val_datasets, val_bs, tokenizer, collate_fn_override)

    metrics_set = set(metrics or ["loss", "answer_acc"])

    history = {
        "steps": [],
        "val": {name: {} for name in val_loaders},
    }

    if "loss" in metrics_set:
        history["train_loss"] = []
        for name in val_loaders:
            history["val"][name]["loss"] = []

    if "answer_acc" in metrics_set:
        history["train_answer_acc"] = []
        for name in val_loaders:
            history["val"][name]["answer_acc"] = []

    for step in range(1, steps + 1):
        if use_lr_schedule:
            schedule_total = lr_schedule_total_steps or steps
            schedule_step = lr_schedule_start_step + step
            cur_lr = get_cosine_lr(schedule_step, schedule_total, lr, min_lr, warmup_steps)
        else:
            cur_lr = lr
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr

        batch = sample_batch(
            batch_size=batch_size,
            task_names=task_names,
            task_weights=task_weights,
            pcfg_gen=pcfg_gen,
            task_reg=task_reg,
            chunk_size=chunk_size,
            mask_answer_only=mask_answer_only,
            use_correlation=use_correlation,
            tokenizer=tokenizer,
            data_pools=data_pools,
            correlation=correlation,
        )

        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        answer_positions = batch['answer_positions']

        model.train()
        logits, loss = model(input_ids, target_ids)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        train_acc = None
        if "answer_acc" in metrics_set:
            acc, total = calculate_answer_accuracy(logits, target_ids, answer_positions)
            train_acc = acc if total > 0 else 0.0

        if step % log_interval == 0:
            history["steps"].append(step)
            if "loss" in metrics_set:
                history["train_loss"].append(loss.item())
            if "answer_acc" in metrics_set:
                history["train_answer_acc"].append(train_acc)

            log_parts = [
                f"{log_prefix} step {step}/{steps}",
                f"LR: {cur_lr:.6f}",
            ]

            if "loss" in metrics_set:
                log_parts.insert(1, f"Loss: {loss.item():.4f}")
            if "answer_acc" in metrics_set:
                log_parts.insert(2, f"Answer Acc: {train_acc:.4f}")

            if val_loaders:
                for name, loader in val_loaders.items():
                    v_loss, v_acc = _evaluate_loader(model, loader, device, metrics_set)
                    if "loss" in metrics_set:
                        history["val"][name]["loss"].append(v_loss)
                        log_parts.append(f"{name} Loss: {v_loss:.4f}")
                    if "answer_acc" in metrics_set:
                        history["val"][name]["answer_acc"].append(v_acc)
                        log_parts.append(f"{name} Acc: {v_acc:.4f}")

                # Backwards-compatible keys for single validation set
                if len(val_loaders) == 1:
                    only_name = next(iter(val_loaders))
                    if "loss" in metrics_set:
                        history.setdefault("val_loss", []).append(
                            history["val"][only_name]["loss"][-1]
                        )
                    if "answer_acc" in metrics_set:
                        history.setdefault("val_answer_acc", []).append(
                            history["val"][only_name]["answer_acc"][-1]
                        )

            print(" | ".join(log_parts))

    return history


def build_task_weights(task_names: List[str], operand_probs: Dict[str, float]) -> List[float]:
    weights = []
    for name in task_names:
        if name.startswith('count_') and name in ['count_a', 'count_b', 'count_c']:
            tok = name.split('_')[1]
            weights.append(operand_probs.get(tok, 1.0))
        elif name.startswith('index_') and name in ['index_a', 'index_b', 'index_c']:
            tok = name.split('_')[1]
            weights.append(operand_probs.get(tok, 1.0))
        else:
            weights.append(1.0)
    return weights
