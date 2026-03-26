"""
Pre-tokenized dataset and fast DataLoader for PCFG training.

Instead of generating + tokenizing examples on-the-fly every step,
this module pre-builds a large pool of tokenized examples once,
then serves shuffled batches via a standard DataLoader with workers.
"""

import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pcfg_gen import (
    CharTokenizer,
    PCFGGenerator,
    TaskRegistry,
    format_example,
)


class PreTokenizedDataset(Dataset):
    """A large pool of pre-tokenized PCFG examples, ready to serve."""

    def __init__(
        self,
        pools: Dict[str, list],
        task_names: List[str],
        task_weights: List[float],
        task_reg: TaskRegistry,
        tokenizer: CharTokenizer,
        correlation: float = 0.0,
        n_examples: int = 100_000,
        max_length: int = 512,
        mask_answer_only: bool = True,
        seed: int = 42,
    ):
        self.max_length = max_length
        self.mask_answer_only = mask_answer_only
        self.tokenizer = tokenizer

        rng = random.Random(seed)

        # Pre-generate and tokenize all examples
        self.encoded = []
        self.answer_positions = []

        for _ in range(n_examples):
            # Pick pool based on correlation
            if correlation > 0.0 and rng.random() < correlation:
                pcfg_string = rng.choice(pools["correlated"])
            else:
                pcfg_string = rng.choice(pools["uncorrelated"])

            task_name = rng.choices(task_names, weights=task_weights, k=1)[0]
            task_def, answer = task_reg.apply_task(task_name, pcfg_string)
            token_list = format_example(pcfg_string, task_def, answer)

            ids = tokenizer.encode(token_list)

            # Find answer positions
            try:
                art_idx = token_list.index("[ART]")
                eos_idx = token_list.index("[EOS]")
                answer_pos = list(range(art_idx + 1, eos_idx))
            except ValueError:
                answer_pos = []

            if len(ids) > max_length:
                ids = ids[:max_length]
                answer_pos = [p for p in answer_pos if p < max_length]

            self.encoded.append(ids)
            self.answer_positions.append(answer_pos)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        tokens = self.encoded[idx]
        answer_pos = self.answer_positions[idx]

        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        answer_pos_shifted = [
            p - 1 for p in answer_pos if p > 0 and p - 1 < len(target_ids)
        ]

        if self.mask_answer_only:
            loss_mask = torch.full((len(target_ids),), -100, dtype=torch.long)
            for pos in answer_pos_shifted:
                if 0 <= pos < len(target_ids):
                    loss_mask[pos] = target_ids[pos]
            target_tensor = loss_mask
        else:
            target_tensor = torch.tensor(target_ids, dtype=torch.long)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": target_tensor,
            "answer_positions": answer_pos_shifted,
        }


def fast_collate_fn(batch, pad_id: int):
    """Collate with padding — no tokenizer reference needed at collate time."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    target_ids = []
    answer_positions = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_id))
        target_ids.append(F.pad(item["target_ids"], (0, pad_len), value=-100))
        answer_positions.append(item["answer_positions"])

    return {
        "input_ids": torch.stack(input_ids),
        "target_ids": torch.stack(target_ids),
        "answer_positions": answer_positions,
    }


def build_fast_loader(
    pools: Dict[str, list],
    task_names: List[str],
    task_weights: List[float],
    task_reg: TaskRegistry,
    tokenizer: CharTokenizer,
    correlation: float = 0.0,
    n_examples: int = 100_000,
    max_length: int = 512,
    mask_answer_only: bool = True,
    batch_size: int = 96,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build a pre-tokenized DataLoader ready for training.

    Args:
        pools: Dict with 'correlated' and 'uncorrelated' string lists.
        task_names: Task names to sample from.
        task_weights: Sampling weights for tasks.
        task_reg: TaskRegistry instance.
        tokenizer: CharTokenizer instance.
        correlation: Fraction of examples from correlated pool.
        n_examples: Total pre-generated examples (dataset cycles through these).
        max_length: Max sequence length.
        mask_answer_only: If True, only compute loss on answer tokens.
        batch_size: Batch size.
        num_workers: DataLoader workers for parallel batch serving.
        seed: RNG seed for reproducible data generation.

    Returns:
        DataLoader that yields batches indefinitely (when used with iter()).
    """
    dataset = PreTokenizedDataset(
        pools=pools,
        task_names=task_names,
        task_weights=task_weights,
        task_reg=task_reg,
        tokenizer=tokenizer,
        correlation=correlation,
        n_examples=n_examples,
        max_length=max_length,
        mask_answer_only=mask_answer_only,
        seed=seed,
    )

    pad_id = tokenizer.pad_id

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=lambda batch: fast_collate_fn(batch, pad_id),
    )

    return loader
