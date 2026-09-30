"""Pretrain GPT-2 on a tokenized dataset file."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from data import FineWebDataset
from model import GPT2, GPT2SmallConfig


@dataclass
class TrainingConfig:
    batch_size: int = 2
    context_len: int = 1024
    learning_rate: float = 6e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    dropout: float = 0.0
    max_steps: int = 5_000
    num_epochs: int = 1_000
    num_workers: int = 1
    seed: int = 1337
    log_interval: int = 1


def train(config, dataset_path):
    """Train a model on the tokenized dataset and return it with the final loss."""
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT2(
        GPT2SmallConfig(context_len=config.context_len, dropout=config.dropout)
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    dataset = FineWebDataset(str(dataset_path), config.context_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    print(f"Training on {device} | {len(dataset):,} examples | {config.max_steps:,} steps max")

    model.train()
    step = 0
    loss_value = float("nan")
    started = time.monotonic()
    step_started = started

    for epoch in range(config.num_epochs):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            step += 1
            loss_value = loss.item()
            now = time.monotonic()
            if step % config.log_interval == 0:
                print(
                    f"Step {step:,}/{config.max_steps:,} | loss {loss_value:.4f} "
                    f"| {now - step_started:.2f}s/step | {now - started:.0f}s elapsed"
                )
            step_started = now

            if step >= config.max_steps:
                break

        if step >= config.max_steps:
            break

    print(f"Completed: {step:,} steps over {epoch + 1} epoch(s), final loss {loss_value:.4f}")
    return model, loss_value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pretrain GPT-2 on a tokenized dataset file."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the uint16 token file produced by data/preprocess.py.",
    )
    args = parser.parse_args(argv)

    config = TrainingConfig()
    train(config, args.dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
