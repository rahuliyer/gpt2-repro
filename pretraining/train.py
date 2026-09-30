"""Pretrain GPT-2 on a tokenized dataset file."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from pretraining import FineWebDataset
from model import GPT2, GPT2SmallConfig


@dataclass
class TrainingConfig:
    batch_size: int = 1
    context_len: int = 1024
    total_batch_size: int = 524_288  # 2**19 tokens per optimizer step (GPT-2 paper)
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 100
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    dropout: float = 0.0
    max_steps: int = 5_000
    num_workers: int = 0
    seed: int = 1337
    log_interval: int = 1
    eval_interval: int = 50
    eval_iters: int = 20
    fused_optimizer: bool = False
    checkpoint_name: str = "gpt2"
    wandb_project: str = "gpt2-repro-test"


def checkpoint_path(checkpoint_dir, config, now=None):
    """Build a timestamped checkpoint path so a run never overwrites another."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(checkpoint_dir) / f"{config.checkpoint_name}_{stamp}.safetensors"


def get_lr(step, config):
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


def resolve_grad_accum_steps(config):
    """Return how many micro-batches make up one optimizer step."""
    tokens_per_micro_batch = config.batch_size * config.context_len
    if config.total_batch_size % tokens_per_micro_batch != 0:
        raise ValueError(
            f"total_batch_size {config.total_batch_size:,} is not divisible by "
            f"batch_size * context_len ({tokens_per_micro_batch:,})"
        )
    return config.total_batch_size // tokens_per_micro_batch


def cycle(dataloader):
    """Yield batches forever, restarting the loader when it is exhausted."""
    while True:
        yield from dataloader


def build_dataloader(dataset_path, config):
    """Open a token file and wrap it in a dataloader."""
    dataset = FineWebDataset(str(dataset_path), config.context_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    return dataset, dataloader

def build_optimizer(model, config):
    decay_params = []
    no_decay_params = []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_params,
                "weight_decay": config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ],
        lr=config.max_lr,
        betas=config.betas,
        eps=1e-8,
        fused=config.fused_optimizer
    )

    return optimizer

@torch.no_grad()
def estimate_val_loss(model, val_dataloader, device, use_bf16, max_batches=None):
    """Return the mean validation loss.

    ``max_batches`` caps how much of the validation file is scored; None runs
    a full pass. The loader is not shuffled, so a capped call always scores the
    same prefix, which keeps the periodic estimates comparable step to step.
    """
    model.eval()
    total_loss = 0.0
    batches = 0

    for x, y in val_dataloader:
        if max_batches is not None and batches >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
            )
        total_loss += loss.item()
        batches += 1

    model.train()
    return (total_loss / batches if batches else float("nan")), batches


def train(config, train_path, val_path):
    """Train a model and return it with the final train and validation losses."""
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # including_emulation=False: is_bf16_supported() defaults to True on GPUs
    # (e.g. Turing) that only emulate bf16 in software, with no tensor-core
    # speedup, so require native hardware support instead.
    use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    grad_accum_steps = resolve_grad_accum_steps(config)

    model = GPT2(
        GPT2SmallConfig(context_len=config.context_len, dropout=config.dropout)
    ).to(device)
    optimizer = build_optimizer(model, config)

    print(f"Compiling model...")
    compiled_model = torch.compile(model)

    dataset, dataloader = build_dataloader(train_path, config)
    _, val_dataloader = build_dataloader(val_path, config)

    precision = "bf16 (autocast)" if use_bf16 else "fp32"
    print(
        f"Training on {device} | {precision} | {len(dataset):,} examples "
        f"| {config.total_batch_size:,} tok/step "
        f"({grad_accum_steps} x {config.batch_size} x {config.context_len}) "
        f"| {config.max_steps:,} steps max"
    )

    model.train()
    step = 0
    loss_value = float("nan")
    started = time.monotonic()
    step_started = started
    batches = cycle(dataloader)

    while step < config.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum_steps):
            x, y = next(batches)
            x = x.to(device)
            y = y.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = compiled_model(x)
                # Scale so the accumulated gradient is the mean over the cycle,
                # not the sum, which would inflate the effective learning rate.
                loss = (
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        y.reshape(-1),
                    )
                    / grad_accum_steps
                )

            loss.backward()
            accum_loss += loss.item()

        # clip the fully accumulated gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        step += 1
        loss_value = accum_loss
        now = time.monotonic()
        step_seconds = now - step_started
        tokens_per_second = config.total_batch_size / step_seconds if step_seconds else 0

        val_loss = None
        val_batches = 0
        if step % config.eval_interval == 0:
            val_loss, val_batches = estimate_val_loss(
                model, val_dataloader, device, use_bf16, config.eval_iters
            )

        if step % config.log_interval == 0:
            message = (
                f"Step {step:,}/{config.max_steps:,} | loss {loss_value:.4f} "
                f"| lr {lr:.2e} | grad norm {grad_norm:.4f} "
                f"| {tokens_per_second:,.0f} tok/s "
                f"| {step_seconds:.2f}s/step | {now - started:.0f}s elapsed"
            )
            if val_loss is not None:
                message += f" | val loss {val_loss:.4f} ({val_batches:,} batches)"
            print(message)

        if wandb.run is not None:
            metrics = {
                "train/loss": loss_value,
                "train/lr": lr,
                "train/grad_norm": float(grad_norm),
                "train/tokens_per_sec": tokens_per_second,
                "train/tokens": step * config.total_batch_size,
            }
            if val_loss is not None:
                metrics["val/loss"] = val_loss
            wandb.log(metrics, step=step)

        step_started = now

    total_tokens = step * config.total_batch_size
    print(
        f"Completed: {step:,} steps ({total_tokens:,} tokens), "
        f"final train loss {loss_value:.4f}"
    )

    # The periodic numbers above are a cheap prefix sample; score the whole
    # validation file once at the end for the number worth reporting.
    final_val_loss, val_batches = estimate_val_loss(
        model, val_dataloader, device, use_bf16
    )
    print(
        f"Final validation loss over {val_batches:,} batches "
        f"({val_batches * config.batch_size * config.context_len:,} tokens): "
        f"{final_val_loss:.4f}"
    )

    return model, loss_value, final_val_loss


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pretrain GPT-2 on a tokenized dataset file."
    )
    parser.add_argument(
        "--train-dataset",
        required=True,
        type=Path,
        help="Path to the uint16 training token file from pretraining/preprocess.py.",
    )
    parser.add_argument(
        "--val-dataset",
        required=True,
        type=Path,
        help="Path to the uint16 validation token file from pretraining/preprocess.py.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Directory to write the timestamped safetensors checkpoint into.",
    )
    args = parser.parse_args(argv)

    config = TrainingConfig()
    wandb.init(project=config.wandb_project, config=asdict(config))

    model, _, final_val_loss = train(config, args.train_dataset, args.val_dataset)
    wandb.summary["final_val_loss"] = final_val_loss

    checkpoint = checkpoint_path(args.checkpoint_dir, config)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(checkpoint))
    print(f"Saved model to {checkpoint}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
