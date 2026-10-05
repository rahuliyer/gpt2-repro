"""Fine-tune GPT-2 as an AG News classifier."""

import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
import time

from safetensors.torch import load_model
import tiktoken
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import wandb

from classification import get_datasets
from classification.classifier import get_last_logits, load_pretrained_classifier
from utils import build_optimizer, get_lr, run_directory, save_config


@dataclass
class TrainingConfig:
    checkpoint_name: str = "classifier"
    head: str = "mlp"  # "mlp" or "linear"
    unfreeze: str = "none"  # "none", "last_block" or "all"
    batch_size: int = 512
    eval_batch_size: int = 256
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 20
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    fused_optimizer: bool = True
    n_epochs: int = 3
    max_steps: int | None = None  # None derives the count from n_epochs
    val_interval: int = 100
    log_interval: int = 10
    max_length: int = 128
    val_split: float = 0.1
    seed: int = 1337
    wandb_project: str = "gpt2-classification-sft"


def resolve_max_steps(config, num_examples):
    """Optimizer steps in `config.n_epochs` passes, unless `max_steps` is set."""
    if config.max_steps is not None:
        return config.max_steps
    return config.n_epochs * math.ceil(num_examples / config.batch_size)


def best_checkpoint_path(run_dir):
    """Path of the weights with the lowest validation loss so far."""
    return Path(run_dir) / "best.safetensors"


@torch.no_grad()
def evaluate(model, dataset, config, device):
    """Return the mean loss and accuracy over every example in `dataset`."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size)

    total_loss = 0.0
    correct = 0
    for x, lengths, y in dataloader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = get_last_logits(model(x), lengths)

        # Sum rather than average per batch, so a short final batch does not
        # carry the same weight as a full one.
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()

    model.train()
    return total_loss / len(dataset), correct / len(dataset)


def train(model, config, datasets, run_dir, device):
    """Train `model`, then score the best checkpoint on the test split.

    `datasets` is the (train, val, test) triple. The weights with the lowest
    validation loss are kept in `run_dir` and are the ones the test numbers
    come from, not whatever the last step produced.
    """
    torch.manual_seed(config.seed)
    train_ds, val_ds, test_ds = datasets

    train_dataloader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    steps_per_epoch = len(train_dataloader)
    config.max_steps = resolve_max_steps(config, len(train_ds))
    total_epochs = math.ceil(config.max_steps / steps_per_epoch)

    model = model.to(device)
    optimizer = build_optimizer(model, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Training on {device} | {config.head} head, unfreeze {config.unfreeze} "
        f"| {trainable:,} of {total:,} params trainable "
        f"| {len(train_ds):,} train / {len(val_ds):,} val / {len(test_ds):,} test examples "
        f"| {config.max_steps:,} steps ({total_epochs} epoch(s) of {steps_per_epoch:,})"
    )

    best_path = best_checkpoint_path(run_dir)
    best_val_loss = float("inf")
    best_step = None

    model.train()
    step = 0
    started = time.monotonic()
    window_started = started
    window_loss = 0.0
    window_correct = 0
    window_examples = 0
    window_steps = 0

    while step < config.max_steps:
        for x, lengths, y in train_dataloader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            logits = get_last_logits(model(x), lengths)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()

            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            step += 1

            window_loss += loss.item() * len(y)
            window_correct += (logits.argmax(dim=-1) == y).sum().item()
            window_examples += len(y)
            window_steps += 1

            last_step = step == config.max_steps
            validate = step % config.val_interval == 0 or last_step

            if step % config.log_interval == 0 or validate:
                # A single batch is too noisy to read, so report the mean over
                # every example seen since the previous line.
                now = time.monotonic()
                train_loss = window_loss / window_examples
                train_acc = window_correct / window_examples
                message = (
                    f"Step {step:,}/{config.max_steps:,} "
                    f"| epoch {(step - 1) // steps_per_epoch + 1}/{total_epochs} "
                    f"| loss {train_loss:.4f} | acc {train_acc:.4f} | lr {lr:.2e} "
                    f"| {(now - window_started) / window_steps:.2f}s/step "
                    f"| {now - started:.0f}s elapsed"
                )
                metrics = {
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "train/lr": lr,
                }

                if validate:
                    val_loss, val_acc = evaluate(model, val_ds, config, device)
                    message += f" | val loss {val_loss:.4f} | val acc {val_acc:.4f}"
                    metrics["val/loss"] = val_loss
                    metrics["val/acc"] = val_acc

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_step = step
                        model.save(str(best_path))
                        message += " (best)"

                print(message)
                if wandb.run is not None:
                    wandb.log(metrics, step=step)

                # Restart the clock after validation so it is not counted
                # against the next window's step time.
                window_started = time.monotonic()
                window_loss = 0.0
                window_correct = 0
                window_examples = 0
                window_steps = 0

            if last_step:
                break

    if best_step is None:
        raise ValueError("validation loss was never finite, so no checkpoint was saved")

    print(
        f"Completed: {step:,} steps in {time.monotonic() - started:.0f}s, "
        f"best val loss {best_val_loss:.4f} at step {best_step:,}"
    )

    load_model(model, str(best_path))
    test_loss, test_acc = evaluate(model, test_ds, config, device)
    print(
        f"Test over {len(test_ds):,} examples with the step {best_step:,} checkpoint: "
        f"loss {test_loss:.4f} | acc {test_acc:.4f}"
    )

    return {
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def main(config, argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 as an AG News classifier."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="Directory holding per-run directories with the model and its config.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        choices=(0, 1),
        default=0,
        help="Index of the GPU to train on.",
    )
    args = parser.parse_args(argv)

    device = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"

    tokenizer = tiktoken.get_encoding("gpt2")
    datasets = get_datasets(tokenizer, config.val_split, config.max_length)
    # Resolve before saving and wandb.init so both record the real step count
    # rather than None. train() resolves again and is a no-op once set.
    config.max_steps = resolve_max_steps(config, len(datasets[0]))

    run_dir = run_directory(args.checkpoint_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saved config to {save_config(run_dir, config)}")

    wandb.init(
        project=config.wandb_project,
        name=run_dir.name,
        config=asdict(config),
    )

    model = load_pretrained_classifier(config.head, config.unfreeze)

    try:
        results = train(model, config, datasets, run_dir, device)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        wandb.finish(exit_code=1)
        return 1

    wandb.summary.update(results)
    print(f"Best model is at {best_checkpoint_path(run_dir)}")

    wandb.finish()
    return 0
