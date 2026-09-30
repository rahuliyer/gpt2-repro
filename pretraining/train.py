"""Pretrain GPT-2 on a tokenized dataset file."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Sampler
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
    checkpoint_interval: int = 1000
    keep_last_n: int = 3
    checkpoint_name: str = "gpt2"
    wandb_project: str = "gpt2-repro-test"


def run_directory(checkpoint_dir, config, now=None):
    """Build a timestamped directory so a fresh run never reuses another's."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(checkpoint_dir) / f"{config.checkpoint_name}_{stamp}"


def state_checkpoint_path(run_dir, step):
    """Path of the full-training-state checkpoint for `step`."""
    return Path(run_dir) / "checkpoints" / f"step_{step:06d}.pt"


def model_checkpoint_path(run_dir, config, step):
    """Path of the final safetensors weights, tagged with the step that produced them."""
    return Path(run_dir) / f"{config.checkpoint_name}_step_{step:06d}.safetensors"


def find_latest_checkpoint(checkpoint_dir, config):
    """Return the newest state checkpoint under `checkpoint_dir`, or None.

    Run directories carry a sortable %Y%m%d_%H%M%S stamp and step files are
    zero padded, so lexicographic order is chronological then step order.
    """
    pattern = f"{config.checkpoint_name}_*/checkpoints/step_*.pt"
    checkpoints = sorted(Path(checkpoint_dir).glob(pattern))
    return checkpoints[-1] if checkpoints else None


class OffsetSampler(Sampler):
    """Sequential order starting at `offset`, wrapping to cover every index once.

    Resuming mid-file needs the data stream to pick up where it stopped; without
    it a resumed run re-trains on the start of the file and may never reach the
    end. Generating indices lazily keeps this O(1) in memory, which matters when
    the dataset has millions of examples.
    """

    def __init__(self, length, offset=0):
        self.length = length
        self.offset = offset % length if length else 0

    def __iter__(self):
        for i in range(self.length):
            yield (self.offset + i) % self.length

    def __len__(self):
        return self.length


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


def build_dataloader(dataset_path, config, offset=0):
    """Open a token file and wrap it in a dataloader.

    `offset` starts iteration at that example index instead of 0, which is how
    a resumed run picks the data stream back up. At offset 0 the order is
    identical to plain sequential iteration.
    """
    dataset = FineWebDataset(str(dataset_path), config.context_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=OffsetSampler(len(dataset), offset),
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    return dataset, dataloader


# Changing any of these invalidates the restored step counter, so a resume
# cannot silently continue with a different value.
INCOMPATIBLE_ON_RESUME = ("total_batch_size", "batch_size", "context_len")


def check_resume_config(saved_config, config):
    """Raise on config changes that break resume, warn on the rest."""
    current = asdict(config)
    conflicts = [
        f"{field}: checkpoint has {saved_config[field]!r}, config has {current[field]!r}"
        for field in INCOMPATIBLE_ON_RESUME
        if field in saved_config and saved_config[field] != current[field]
    ]
    if conflicts:
        raise ValueError(
            "cannot resume, these settings must match the checkpoint:\n  "
            + "\n  ".join(conflicts)
        )

    changed = [
        f"{field}: {saved_config[field]!r} -> {current[field]!r}"
        for field in sorted(saved_config)
        if field not in INCOMPATIBLE_ON_RESUME
        and field in current
        and saved_config[field] != current[field]
    ]
    for line in changed:
        print(f"warning: config changed since the checkpoint: {line}")


def prune_checkpoints(run_dir, keep_last_n):
    """Delete all but the newest `keep_last_n` state checkpoints of this run.

    Only touches `step_*.pt` inside this run's own `checkpoints/` directory, so
    the final `.safetensors` weights and other runs are never at risk. A
    non-positive `keep_last_n` keeps everything.
    """
    if keep_last_n <= 0:
        return []
    # Step numbers are zero padded, so lexicographic order is step order.
    checkpoints = sorted((Path(run_dir) / "checkpoints").glob("step_*.pt"))
    stale = checkpoints[:-keep_last_n]
    for path in stale:
        path.unlink()
    return stale


def save_state(path, model, optimizer, config, step, grad_accum_steps):
    """Write everything needed to continue this run later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        # The uncompiled module: torch.compile prefixes state_dict keys with
        # "_orig_mod.", which will not load back into a plain GPT2.
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "examples_consumed": step * grad_accum_steps * config.batch_size,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
    }
    torch.save(state, path)
    return path


def load_state(path):
    """Read a state checkpoint written by `save_state`."""
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng_state(state):
    """Put the RNGs back where the checkpoint left them."""
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"].cpu().to(torch.uint8))
    cuda_state = state.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)

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


def train(config, train_path, val_path, run_dir, state=None):
    """Train a model and return it with the final train and validation losses.

    `state` is a checkpoint from `load_state`; when given, training continues
    from its step with the optimizer, RNG and data position restored.
    """
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

    step = 0
    examples_consumed = 0
    if state is not None:
        check_resume_config(state["config"], config)
        # Load into the plain module before compiling, so the key names match.
        model.load_state_dict(state["model"])
        step = state["step"]
        examples_consumed = state.get("examples_consumed", 0)
        # Drop the CPU copy now that it lives on the device.
        del state["model"]

    optimizer = build_optimizer(model, config)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        del state["optimizer"]
        restore_rng_state(state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Compiling model...")
    compiled_model = torch.compile(model)

    # Peek at the dataset length to turn examples consumed into an offset,
    # then build the loader once with that offset already applied.
    probe = FineWebDataset(str(train_path), config.context_len)
    offset = examples_consumed % len(probe) if len(probe) else 0
    del probe
    dataset, dataloader = build_dataloader(train_path, config, offset=offset)
    # Validation always scores the same prefix, so it never takes an offset.
    _, val_dataloader = build_dataloader(val_path, config)

    precision = "bf16 (autocast)" if use_bf16 else "fp32"
    print(
        f"Training on {device} | {precision} | {len(dataset):,} examples "
        f"| {config.total_batch_size:,} tok/step "
        f"({grad_accum_steps} x {config.batch_size} x {config.context_len}) "
        f"| {config.max_steps:,} steps max"
    )

    if state is not None:
        print(
            f"Resuming at step {step:,} "
            f"(example offset {offset:,} of {len(dataset):,})"
        )

    model.train()
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

        if step % config.checkpoint_interval == 0:
            written = save_state(
                state_checkpoint_path(run_dir, step),
                model,
                optimizer,
                config,
                step,
                grad_accum_steps,
            )
            print(f"Saved state checkpoint to {written}")
            removed = prune_checkpoints(run_dir, config.keep_last_n)
            if removed:
                print(
                    f"Pruned {len(removed)} old checkpoint(s), "
                    f"keeping the newest {config.keep_last_n}"
                )

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

    # The loop already wrote this step if it landed on the interval.
    final_state = state_checkpoint_path(run_dir, step)
    if not final_state.exists():
        save_state(final_state, model, optimizer, config, step, grad_accum_steps)
        print(f"Saved state checkpoint to {final_state}")
        prune_checkpoints(run_dir, config.keep_last_n)

    return model, loss_value, final_val_loss, step


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
        help="Directory holding per-run checkpoint directories.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue the most recent run found under --checkpoint-dir.",
    )
    args = parser.parse_args(argv)

    config = TrainingConfig()

    state = None
    if args.resume:
        latest = find_latest_checkpoint(args.checkpoint_dir, config)
        if latest is None:
            print(
                f"error: --resume found no checkpoint under {args.checkpoint_dir}",
                file=sys.stderr,
            )
            return 1
        print(f"Resuming from {latest}")
        state = load_state(latest)
        # A resumed run continues inside the directory it came from.
        run_dir = latest.parent.parent
    else:
        run_dir = run_directory(args.checkpoint_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reusing the run id keeps one continuous curve across restarts.
    wandb.init(
        project=config.wandb_project,
        config=asdict(config),
        id=state.get("wandb_run_id") if state else None,
        resume="allow",
    )

    try:
        model, _, final_val_loss, step = train(
            config, args.train_dataset, args.val_dataset, run_dir, state
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        wandb.finish(exit_code=1)
        return 1

    wandb.summary["final_val_loss"] = final_val_loss

    checkpoint = model_checkpoint_path(run_dir, config, step)
    model.save(str(checkpoint))
    print(f"Saved model to {checkpoint}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
