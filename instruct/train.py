"""Fine-tune GPT-2 to chat on SmolTalk."""

import argparse
from dataclasses import asdict, dataclass
from functools import partial
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import wandb

from inference import generate_text
from instruct import SmolTalkDataset, get_tokenizer
from instruct.chat import encode_prompt
from instruct.chat_model import build_model
from instruct.smalltalk_dataset import collate_fn
from instruct.tokenizer import ASSISTANT_END_TOKEN, IGNORE_TOKEN_ID
from utils import build_optimizer, get_lr, run_directory, save_config


@dataclass
class TrainingConfig:
    checkpoint_name: str = "instruct"
    model_size: str = "small"  # a key of MODEL_SIZES
    batch_size: int = 2  # conversations per micro-batch
    grad_accum_steps: int = 16  # micro-batches per optimizer step
    eval_batch_size: int = 4
    num_workers: int = 4  # tokenize training batches off the main process
    max_lr: float = 5e-5
    min_lr: float = 5e-6
    warmup_steps: int = 100
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    fused_optimizer: bool = True
    n_epochs: int = 1
    max_steps: int | None = None  # None derives the count from n_epochs
    val_interval: int = 250
    log_interval: int = 10
    max_length: int = 1024
    val_examples: int = 1000  # prefix of the test split scored at each validation
    sample_max_tokens: int = 128
    seed: int = 1337
    wandb_project: str = "gpt2-instruct-sft"


# Fixed prompts whose replies are printed at every validation, to watch the
# model pick up the chat format.
SAMPLE_CHATS = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    [
        {"role": "user", "content": "How do I make a cup of tea?"},
    ],
    [
        {"role": "user", "content": "Write a haiku about the ocean."},
    ],
]


def resolve_max_steps(config, num_examples):
    """Optimizer steps in `config.n_epochs` passes, unless `max_steps` is set."""
    if config.max_steps is not None:
        return config.max_steps
    examples_per_step = config.batch_size * config.grad_accum_steps
    return config.n_epochs * math.ceil(num_examples / examples_per_step)


def best_checkpoint_path(run_dir):
    """Path of the weights with the lowest validation loss so far."""
    return Path(run_dir) / "best.safetensors"


def cycle(dataloader):
    """Yield batches forever, reshuffling on every pass."""
    while True:
        yield from dataloader


def count_targets(y):
    return (y != IGNORE_TOKEN_ID).sum().item()


@torch.no_grad()
def evaluate(model, dataloader, device, use_bf16):
    """Return the mean loss per assistant token over every batch in `dataloader`."""
    model.eval()

    total_loss = 0.0
    total_targets = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=IGNORE_TOKEN_ID,
                reduction="sum",
            )
        total_loss += loss.item()
        total_targets += count_targets(y)

    model.train()
    return total_loss / total_targets


def sample_replies(model, tokenizer, config, step):
    """Generate a reply to each of SAMPLE_CHATS, print them and log them to wandb."""
    stop_token_id = tokenizer.encode_single_token(ASSISTANT_END_TOKEN)
    rows = []
    for messages in SAMPLE_CHATS:
        prompt = encode_prompt(tokenizer, messages)
        text = generate_text(
            model,
            tokenizer,
            prompt,
            max_length=config.sample_max_tokens,
            stop_token_id=stop_token_id,
        )
        reply = text[len(tokenizer.decode(prompt)):]
        rows.append([step, messages[-1]["content"], reply])

        for message in messages:
            print(f"  [{message['role']}] {message['content']}")
        print(f"  [assistant] {reply}\n")

    # generate_text leaves the model in eval mode.
    model.train()

    if wandb.run is not None:
        table = wandb.Table(columns=["step", "prompt", "reply"], data=rows)
        wandb.log({"samples": table}, step=step)


def train(model, tokenizer, config, datasets, run_dir, device):
    """Train `model`, keeping the weights with the lowest validation loss.

    `datasets` is the (train, val) pair.
    """
    torch.manual_seed(config.seed)
    # including_emulation=False: is_bf16_supported() defaults to True on GPUs
    # (e.g. Turing) that only emulate bf16 in software, with no tensor-core
    # speedup, so require native hardware support instead.
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(
        including_emulation=False
    )
    train_ds, val_ds = datasets
    collate = partial(collate_fn, pad_id=tokenizer.eot_token)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    # Unshuffled, so every validation scores the same conversations.
    val_dataloader = DataLoader(
        val_ds, batch_size=config.eval_batch_size, collate_fn=collate
    )
    config.max_steps = resolve_max_steps(config, len(train_ds))
    examples_per_step = config.batch_size * config.grad_accum_steps

    model = model.to(device)
    optimizer = build_optimizer(model, config)

    print("Compiling model...")
    # Used for the training and validation passes. Sampling and saving use the
    # plain module: generation grows the sequence one token at a time, and a
    # compiled module's state_dict keys carry an "_orig_mod." prefix.
    compiled_model = torch.compile(model)

    precision = "bf16 (autocast)" if use_bf16 else "fp32"
    print(
        f"Training on {device} | {precision} "
        f"| {len(train_ds):,} train / {len(val_ds):,} val conversations "
        f"| {examples_per_step} conversations/step "
        f"({config.grad_accum_steps} x {config.batch_size}) "
        f"| {config.max_steps:,} steps"
    )

    print("Samples before training:")
    sample_replies(model, tokenizer, config, step=0)

    best_path = best_checkpoint_path(run_dir)
    best_val_loss = float("inf")
    best_step = None

    model.train()
    batches = cycle(train_dataloader)
    started = time.monotonic()
    window_started = started
    window_loss = 0.0
    window_targets = 0
    window_steps = 0

    for step in range(1, config.max_steps + 1):
        micro_batches = [next(batches) for _ in range(config.grad_accum_steps)]
        # Average over every assistant token in the step rather than per
        # micro-batch, so long replies are not down-weighted against short ones.
        step_targets = max(sum(count_targets(y) for _, y in micro_batches), 1)

        optimizer.zero_grad()
        step_loss = 0.0
        for x, y in micro_batches:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = compiled_model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    ignore_index=IGNORE_TOKEN_ID,
                    reduction="sum",
                ) / step_targets
            loss.backward()
            step_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        lr = get_lr(step - 1, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        window_loss += step_loss * step_targets
        window_targets += step_targets
        window_steps += 1

        last_step = step == config.max_steps
        validate = step % config.val_interval == 0 or last_step

        if step % config.log_interval == 0 or validate:
            # A single step is too noisy to read, so report the mean over
            # every assistant token seen since the previous line.
            now = time.monotonic()
            train_loss = window_loss / window_targets
            message = (
                f"Step {step:,}/{config.max_steps:,} "
                f"| loss {train_loss:.4f} | lr {lr:.2e} | grad norm {grad_norm:.4f} "
                f"| {(now - window_started) / window_steps:.2f}s/step "
                f"| {now - started:.0f}s elapsed"
            )
            metrics = {
                "train/loss": train_loss,
                "train/lr": lr,
                "train/grad_norm": float(grad_norm),
            }

            if validate:
                val_loss = evaluate(compiled_model, val_dataloader, device, use_bf16)
                message += f" | val loss {val_loss:.4f}"
                metrics["val/loss"] = val_loss

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    model.save(str(best_path))
                    message += " (best)"

            print(message)
            if wandb.run is not None:
                wandb.log(metrics, step=step)

            if validate:
                sample_replies(model, tokenizer, config, step)

            # Restart the clock after validation so it is not counted
            # against the next window's step time.
            window_started = time.monotonic()
            window_loss = 0.0
            window_targets = 0
            window_steps = 0

    if best_step is None:
        raise ValueError("validation loss was never finite, so no checkpoint was saved")

    print(
        f"Completed: {config.max_steps:,} steps in {time.monotonic() - started:.0f}s, "
        f"best val loss {best_val_loss:.4f} at step {best_step:,}"
    )

    return {"best_val_loss": best_val_loss, "best_step": best_step}


def main(config, argv=None):
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 to chat on SmolTalk.")
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

    tokenizer = get_tokenizer()
    train_ds = SmolTalkDataset("train", tokenizer, config.max_length)
    # SmolTalk has no validation split, so validate on a fixed prefix of test.
    test_ds = SmolTalkDataset("test", tokenizer, config.max_length)
    val_ds = Subset(test_ds, range(min(config.val_examples, len(test_ds))))
    # Resolve before saving and wandb.init so both record the real step count
    # rather than None. train() resolves again and is a no-op once set.
    config.max_steps = resolve_max_steps(config, len(train_ds))

    run_dir = run_directory(args.checkpoint_dir, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saved config to {save_config(run_dir, config)}")

    wandb.init(
        project=config.wandb_project,
        name=run_dir.name,
        config=asdict(config),
    )

    model = build_model(config.model_size, tokenizer.n_vocab)

    try:
        results = train(model, tokenizer, config, (train_ds, val_ds), run_dir, device)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        wandb.finish(exit_code=1)
        return 1

    wandb.summary.update(results)
    print(f"Best model is at {best_checkpoint_path(run_dir)}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main(TrainingConfig()))
