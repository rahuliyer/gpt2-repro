"""Training helpers shared by the pretraining and classification trainers."""

from datetime import datetime
import math
from pathlib import Path

import torch


def run_directory(checkpoint_dir, config, now=None):
    """Build a timestamped directory so a fresh run never reuses another's."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return Path(checkpoint_dir) / f"{config.checkpoint_name}_{stamp}"


def get_lr(step, config):
    """Cosine learning rate schedule with linear warmup."""
    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


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
