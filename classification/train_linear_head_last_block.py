"""Train a single linear layer with the last transformer block unfrozen."""

from classification.train import TrainingConfig, main

CONFIG = TrainingConfig(
    checkpoint_name="linear_head_last_block",
    head="linear",
    unfreeze="last_block",
    batch_size=128,
    max_lr=1e-4,
    min_lr=1e-5,
    warmup_steps=100,
    val_interval=200,
    log_interval=20,
)

if __name__ == "__main__":
    raise SystemExit(main(CONFIG))
