"""Train a single linear layer with the whole model unfrozen."""

from classification.train import TrainingConfig, main

CONFIG = TrainingConfig(
    checkpoint_name="linear_head_full",
    head="linear",
    unfreeze="all",
    batch_size=32,
    max_lr=3e-5,
    min_lr=3e-6,
    warmup_steps=300,
    val_interval=500,
    log_interval=50,
)

if __name__ == "__main__":
    raise SystemExit(main(CONFIG))
