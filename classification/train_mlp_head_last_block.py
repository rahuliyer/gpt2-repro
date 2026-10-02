"""Train an MLP head with the last transformer block unfrozen."""

from classification.train import TrainingConfig, main

CONFIG = TrainingConfig(
    checkpoint_name="mlp_head_last_block",
    head="mlp",
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
