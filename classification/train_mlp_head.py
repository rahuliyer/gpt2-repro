"""Train an MLP head on a frozen GPT-2."""

from classification.train import TrainingConfig, main

CONFIG = TrainingConfig(
    checkpoint_name="mlp_head",
    head="mlp",
    unfreeze="none",
    batch_size=512,
    max_lr=3e-4,
    min_lr=3e-5,
    warmup_steps=20,
    val_interval=100,
    log_interval=10,
)

if __name__ == "__main__":
    raise SystemExit(main(CONFIG))
