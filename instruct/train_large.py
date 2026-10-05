"""Fine-tune GPT-2 large to chat on SmolTalk, sized for an 80 GB H100."""

from instruct.train import TrainingConfig, main

# 32 conversations per step, as in the small defaults, so the step count and
# learning rate schedule line up with small runs.
CONFIG = TrainingConfig(
    checkpoint_name="instruct_large",
    model_size="large",
    batch_size=8,
    grad_accum_steps=4,
    eval_batch_size=16,
    max_lr=2e-5,
    min_lr=2e-6,
    val_interval=500,
)

if __name__ == "__main__":
    raise SystemExit(main(CONFIG))
