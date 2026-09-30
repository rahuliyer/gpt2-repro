# GPT-2 Reproduction

A from-scratch implementation of GPT-2 — model, tokenizer pipeline, and
pretraining loop — written directly against PyTorch rather than using
Transformers' training utilities.

- `model/` the GPT-2 architecture (`GPT2`, `TransformerBlock`, configs)
- `pretraining/` dataset preparation and the pretraining loop
- `classification/` classification fine-tuning
- `instruct/` instruction fine-tuning

## Setup

Install the locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Prepare data

`pretraining/preprocess.py` streams the FineWeb-Edu `sample-10BT` config,
tokenizes it with the GPT-2 BPE vocabulary, and writes raw `uint16` token
files — one for training, one for validation:

```bash
uv run python pretraining/preprocess.py \
    datasets/fineweb/train.bin \
    datasets/fineweb/val.bin
```

Validation takes 10% of the tokens, capped at 50M so a full-sample run does
not spend ~1B tokens on validation; everything above the cap stays in train.
Pass `--max-tokens` to process only part of the sample, which is much faster
for a smoke test:

```bash
uv run python pretraining/preprocess.py \
    datasets/fineweb/train.bin \
    datasets/fineweb/val.bin \
    --max-tokens 2000000
```

Tokenizing the full 10BT sample produces roughly 20GB of `.bin` output, so
budget disk accordingly. The `datasets/` directory is gitignored.

## Pretrain

```bash
uv run python pretraining/train.py \
    --train-dataset datasets/fineweb/train.bin \
    --val-dataset datasets/fineweb/val.bin \
    --checkpoint-dir checkpoints
```

The checkpoint directory is created if missing, and the filename is built
from `checkpoint_name` in the config plus a timestamp — for example
`checkpoints/gpt2_20260930_143022.safetensors` — so consecutive runs never
overwrite each other.

Hyperparameters live in the `TrainingConfig` dataclass at the top of
`pretraining/train.py` — edit them there rather than passing flags. The
defaults follow the GPT-2 paper: AdamW at 6e-4 with betas (0.9, 0.95) and
0.1 weight decay applied to matrices but not to biases or LayerNorm scales,
gradient clipping at 1.0, 100 warmup steps, and a cosine decay to 6e-5.
Weights initialize from N(0, 0.02), with the residual-path projections
scaled by `1 / sqrt(2 * n_layers)`.

### Effective batch size

`total_batch_size` is a **token budget per optimizer step** (524,288 = 2^19,
the GPT-2 paper's ~0.5M). The number of gradient accumulation steps is
derived from it:

```
grad_accum_steps = total_batch_size // (batch_size * context_len)
```

So `batch_size` is purely a memory knob — raise it on a larger GPU and the
accumulation count drops automatically while the effective batch, and
therefore the learning-rate schedule, stays valid. `total_batch_size` must be
divisible by `batch_size * context_len` or startup fails with an error.

Each accumulation cycle scales its loss by `1 / grad_accum_steps` before
`backward()`, so the accumulated gradient is the mean over the cycle rather
than the sum. Gradients are clipped once per optimizer step, after the full
cycle has accumulated.

### Precision

Training uses bf16 autocast when the GPU supports it natively, and falls back
to fp32 otherwise. The check deliberately passes `including_emulation=False`:
`torch.cuda.is_bf16_supported()` returns `True` by default on cards that only
emulate bf16 in software (Turing and earlier), which would train through a
slow path with no tensor-core benefit. The startup line reports which mode is
active.

### Logging

Every step logs loss, learning rate, pre-clip gradient norm, throughput, and
timing.

Validation is measured two ways. During training, every `eval_interval` steps
scores `eval_iters` batches — a cheap sample, and since the loader is not
shuffled it is always the same prefix, so the periodic values are comparable
to each other step over step. After the loop, one full pass over the whole
validation file produces the number actually worth reporting, which is what
`train()` returns and what lands in the W&B run summary.

The two are not directly comparable, so they are labelled differently in the
output. Keep `eval_iters` modest: a full pass over a 50M-token validation set
is tens of thousands of batches, which is fine once at the end but would
dominate the run if done at every interval.

Runs are tracked in Weights & Biases under the `wandb_project` name.
`wandb.init()` happens in `main()`, and the training loop only logs when a run
is active, so importing and calling `train()` directly from a script or test
never starts a tracked run. Set `WANDB_MODE=disabled` or `offline` to turn
tracking off without touching code.

## Inspect token files

Decode a raw `uint16` token file back to text:

```bash
uv run python pretraining/decode.py --input datasets/fineweb/val.bin
uv run python pretraining/decode.py --input datasets/fineweb/val.bin --output decoded.txt
```

## Tests

```bash
uv run pytest
```

`tests/test_eqv.py` checks this implementation against Hugging Face's
`GPT2LMHeadModel` and needs to download the pretrained weights.
