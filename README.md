# GPT-2 Reproduction

A from-scratch implementation of GPT-2 — model, tokenizer pipeline, and
pretraining loop — written directly against PyTorch rather than using
Transformers' training utilities.

- `model/` the GPT-2 architecture (`GPT2`, `TransformerBlock`, configs)
- `pretraining/` dataset preparation and the pretraining loop
- `classification/` classification fine-tuning
- `instruct/` instruction fine-tuning
- `inference/` text completion
- `utils/` helpers shared by the trainers (LR schedule, optimizer, run directories)

## Weights

The trained weights are on Hugging Face. Each model card has loading code.

| Model | Hugging Face | Write-up |
|---|---|---|
| GPT-2 small pretrained from scratch on FineWeb-Edu | [rahuliyer/gpt2-small-fineweb-edu](https://huggingface.co/rahuliyer/gpt2-small-fineweb-edu) | [pretraining](pretraining/README.md) |
| GPT-2 small fine-tuned on AG News (four setups) | [rahuliyer/gpt2-small-ag-news](https://huggingface.co/rahuliyer/gpt2-small-ag-news) | [classification](classification/README.md) |
| GPT-2 small instruction-tuned on SmolTalk | [rahuliyer/gpt2-small-smoltalk](https://huggingface.co/rahuliyer/gpt2-small-smoltalk) | [instruct](instruct/README.md) |
| GPT-2 medium instruction-tuned on SmolTalk | [rahuliyer/gpt2-medium-smoltalk](https://huggingface.co/rahuliyer/gpt2-medium-smoltalk) | [instruct](instruct/README.md) |
| GPT-2 large instruction-tuned on SmolTalk | [rahuliyer/gpt2-large-smoltalk](https://huggingface.co/rahuliyer/gpt2-large-smoltalk) | [instruct](instruct/README.md) |

The weights load into this repository's `GPT2` class, not Hugging Face
Transformers' `GPT2LMHeadModel`.

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

A full run takes a while, so progress is checkpointed once a minute to
`<train file>.ckpt.json` (here `datasets/fineweb/train.bin.ckpt.json`). If the
run is interrupted, repeat the same command with `--resume` to continue from
the last checkpoint instead of starting over:

```bash
uv run python pretraining/preprocess.py \
    datasets/fineweb/train.bin \
    datasets/fineweb/val.bin \
    --resume
```

`--max-tokens` must match the interrupted run. The checkpoint is deleted once
both files are finished; without `--resume` an existing checkpoint is ignored
and the run starts from the beginning.

Tokenizing the full 10BT sample produces roughly 20GB of `.bin` output, so
budget disk accordingly. The `datasets/` directory is gitignored.

## Pretrain

```bash
uv run python pretraining/train.py \
    --train-dataset datasets/fineweb/train.bin \
    --val-dataset datasets/fineweb/val.bin \
    --checkpoint-dir checkpoints
```

Hyperparameters live in the `TrainingConfig` dataclass at the top of
`pretraining/train.py` — edit them there rather than passing flags. The
defaults follow the GPT-2 paper: AdamW at 6e-4 with betas (0.9, 0.95) and
0.1 weight decay applied to matrices but not to biases or LayerNorm scales,
gradient clipping at 1.0, 700 warmup steps, and a cosine decay to 6e-5.
Weights initialize from N(0, 0.02), with the residual-path projections
scaled by `1 / sqrt(2 * n_layers)`.

### Run length

`max_steps` defaults to `None`, meaning the step count is derived from the
training file so it cannot go stale when the dataset or token budget changes:

```
max_steps = epochs * usable_tokens // total_batch_size
```

`epochs` defaults to `1.0` and accepts fractions, so half a pass is
`epochs=0.5` rather than a recomputed step count. Setting `max_steps`
explicitly overrides the derivation, which is the easy way to run a short
smoke test. The derived value is printed at startup and is what drives the
cosine schedule, not just the stopping condition.

For the full FineWeb-Edu `sample-10BT` sample that works out to ~18,890
steps over ~9.90B tokens.

### Checkpoints

Each fresh run creates its own timestamped directory under
`--checkpoint-dir`; `--resume` continues inside the one it finds:

```
checkpoints/
  gpt2_20260930_143022/
    gpt2_step_018890.safetensors     final weights, tagged with the step
    checkpoints/
      step_017000.pt                 full training state
      step_018000.pt
```

A state checkpoint every `checkpoint_interval` steps holds the model,
optimizer moments, step counter, RNG state, data position and W&B run id —
everything needed to continue, which the weights alone cannot do. Only the
newest `keep_last_n` are kept (3 by default, ~1.5GB each).

```bash
uv run python pretraining/train.py \
    --train-dataset datasets/fineweb/train.bin \
    --val-dataset datasets/fineweb/val.bin \
    --checkpoint-dir checkpoints \
    --resume
```

`--resume` picks the newest checkpoint under `--checkpoint-dir`, restores the
data stream to where it stopped, and reuses the W&B run so the loss curve
stays continuous. Settings that would invalidate the restored step counter
(`total_batch_size`, `batch_size`, `context_len`) are refused; everything else
warns.

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

## Classification

Fine-tunes the released GPT-2 small weights to classify
[AG News](https://huggingface.co/datasets/sh0416/ag_news) into its four
topics. The last 10% of the training split is held out for validation. Each
example is classified from the logits at its last real token, not at the
padded end of the sequence.

There is one script per setup, each a config passed to the shared loop in
`classification/train.py`:

| Script | Head | Backbone | Batch | Peak LR |
|---|---|---|---|---|
| `train_mlp_head.py` | MLP | frozen | 512 | 3e-4 |
| `train_mlp_head_last_block.py` | MLP | last block trains | 128 | 1e-4 |
| `train_linear_head_last_block.py` | single linear layer | last block trains | 128 | 1e-4 |
| `train_linear_head_full.py` | single linear layer | everything trains | 32 | 3e-5 |

"Last block" is the final transformer block together with the layer norm that
follows it.

```bash
uv run python classification/train_mlp_head.py \
    --checkpoint-dir checkpoints/classification \
    --device-id 1
```

`--device-id` picks the GPU (0 or 1) and defaults to 0.

Each run trains for `n_epochs` on a cosine learning rate schedule with linear
warmup, and gets its own `<checkpoint_name>_<YYYYmmdd_HHMMSS>` directory under
`--checkpoint-dir`:

- `config.json` the training config, including the head and what was unfrozen
- `best.safetensors` the full model at its lowest validation loss

Validation runs over the whole held-out set every `val_interval` steps and
once more on the final step. The test split is scored once at the end, with
`best.safetensors` loaded back in rather than the last step's weights.

Printed and logged training loss and accuracy are means over the examples seen
since the previous log line, since a single batch is too noisy to read. Runs
are tracked in the `gpt2-classification-sft` W&B project, one run per
invocation, named after its run directory.

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
