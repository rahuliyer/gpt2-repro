# GPT-2 Reproduction

A from-scratch implementation of the GPT-2 workflow, organized into:

- `pretraining/` for language-model pretraining, plus the dataset acquisition
  and preparation scripts it depends on
- `classification/` for classification fine-tuning
- `instruct/` for instruction fine-tuning

Implement each workflow directly in its corresponding directory.

## Setup

Install the locked dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The environment includes PyTorch and NumPy, along with Hugging Face libraries
for downloading GPT-2 weights, loading safetensors files, and accessing datasets
such as FineWeb. The model and training implementations are left to this
project rather than supplied by Transformers.

## Decode token files

Decode a raw `uint16` token file created by `pretraining/preprocess.py` to stdout:

```bash
python pretraining/decode.py --input tokens.bin
```

Pass an optional output path to write the decoded text to a file instead:

```bash
python pretraining/decode.py --input tokens.bin --output decoded.txt
```
