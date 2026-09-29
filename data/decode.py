"""Decode a raw GPT-2 uint16 token file into text."""

import argparse
from pathlib import Path
import sys

import numpy as np
import tiktoken


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Decode a raw uint16 GPT-2 token file."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the raw uint16 token file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path; omitted writes the decoded text to stdout.",
    )
    return parser.parse_args(argv)


def decode_file(filename, tokenizer):
    """Read and decode all uint16 tokens from filename."""
    tokens = np.fromfile(filename, dtype=np.uint16)
    return tokenizer.decode(tokens.tolist())


def write_decoded_text(
    text,
    target_filename,
    *,
    stdout=None,
):
    """Write decoded text to a file or stdout without modifying its contents."""
    if target_filename is None:
        if stdout is None:
            stdout = sys.stdout
        stdout.write(text)
    else:
        target_filename.write_text(text, encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)

    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        text = decode_file(args.input, tokenizer)
        write_decoded_text(text, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: failed to decode token file: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
