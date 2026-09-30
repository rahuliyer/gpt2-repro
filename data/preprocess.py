"""Stream a Hugging Face dataset into a GPT-2 token file."""

import argparse
from collections.abc import Mapping
from itertools import islice
import os
from pathlib import Path
import sys
import time

from datasets import load_dataset
import numpy as np
import tiktoken


PROGRESS_INTERVAL = 1_000
PROGRESS_UPDATES = 100
PROGRESS_SECONDS = 30


def positive_int(value):
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tokenize a Hugging Face dataset for GPT-2 training."
    )
    parser.add_argument(
        "--filename",
        required=True,
        type=Path,
        help="Output path for the raw uint16 token file.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--dataset-name",
        help="Optional Hugging Face dataset config or subset name.",
    )
    parser.add_argument(
        "--num-rows",
        type=positive_int,
        help="Maximum rows to process; omitted means the entire split.",
    )
    parser.add_argument(
        "--max-tokens",
        "--max_tokens",
        dest="max_tokens",
        type=positive_int,
        help="Maximum tokens to write; the final row may be truncated.",
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=("train", "val", "test"),
        dest="split_type",
        help="Dataset split to process. 'val' maps to 'validation'.",
    )
    return parser.parse_args(argv)


def load_streaming_dataset(
    dataset_id, dataset_name, split_type
):
    """Load the requested dataset split without materializing it in memory."""
    split = "validation" if split_type == "val" else split_type
    options = {"split": split, "streaming": True}
    if dataset_name is not None:
        options["name"] = dataset_name
    return load_dataset(dataset_id, **options)


def resolve_total_rows(dataset, num_rows):
    """Return the expected number of rows, or None when it is unknown."""
    if num_rows is not None:
        return num_rows
    splits = getattr(getattr(dataset, "info", None), "splits", None)
    if not splits:
        return None
    totals = [
        split.num_examples
        for split in splits.values()
        if getattr(split, "num_examples", None)
    ]
    return totals[0] if len(totals) == 1 else None


def report_progress(
    progress_stream, row_count, token_count, total_rows, max_tokens, elapsed
):
    """Print one progress line, with a percentage when a total is known."""
    if max_tokens is not None:
        headline = (
            f"Processed {token_count:,}/{max_tokens:,} tokens "
            f"({100 * token_count / max_tokens:.1f}%)"
        )
        detail = f"{row_count:,} rows"
    elif total_rows is not None:
        headline = (
            f"Processed {row_count:,}/{total_rows:,} rows "
            f"({100 * row_count / total_rows:.1f}%)"
        )
        detail = f"{token_count:,} tokens"
    else:
        headline = f"Processed {row_count:,} rows"
        detail = f"{token_count:,} tokens"

    rate = token_count / elapsed if elapsed > 0 else 0
    print(
        f"{headline}, {detail}, {elapsed:.0f}s elapsed, {rate:,.0f} tok/s",
        file=progress_stream,
        flush=True,
    )


def preprocess_dataset(
    dataset,
    filename,
    tokenizer,
    num_rows=None,
    max_tokens=None,
    *,
    progress_stream=sys.stderr,
    progress_interval=PROGRESS_INTERVAL,
):
    """Tokenize dataset rows and return the numbers of rows and tokens written."""
    total_rows = resolve_total_rows(dataset, num_rows)
    rows = islice(dataset, num_rows) if num_rows is not None else iter(dataset)
    row_count = 0
    token_count = 0
    started = time.monotonic()
    last_report = started
    reported_milestone = 0

    # Aim for roughly PROGRESS_UPDATES lines whenever a total is known, and
    # fall back to a fixed row interval when the split size is unknown.
    if max_tokens is not None:
        token_interval = max(1, max_tokens // PROGRESS_UPDATES)
        row_interval = None
    elif total_rows is not None:
        token_interval = None
        row_interval = max(1, total_rows // PROGRESS_UPDATES)
    else:
        token_interval = None
        row_interval = progress_interval

    with filename.open("wb") as output:
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping) or "text" not in row:
                raise ValueError(f"row {row_number} does not contain a 'text' field")

            text = row["text"]
            if not isinstance(text, str):
                raise ValueError(f"row {row_number} has a non-string 'text' field")

            tokens = tokenizer.encode(text)
            tokens.append(tokenizer.eot_token)
            if max_tokens is not None:
                remaining_tokens = max_tokens - token_count
                tokens = tokens[:remaining_tokens]
            np.asarray(tokens, dtype=np.uint16).tofile(output)

            row_count = row_number
            token_count += len(tokens)
            if token_interval is not None:
                milestone = token_count // token_interval
                due = milestone > reported_milestone
                reported_milestone = milestone
            else:
                due = row_count % row_interval == 0

            now = time.monotonic()
            # Report on a timer too, so a slow stream never looks hung.
            if row_count == 1 or due or now - last_report >= PROGRESS_SECONDS:
                last_report = now
                report_progress(
                    progress_stream,
                    row_count,
                    token_count,
                    total_rows,
                    max_tokens,
                    now - started,
                )

            if max_tokens is not None and token_count >= max_tokens:
                break

    print(
        f"Completed: wrote {row_count} rows and {token_count} tokens to {filename}"
        f" in {time.monotonic() - started:.0f}s",
        file=progress_stream,
        flush=True,
    )
    return row_count, token_count


def main(argv=None):
    args = parse_args(argv)

    try:
        dataset = load_streaming_dataset(
            args.dataset_id, args.dataset_name, args.split_type
        )
    except Exception as exc:
        print(f"error: failed to load dataset: {exc}", file=sys.stderr)
        return 1

    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        preprocess_dataset(
            dataset,
            args.filename,
            tokenizer,
            args.num_rows,
            args.max_tokens,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: failed while processing dataset: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    # Skip interpreter shutdown: abandoning a streaming `datasets` iterator
    # leaves native parquet prefetch threads that never join, so a normal exit
    # hangs indefinitely after all the work is already done.
    returncode = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(returncode)
