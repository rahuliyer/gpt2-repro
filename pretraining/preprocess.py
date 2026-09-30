"""Stream the FineWeb 10BT sample into GPT-2 train and validation token files."""

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import sys
import time

from datasets import load_dataset
import numpy as np
import tiktoken


DATASET_ID = "HuggingFaceFW/fineweb"
DATASET_NAME = "sample-10BT"
DATASET_SPLIT = "train"
TRAIN_FRACTION = 0.9
TOKEN_BYTES = 2  # uint16

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
        description="Tokenize the FineWeb 10BT sample into train and val token files."
    )
    parser.add_argument(
        "train_filename",
        type=Path,
        help="Output path for the training tokens (raw uint16).",
    )
    parser.add_argument(
        "val_filename",
        type=Path,
        help="Output path for the validation tokens (raw uint16).",
    )
    parser.add_argument(
        "--max-tokens",
        "--max_tokens",
        dest="max_tokens",
        type=positive_int,
        help="Total tokens to write across both files; omitted means the whole sample.",
    )
    return parser.parse_args(argv)


def load_streaming_dataset():
    """Load the FineWeb 10BT sample without materializing it in memory."""
    return load_dataset(
        DATASET_ID, name=DATASET_NAME, split=DATASET_SPLIT, streaming=True
    )


def resolve_total_rows(dataset):
    """Return the expected number of rows, or None when it is unknown."""
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
    max_tokens=None,
    *,
    progress_stream=sys.stderr,
    progress_interval=PROGRESS_INTERVAL,
):
    """Tokenize dataset rows and return the numbers of rows and tokens written."""
    total_rows = resolve_total_rows(dataset)
    rows = iter(dataset)
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


def split_train_val(
    train_filename,
    val_filename,
    train_fraction=TRAIN_FRACTION,
    *,
    progress_stream=sys.stderr,
):
    """Carve the tail off a token file into a separate validation file.

    The train file is written first as one contiguous stream; this moves the
    last ``1 - train_fraction`` of it into ``val_filename`` and truncates the
    train file in place, so only the validation tokens are ever copied.
    """
    tokens = np.memmap(train_filename, dtype=np.uint16, mode="r")
    total_tokens = len(tokens)
    if total_tokens == 0:
        del tokens
        raise ValueError(f"{train_filename} is empty, nothing to split")

    split_index = round(total_tokens * train_fraction)
    tokens[split_index:].tofile(val_filename)
    # Release the mapping before resizing the file underneath it.
    del tokens
    os.truncate(train_filename, split_index * TOKEN_BYTES)

    val_tokens = total_tokens - split_index
    print(
        f"Split {total_tokens:,} tokens: {split_index:,} train "
        f"({100 * split_index / total_tokens:.1f}%) -> {train_filename}, "
        f"{val_tokens:,} val -> {val_filename}",
        file=progress_stream,
        flush=True,
    )
    return split_index, val_tokens


def main(argv=None):
    args = parse_args(argv)

    try:
        dataset = load_streaming_dataset()
    except Exception as exc:
        print(f"error: failed to load dataset: {exc}", file=sys.stderr)
        return 1

    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        args.train_filename.parent.mkdir(parents=True, exist_ok=True)
        args.val_filename.parent.mkdir(parents=True, exist_ok=True)
        preprocess_dataset(
            dataset,
            args.train_filename,
            tokenizer,
            args.max_tokens,
        )
        split_train_val(args.train_filename, args.val_filename)
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
