"""Stream the FineWeb 10BT sample into GPT-2 train and validation token files."""

import argparse
from collections.abc import Mapping
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time

from datasets import load_dataset
import numpy as np
import tiktoken


DATASET_ID = "HuggingFaceFW/fineweb-edu"
DATASET_NAME = "sample-10BT"
DATASET_SPLIT = "train"
DATASET_SOURCE = f"{DATASET_ID}/{DATASET_NAME}/{DATASET_SPLIT}"
TRAIN_FRACTION = 0.9
VAL_TOKEN_CAP = 50_000_000  # 10% of the full sample would be ~1B, which is wasteful
TOKEN_BYTES = 2  # uint16

PROGRESS_INTERVAL = 1_000
PROGRESS_UPDATES = 100
PROGRESS_SECONDS = 30

CHECKPOINT_SUFFIX = ".ckpt.json"
CHECKPOINT_SECONDS = 60


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from the checkpoint beside train_filename.",
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


def checkpoint_path(train_filename):
    """Return where the checkpoint for a given training token file lives."""
    return train_filename.with_name(train_filename.name + CHECKPOINT_SUFFIX)


def save_checkpoint(path, checkpoint):
    """Write a checkpoint atomically, so a crash never leaves a torn one."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as output:
        json.dump(checkpoint, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    """Read a checkpoint written by `save_checkpoint`."""
    with path.open() as source:
        return json.load(source)


def check_resume_checkpoint(checkpoint, max_tokens):
    """Raise when a checkpoint was written by a run with different settings."""
    current = {"dataset": DATASET_SOURCE, "max_tokens": max_tokens}
    conflicts = [
        f"{field}: checkpoint has {checkpoint[field]!r}, this run has {value!r}"
        for field, value in current.items()
        if checkpoint[field] != value
    ]
    if conflicts:
        raise ValueError(
            "cannot resume, these settings must match the checkpoint:\n  "
            + "\n  ".join(conflicts)
        )


def resume_rows(dataset, checkpoint):
    """Return an iterator over the rows after the last checkpointed one."""
    state = checkpoint["dataset_state"]
    if state is not None and hasattr(dataset, "load_state_dict"):
        # Jumps straight to the right shard instead of re-reading the stream.
        dataset.load_state_dict(state)
        return iter(dataset)
    return islice(dataset, checkpoint["rows"], None)


def open_output(filename, token_count):
    """Open the token file positioned after its first `token_count` tokens."""
    if token_count == 0:
        return filename.open("wb")

    expected_bytes = token_count * TOKEN_BYTES
    actual_bytes = filename.stat().st_size
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"cannot resume, {filename} holds {actual_bytes // TOKEN_BYTES:,} tokens"
            f" but the checkpoint expects {token_count:,}"
        )
    output = filename.open("r+b")
    # Anything past the checkpoint was written after it and will be redone.
    output.truncate(expected_bytes)
    output.seek(expected_bytes)
    return output


def report_progress(
    progress_stream,
    row_count,
    token_count,
    total_rows,
    max_tokens,
    elapsed,
    resumed_tokens=0,
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

    rate = (token_count - resumed_tokens) / elapsed if elapsed > 0 else 0
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
    checkpoint_filename=None,
    resume_from=None,
    checkpoint_seconds=CHECKPOINT_SECONDS,
    progress_stream=sys.stderr,
    progress_interval=PROGRESS_INTERVAL,
):
    """Tokenize dataset rows and return the numbers of rows and tokens written.

    With ``checkpoint_filename`` set, progress is saved there every
    ``checkpoint_seconds``. Passing a loaded checkpoint as ``resume_from``
    continues from it; the returned counts then cover the whole file, not just
    this call.
    """
    total_rows = resolve_total_rows(dataset)
    if resume_from is None:
        rows = iter(dataset)
        row_count = 0
        token_count = 0
    else:
        rows = resume_rows(dataset, resume_from)
        row_count = resume_from["rows"]
        token_count = resume_from["tokens"]
        print(
            f"Resuming after {row_count:,} rows and {token_count:,} tokens",
            file=progress_stream,
            flush=True,
        )
    resumed_rows = row_count
    resumed_tokens = token_count
    started = time.monotonic()
    last_report = started
    last_checkpoint = started
    reported_milestone = 0

    def write_checkpoint(output, complete=False):
        if checkpoint_filename is None:
            return
        # The tokens have to be durable before a checkpoint may count them.
        output.flush()
        os.fsync(output.fileno())
        state_dict = getattr(dataset, "state_dict", None)
        save_checkpoint(
            checkpoint_filename,
            {
                "dataset": DATASET_SOURCE,
                "max_tokens": max_tokens,
                "rows": row_count,
                "tokens": token_count,
                "complete": complete,
                "dataset_state": state_dict() if state_dict is not None else None,
            },
        )

    # Aim for roughly PROGRESS_UPDATES lines whenever a total is known, and
    # fall back to a fixed row interval when the split size is unknown.
    if max_tokens is not None:
        token_interval = max(1, max_tokens // PROGRESS_UPDATES)
        row_interval = None
        reported_milestone = token_count // token_interval
    elif total_rows is not None:
        token_interval = None
        row_interval = max(1, total_rows // PROGRESS_UPDATES)
    else:
        token_interval = None
        row_interval = progress_interval

    with open_output(filename, token_count) as output:
        if resume_from is None:
            # A checkpoint left by an earlier run no longer describes this file.
            write_checkpoint(output)
        for row_number, row in enumerate(rows, start=row_count + 1):
            if not isinstance(row, Mapping) or "text" not in row:
                raise ValueError(f"row {row_number} does not contain a 'text' field")

            text = row["text"]
            if not isinstance(text, str):
                raise ValueError(f"row {row_number} has a non-string 'text' field")

            # Some FineWeb documents contain the literal string "<|endoftext|>".
            # tiktoken refuses to encode special tokens by default; allowing it
            # would turn dataset text into a real document boundary, so encode
            # it as ordinary text and keep the appended EOT as the only one.
            tokens = tokenizer.encode(text, disallowed_special=())
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
            first_row = row_count == resumed_rows + 1
            if first_row or due or now - last_report >= PROGRESS_SECONDS:
                last_report = now
                report_progress(
                    progress_stream,
                    row_count,
                    token_count,
                    total_rows,
                    max_tokens,
                    now - started,
                    resumed_tokens,
                )

            if now - last_checkpoint >= checkpoint_seconds:
                last_checkpoint = now
                write_checkpoint(output)

            if max_tokens is not None and token_count >= max_tokens:
                break

        write_checkpoint(output, complete=True)

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
    val_token_cap=VAL_TOKEN_CAP,
    *,
    progress_stream=sys.stderr,
):
    """Carve the tail off a token file into a separate validation file.

    Validation takes ``1 - train_fraction`` of the tokens, capped at
    ``val_token_cap`` so a full-sample run does not spend ~1B tokens on
    validation. Everything above the cap stays in train.

    The train file is written first as one contiguous stream; this moves the
    tail into ``val_filename`` and truncates the train file in place, so only
    the validation tokens are ever copied.
    """
    # Check the size up front: np.memmap raises on an empty file, so the
    # guard has to run before the mapping is created.
    total_tokens = os.path.getsize(train_filename) // TOKEN_BYTES
    if total_tokens < 2:
        raise ValueError(
            f"{train_filename} holds {total_tokens} tokens, too few to split"
        )

    tokens = np.memmap(train_filename, dtype=np.uint16, mode="r")
    split_index = round(total_tokens * train_fraction)
    val_tokens = total_tokens - split_index
    capped = val_tokens > val_token_cap
    if capped:
        val_tokens = val_token_cap
        split_index = total_tokens - val_tokens
    val_tokens = max(1, val_tokens)
    split_index = total_tokens - val_tokens

    tokens[split_index:].tofile(val_filename)
    # Release the mapping before resizing the file underneath it.
    del tokens
    os.truncate(train_filename, split_index * TOKEN_BYTES)

    reason = f" (capped at {val_token_cap:,})" if capped else ""
    print(
        f"Split {total_tokens:,} tokens: {split_index:,} train "
        f"({100 * split_index / total_tokens:.1f}%) -> {train_filename}, "
        f"{val_tokens:,} val{reason} -> {val_filename}",
        file=progress_stream,
        flush=True,
    )
    return split_index, val_tokens


def main(argv=None):
    args = parse_args(argv)
    checkpoint_filename = checkpoint_path(args.train_filename)

    checkpoint = None
    try:
        if args.resume:
            if not checkpoint_filename.exists():
                print(
                    f"error: --resume found no checkpoint at {checkpoint_filename}",
                    file=sys.stderr,
                )
                return 1
            checkpoint = load_checkpoint(checkpoint_filename)
            check_resume_checkpoint(checkpoint, args.max_tokens)
        elif checkpoint_filename.exists():
            print(
                f"warning: starting over, ignoring the checkpoint at"
                f" {checkpoint_filename} (pass --resume to continue it)",
                file=sys.stderr,
            )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tokenized = checkpoint is not None and checkpoint["complete"]
    if not tokenized:
        try:
            dataset = load_streaming_dataset()
        except Exception as exc:
            print(f"error: failed to load dataset: {exc}", file=sys.stderr)
            return 1

    try:
        args.train_filename.parent.mkdir(parents=True, exist_ok=True)
        args.val_filename.parent.mkdir(parents=True, exist_ok=True)
        if not tokenized:
            tokenizer = tiktoken.get_encoding("gpt2")
            _, token_count = preprocess_dataset(
                dataset,
                args.train_filename,
                tokenizer,
                args.max_tokens,
                checkpoint_filename=checkpoint_filename,
                resume_from=checkpoint,
            )
        else:
            token_count = checkpoint["tokens"]

        # The split shrinks the train file as its last step, so a train file
        # already below the tokenized size means an earlier run finished it.
        train_bytes = args.train_filename.stat().st_size
        if train_bytes >= token_count * TOKEN_BYTES:
            split_train_val(args.train_filename, args.val_filename)
        checkpoint_filename.unlink()
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
