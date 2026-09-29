"""Stream a Hugging Face dataset into a GPT-2 token file."""

import argparse
from collections.abc import Mapping
from itertools import islice
from pathlib import Path
import sys

from datasets import load_dataset
import numpy as np
import tiktoken


PROGRESS_INTERVAL = 1_000


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


def preprocess_dataset(
    dataset,
    filename,
    tokenizer,
    num_rows=None,
    *,
    progress_stream=sys.stderr,
    progress_interval=PROGRESS_INTERVAL,
):
    """Tokenize dataset rows and return the numbers of rows and tokens written."""
    rows = islice(dataset, num_rows) if num_rows is not None else iter(dataset)
    row_count = 0
    token_count = 0

    with filename.open("wb") as output:
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping) or "text" not in row:
                raise ValueError(f"row {row_number} does not contain a 'text' field")

            text = row["text"]
            if not isinstance(text, str):
                raise ValueError(f"row {row_number} has a non-string 'text' field")

            tokens = tokenizer.encode(text)
            tokens.append(tokenizer.eot_token)
            np.asarray(tokens, dtype=np.uint16).tofile(output)

            row_count = row_number
            token_count += len(tokens)
            if row_count == 1 or row_count % progress_interval == 0:
                if num_rows is None:
                    print(f"Processed {row_count} rows", file=progress_stream)
                else:
                    print(
                        f"Processed {row_count}/{num_rows} rows",
                        file=progress_stream,
                    )

    print(
        f"Completed: wrote {row_count} rows and {token_count} tokens to {filename}",
        file=progress_stream,
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
        preprocess_dataset(dataset, args.filename, tokenizer, args.num_rows)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: failed while processing dataset: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
