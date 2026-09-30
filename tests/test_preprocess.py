from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from pretraining import preprocess


class FakeTokenizer:
    eot_token = 99

    def encode(self, text, disallowed_special=None):
        # Mirrors tiktoken's signature so the call in preprocess_dataset is
        # exercised as written.
        return [ord(character) for character in text]


class FakeSplit:
    def __init__(self, num_examples):
        self.num_examples = num_examples


class FakeInfo:
    def __init__(self, num_examples):
        self.splits = {"train": FakeSplit(num_examples)}


class FakeInfoDataset:
    """A streaming-like dataset that advertises its split size."""

    def __init__(self, num_examples, rows=()):
        self.info = FakeInfo(num_examples)
        self._rows = list(rows)

    def __iter__(self):
        return iter(self._rows)


class PreprocessTests(unittest.TestCase):
    def test_parses_output_paths_and_optional_token_limit(self):
        args = preprocess.parse_args(["train.bin", "val.bin"])

        self.assertEqual(args.train_filename, Path("train.bin"))
        self.assertEqual(args.val_filename, Path("val.bin"))
        self.assertIsNone(args.max_tokens)

    def test_max_tokens_must_be_positive(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            preprocess.parse_args(["train.bin", "val.bin", "--max_tokens", "0"])

    def test_both_output_paths_are_required(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            preprocess.parse_args(["train.bin"])

    @patch("pretraining.preprocess.load_dataset")
    def test_loads_the_configured_fineweb_sample(self, load_dataset):
        preprocess.load_streaming_dataset()
        load_dataset.assert_called_once_with(
            preprocess.DATASET_ID,
            name=preprocess.DATASET_NAME,
            split=preprocess.DATASET_SPLIT,
            streaming=True,
        )
        self.assertEqual(preprocess.DATASET_NAME, "sample-10BT")

    def test_processes_entire_dataset_and_reports_progress(self):
        rows = [{"text": "a"}, {"text": "bc"}, {"text": "d"}]
        progress = StringIO()

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            result = preprocess.preprocess_dataset(
                rows,
                filename,
                FakeTokenizer(),
                progress_stream=progress,
                progress_interval=2,
            )
            tokens = np.fromfile(filename, dtype=np.uint16).tolist()

        self.assertEqual(result, (3, 7))
        self.assertEqual(tokens, [97, 99, 98, 99, 99, 100, 99])
        self.assertIn("Processed 1 rows", progress.getvalue())
        self.assertIn("Processed 2 rows", progress.getvalue())
        self.assertIn("Completed: wrote 3 rows and 7 tokens", progress.getvalue())

    def test_limits_tokens_by_truncating_the_final_row(self):
        progress = StringIO()

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            result = preprocess.preprocess_dataset(
                [{"text": "ab"}, {"text": "de"}, {"text": "f"}],
                filename,
                FakeTokenizer(),
                max_tokens=5,
                progress_stream=progress,
            )
            tokens = np.fromfile(filename, dtype=np.uint16).tolist()

        self.assertEqual(result, (2, 5))
        self.assertEqual(tokens, [97, 98, 99, 100, 101])
        self.assertIn("Completed: wrote 2 rows and 5 tokens", progress.getvalue())

    def test_writes_all_available_tokens_when_limit_is_larger(self):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            result = preprocess.preprocess_dataset(
                [{"text": "a"}, {"text": "b"}],
                filename,
                FakeTokenizer(),
                max_tokens=10,
                progress_stream=StringIO(),
            )
            tokens = np.fromfile(filename, dtype=np.uint16).tolist()

        self.assertEqual(result, (2, 4))
        self.assertEqual(tokens, [97, 99, 98, 99])

    def test_reports_token_progress_against_the_token_limit(self):
        progress = StringIO()

        with TemporaryDirectory() as directory:
            preprocess.preprocess_dataset(
                [{"text": "ab"}] * 50,
                Path(directory) / "tokens.bin",
                FakeTokenizer(),
                max_tokens=100,
                progress_stream=progress,
            )

        output = progress.getvalue()
        # A token limit switches the headline from rows to tokens.
        self.assertIn("Processed 3/100 tokens (3.0%)", output)
        self.assertIn("/100 tokens (100.0%)", output)
        self.assertNotIn("Processed 1 rows,", output)

    def test_progress_falls_back_to_rows_without_a_token_limit(self):
        progress = StringIO()

        with TemporaryDirectory() as directory:
            preprocess.preprocess_dataset(
                [{"text": "a"}, {"text": "b"}],
                Path(directory) / "tokens.bin",
                FakeTokenizer(),
                progress_stream=progress,
                progress_interval=1,
            )

        output = progress.getvalue()
        self.assertIn("Processed 1 rows, 2 tokens", output)
        self.assertIn("Processed 2 rows, 4 tokens", output)

    def test_resolve_total_rows_reads_the_split_size(self):
        self.assertEqual(preprocess.resolve_total_rows(FakeInfoDataset(500)), 500)

    def test_resolve_total_rows_is_none_when_unknown(self):
        self.assertIsNone(preprocess.resolve_total_rows([{"text": "a"}]))

    def test_reports_row_percentage_when_the_split_size_is_known(self):
        progress = StringIO()

        with TemporaryDirectory() as directory:
            preprocess.preprocess_dataset(
                FakeInfoDataset(200, rows=[{"text": "a"}] * 4),
                Path(directory) / "tokens.bin",
                FakeTokenizer(),
                progress_stream=progress,
            )

        self.assertIn("Processed 1/200 rows (0.5%)", progress.getvalue())

    def test_literal_endoftext_in_a_document_is_ordinary_text(self):
        # FineWeb documents sometimes contain the literal "<|endoftext|>".
        # tiktoken raises on it unless disallowed_special is cleared, and
        # allowing it would inject a spurious document boundary.
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            rows, tokens_written = preprocess.preprocess_dataset(
                [{"text": "before <|endoftext|> after"}],
                filename,
                tokenizer,
                progress_stream=StringIO(),
            )
            tokens = np.fromfile(filename, dtype=np.uint16).tolist()

        self.assertEqual(rows, 1)
        # Exactly one EOT, the one appended as the document terminator.
        self.assertEqual(tokens.count(tokenizer.eot_token), 1)
        self.assertEqual(tokens[-1], tokenizer.eot_token)
        # The literal text round-trips instead of becoming a boundary.
        self.assertEqual(
            tokenizer.decode(tokens[:-1]), "before <|endoftext|> after"
        )

    def test_split_train_val_splits_ninety_ten(self):
        with TemporaryDirectory() as directory:
            train = Path(directory) / "train.bin"
            val = Path(directory) / "val.bin"
            original = list(range(100))
            np.asarray(original, dtype=np.uint16).tofile(train)

            result = preprocess.split_train_val(train, val, progress_stream=StringIO())

            train_tokens = np.fromfile(train, dtype=np.uint16).tolist()
            val_tokens = np.fromfile(val, dtype=np.uint16).tolist()

        self.assertEqual(result, (90, 10))
        # The train file is truncated in place and val holds the tail.
        self.assertEqual(train_tokens, original[:90])
        self.assertEqual(val_tokens, original[90:])
        self.assertEqual(train_tokens + val_tokens, original)

    def test_split_train_val_rounds_uneven_totals(self):
        with TemporaryDirectory() as directory:
            train = Path(directory) / "train.bin"
            val = Path(directory) / "val.bin"
            np.asarray(range(7), dtype=np.uint16).tofile(train)

            # round(7 * 0.9) == 6
            result = preprocess.split_train_val(train, val, progress_stream=StringIO())

            train_tokens = np.fromfile(train, dtype=np.uint16).tolist()
            val_tokens = np.fromfile(val, dtype=np.uint16).tolist()

        self.assertEqual(result, (6, 1))
        self.assertEqual(train_tokens, [0, 1, 2, 3, 4, 5])
        self.assertEqual(val_tokens, [6])

    def test_split_train_val_caps_the_validation_size(self):
        with TemporaryDirectory() as directory:
            train = Path(directory) / "train.bin"
            val = Path(directory) / "val.bin"
            np.asarray(range(100), dtype=np.uint16).tofile(train)

            # 10% would be 10 tokens; the cap pulls it down to 3.
            result = preprocess.split_train_val(
                train, val, val_token_cap=3, progress_stream=StringIO()
            )

            train_tokens = np.fromfile(train, dtype=np.uint16).tolist()
            val_tokens = np.fromfile(val, dtype=np.uint16).tolist()

        self.assertEqual(result, (97, 3))
        self.assertEqual(val_tokens, [97, 98, 99])
        # Tokens above the cap stay in train rather than being dropped.
        self.assertEqual(len(train_tokens) + len(val_tokens), 100)

    def test_split_train_val_ignores_a_cap_above_the_fraction(self):
        with TemporaryDirectory() as directory:
            train = Path(directory) / "train.bin"
            val = Path(directory) / "val.bin"
            np.asarray(range(100), dtype=np.uint16).tofile(train)

            # 10% is 10 tokens, well under the cap, so the fraction wins.
            result = preprocess.split_train_val(
                train, val, val_token_cap=1_000, progress_stream=StringIO()
            )

        self.assertEqual(result, (90, 10))

    def test_split_train_val_honours_a_custom_fraction(self):
        with TemporaryDirectory() as directory:
            train = Path(directory) / "train.bin"
            val = Path(directory) / "val.bin"
            np.asarray(range(10), dtype=np.uint16).tofile(train)

            result = preprocess.split_train_val(
                train, val, train_fraction=0.5, progress_stream=StringIO()
            )

            self.assertEqual(np.fromfile(val, dtype=np.uint16).tolist(), [5, 6, 7, 8, 9])

        self.assertEqual(result, (5, 5))

    def test_split_train_val_rejects_a_too_small_token_file(self):
        with TemporaryDirectory() as directory:
            train = Path(directory) / "train.bin"
            val = Path(directory) / "val.bin"
            train.write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "too few"):
                preprocess.split_train_val(train, val, progress_stream=StringIO())

            self.assertFalse(val.exists())

    def test_rejects_missing_or_non_string_text(self):
        for row in ({"other": "value"}, {"text": None}):
            with self.subTest(row=row), TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "row 1"):
                    preprocess.preprocess_dataset(
                        [row],
                        Path(directory) / "tokens.bin",
                        FakeTokenizer(),
                        progress_stream=StringIO(),
                    )


if __name__ == "__main__":
    unittest.main()
