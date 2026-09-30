from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from data import preprocess


class FakeTokenizer:
    eot_token = 99

    def encode(self, text):
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
    def test_limits_are_optional(self):
        args = preprocess.parse_args(
            [
                "--filename",
                "tokens.bin",
                "--dataset-id",
                "owner/dataset",
                "--type",
                "train",
            ]
        )

        self.assertIsNone(args.num_rows)
        self.assertIsNone(args.max_tokens)

    def test_num_rows_must_be_positive(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            preprocess.parse_args(
                [
                    "--filename",
                    "tokens.bin",
                    "--dataset-id",
                    "owner/dataset",
                    "--num-rows",
                    "0",
                    "--type",
                    "train",
                ]
            )

    def test_max_tokens_must_be_positive(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            preprocess.parse_args(
                [
                    "--filename",
                    "tokens.bin",
                    "--dataset-id",
                    "owner/dataset",
                    "--max_tokens",
                    "0",
                    "--type",
                    "train",
                ]
            )

    def test_type_must_be_supported(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            preprocess.parse_args(
                [
                    "--filename",
                    "tokens.bin",
                    "--dataset-id",
                    "owner/dataset",
                    "--type",
                    "validation",
                ]
            )

    @patch("data.preprocess.load_dataset")
    def test_validation_split_and_optional_dataset_name(self, load_dataset):
        preprocess.load_streaming_dataset("owner/dataset", "subset", "val")
        load_dataset.assert_called_once_with(
            "owner/dataset", name="subset", split="validation", streaming=True
        )

        load_dataset.reset_mock()
        preprocess.load_streaming_dataset("owner/dataset", None, "test")
        load_dataset.assert_called_once_with(
            "owner/dataset", split="test", streaming=True
        )

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

    def test_limits_rows_and_reports_requested_total(self):
        rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        progress = StringIO()

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            result = preprocess.preprocess_dataset(
                rows,
                filename,
                FakeTokenizer(),
                num_rows=2,
                progress_stream=progress,
            )
            tokens = np.fromfile(filename, dtype=np.uint16).tolist()

        self.assertEqual(result, (2, 4))
        self.assertEqual(tokens, [97, 99, 98, 99])
        self.assertIn("Processed 1/2 rows", progress.getvalue())

    def test_writes_all_available_rows_when_limit_is_larger(self):
        progress = StringIO()

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            result = preprocess.preprocess_dataset(
                [{"text": "a"}, {"text": "b"}],
                filename,
                FakeTokenizer(),
                num_rows=10,
                progress_stream=progress,
            )

        self.assertEqual(result, (2, 4))
        self.assertIn("Completed: wrote 2 rows and 4 tokens", progress.getvalue())

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

    def test_resolve_total_rows_prefers_the_requested_row_limit(self):
        self.assertEqual(preprocess.resolve_total_rows(FakeInfoDataset(500), 10), 10)

    def test_resolve_total_rows_reads_the_split_size(self):
        self.assertEqual(preprocess.resolve_total_rows(FakeInfoDataset(500), None), 500)

    def test_resolve_total_rows_is_none_when_unknown(self):
        self.assertIsNone(preprocess.resolve_total_rows([{"text": "a"}], None))

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
