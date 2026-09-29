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


class PreprocessTests(unittest.TestCase):
    def test_num_rows_is_optional(self):
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
