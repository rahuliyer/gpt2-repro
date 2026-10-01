from contextlib import redirect_stderr
from io import StringIO
from itertools import count
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


class FakeStatefulDataset:
    """A streaming-like dataset that can save and restore its position."""

    def __init__(self, rows):
        self._rows = list(rows)
        self._position = 0
        self.rows_read = 0

    def __iter__(self):
        while self._position < len(self._rows):
            self._position += 1
            self.rows_read += 1
            yield self._rows[self._position - 1]

    def state_dict(self):
        return {"position": self._position}

    def load_state_dict(self, state):
        self._position = state["position"]


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

    def test_resume_is_off_by_default(self):
        self.assertFalse(preprocess.parse_args(["train.bin", "val.bin"]).resume)
        self.assertTrue(
            preprocess.parse_args(["train.bin", "val.bin", "--resume"]).resume
        )

    def test_checkpoint_sits_beside_the_train_file(self):
        self.assertEqual(
            preprocess.checkpoint_path(Path("data/train.bin")),
            Path("data/train.bin.ckpt.json"),
        )

    def test_writes_a_checkpoint_while_processing(self):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            checkpoint_filename = preprocess.checkpoint_path(filename)
            # The bad third row stops the run with two rows on disk.
            with self.assertRaises(ValueError):
                preprocess.preprocess_dataset(
                    [{"text": "a"}, {"text": "bc"}, {"text": None}],
                    filename,
                    FakeTokenizer(),
                    checkpoint_filename=checkpoint_filename,
                    checkpoint_seconds=0,
                    progress_stream=StringIO(),
                )
            checkpoint = preprocess.load_checkpoint(checkpoint_filename)
            leftovers = [path.name for path in Path(directory).glob("*.tmp")]

        self.assertEqual(checkpoint["rows"], 2)
        self.assertEqual(checkpoint["tokens"], 5)
        self.assertFalse(checkpoint["complete"])
        self.assertEqual(checkpoint["dataset"], preprocess.DATASET_SOURCE)
        self.assertEqual(leftovers, [])

    def test_marks_the_checkpoint_complete_when_the_dataset_ends(self):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            checkpoint_filename = preprocess.checkpoint_path(filename)
            preprocess.preprocess_dataset(
                [{"text": "a"}, {"text": "bc"}],
                filename,
                FakeTokenizer(),
                checkpoint_filename=checkpoint_filename,
                progress_stream=StringIO(),
            )
            checkpoint = preprocess.load_checkpoint(checkpoint_filename)

        self.assertTrue(checkpoint["complete"])
        self.assertEqual((checkpoint["rows"], checkpoint["tokens"]), (2, 5))

    def test_resume_matches_an_uninterrupted_run(self):
        rows = [{"text": "a"}, {"text": "bc"}, {"text": "d"}, {"text": "ef"}]
        progress = StringIO()

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            checkpoint_filename = preprocess.checkpoint_path(filename)
            with self.assertRaises(ValueError):
                preprocess.preprocess_dataset(
                    rows[:2] + [{"text": None}],
                    filename,
                    FakeTokenizer(),
                    checkpoint_filename=checkpoint_filename,
                    checkpoint_seconds=0,
                    progress_stream=StringIO(),
                )
            # Tokens that reached the file after the last checkpoint.
            with filename.open("ab") as output:
                output.write(b"\x01\x00\x02\x00")

            result = preprocess.preprocess_dataset(
                rows,
                filename,
                FakeTokenizer(),
                checkpoint_filename=checkpoint_filename,
                resume_from=preprocess.load_checkpoint(checkpoint_filename),
                progress_stream=progress,
            )
            tokens = np.fromfile(filename, dtype=np.uint16).tolist()

        self.assertEqual(result, (4, 10))
        self.assertEqual(tokens, [97, 99, 98, 99, 99, 100, 99, 101, 102, 99])
        self.assertIn("Resuming after 2 rows and 5 tokens", progress.getvalue())
        self.assertIn("Processed 3 rows", progress.getvalue())

    def test_resume_restores_the_dataset_position_when_it_can(self):
        rows = [{"text": "a"}, {"text": "bc"}, {"text": "d"}]

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            checkpoint_filename = preprocess.checkpoint_path(filename)
            with self.assertRaises(ValueError):
                preprocess.preprocess_dataset(
                    FakeStatefulDataset(rows[:2] + [{"text": None}]),
                    filename,
                    FakeTokenizer(),
                    checkpoint_filename=checkpoint_filename,
                    checkpoint_seconds=0,
                    progress_stream=StringIO(),
                )
            checkpoint = preprocess.load_checkpoint(checkpoint_filename)

            dataset = FakeStatefulDataset(rows)
            result = preprocess.preprocess_dataset(
                dataset,
                filename,
                FakeTokenizer(),
                checkpoint_filename=checkpoint_filename,
                resume_from=checkpoint,
                progress_stream=StringIO(),
            )
            tokens = np.fromfile(filename, dtype=np.uint16).tolist()

        self.assertEqual(checkpoint["dataset_state"], {"position": 2})
        self.assertEqual(result, (3, 7))
        self.assertEqual(tokens, [97, 99, 98, 99, 99, 100, 99])
        # Only the unfinished row is read; the first two are never revisited.
        self.assertEqual(dataset.rows_read, 1)

    def test_resume_honours_the_token_limit(self):
        rows = [{"text": "ab"}, {"text": "de"}, {"text": "f"}]

        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            checkpoint_filename = preprocess.checkpoint_path(filename)
            with self.assertRaises(ValueError):
                preprocess.preprocess_dataset(
                    rows[:1] + [{"text": None}],
                    filename,
                    FakeTokenizer(),
                    max_tokens=5,
                    checkpoint_filename=checkpoint_filename,
                    checkpoint_seconds=0,
                    progress_stream=StringIO(),
                )

            result = preprocess.preprocess_dataset(
                rows,
                filename,
                FakeTokenizer(),
                max_tokens=5,
                checkpoint_filename=checkpoint_filename,
                resume_from=preprocess.load_checkpoint(checkpoint_filename),
                progress_stream=StringIO(),
            )
            tokens = np.fromfile(filename, dtype=np.uint16).tolist()

        self.assertEqual(result, (2, 5))
        self.assertEqual(tokens, [97, 98, 99, 100, 101])

    def test_resume_rejects_a_token_file_shorter_than_the_checkpoint(self):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            np.asarray([1, 2], dtype=np.uint16).tofile(filename)

            with self.assertRaisesRegex(ValueError, "cannot resume"):
                preprocess.preprocess_dataset(
                    [{"text": "a"}],
                    filename,
                    FakeTokenizer(),
                    resume_from={"rows": 3, "tokens": 9, "dataset_state": None},
                    progress_stream=StringIO(),
                )

            # The file is left alone for the user to inspect.
            self.assertEqual(np.fromfile(filename, dtype=np.uint16).tolist(), [1, 2])

    def test_resume_rejects_a_checkpoint_from_different_settings(self):
        checkpoint = {"dataset": preprocess.DATASET_SOURCE, "max_tokens": 100}

        preprocess.check_resume_checkpoint(checkpoint, 100)
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            preprocess.check_resume_checkpoint(checkpoint, None)
        with self.assertRaisesRegex(ValueError, "dataset"):
            preprocess.check_resume_checkpoint(
                {"dataset": "other/data/train", "max_tokens": 100}, 100
            )


class MainTests(unittest.TestCase):
    ROWS = [{"text": "abcd"}] * 4  # 20 tokens: 18 train, 2 val

    def run_main(self, directory, *flags, rows=ROWS):
        train = Path(directory) / "train.bin"
        val = Path(directory) / "val.bin"
        stderr = StringIO()
        # Each clock reading is far past the last, so every row checkpoints.
        clock = count(step=preprocess.CHECKPOINT_SECONDS)
        with (
            patch("pretraining.preprocess.load_streaming_dataset") as load,
            patch("pretraining.preprocess.tiktoken.get_encoding") as get_encoding,
            patch("pretraining.preprocess.time.monotonic", side_effect=clock.__next__),
            redirect_stderr(stderr),
        ):
            load.return_value = rows
            get_encoding.return_value = FakeTokenizer()
            returncode = preprocess.main([str(train), str(val), *flags])
        return returncode, train, val, stderr.getvalue(), load

    def test_a_finished_run_leaves_no_checkpoint(self):
        with TemporaryDirectory() as directory:
            returncode, train, val, _, _ = self.run_main(directory)

            self.assertEqual(returncode, 0)
            self.assertEqual(train.stat().st_size, 18 * preprocess.TOKEN_BYTES)
            self.assertEqual(val.stat().st_size, 2 * preprocess.TOKEN_BYTES)
            self.assertFalse(preprocess.checkpoint_path(train).exists())

    def test_resume_without_a_checkpoint_fails(self):
        with TemporaryDirectory() as directory:
            returncode, train, _, stderr, load = self.run_main(directory, "--resume")

            self.assertEqual(returncode, 1)
            self.assertIn("--resume found no checkpoint", stderr)
            load.assert_not_called()
            self.assertFalse(train.exists())

    def test_resume_finishes_an_interrupted_run(self):
        with TemporaryDirectory() as directory:
            interrupted = self.ROWS[:2] + [{"text": None}]
            returncode, train, val, _, _ = self.run_main(directory, rows=interrupted)
            checkpoint = preprocess.load_checkpoint(preprocess.checkpoint_path(train))
            self.assertEqual(returncode, 1)
            self.assertEqual((checkpoint["rows"], checkpoint["tokens"]), (2, 10))

            returncode, train, val, _, _ = self.run_main(directory, "--resume")
            tokens = (
                np.fromfile(train, dtype=np.uint16).tolist()
                + np.fromfile(val, dtype=np.uint16).tolist()
            )

            self.assertEqual(returncode, 0)
            self.assertEqual(tokens, [97, 98, 99, 100, 99] * 4)
            self.assertFalse(preprocess.checkpoint_path(train).exists())

    def test_resume_rejects_a_changed_token_limit(self):
        with TemporaryDirectory() as directory:
            self.run_main(directory, rows=self.ROWS[:2] + [{"text": None}])

            returncode, _, _, stderr, load = self.run_main(
                directory, "--resume", "--max-tokens", "8"
            )

            self.assertEqual(returncode, 1)
            self.assertIn("cannot resume", stderr)
            load.assert_not_called()

    def test_resume_splits_without_retokenizing_when_only_the_split_is_left(self):
        with TemporaryDirectory() as directory:
            with patch(
                "pretraining.preprocess.split_train_val", side_effect=OSError("disk")
            ):
                returncode, train, val, _, _ = self.run_main(directory)
            self.assertEqual(returncode, 1)

            returncode, train, val, _, load = self.run_main(directory, "--resume")

            self.assertEqual(returncode, 0)
            load.assert_not_called()
            self.assertEqual(train.stat().st_size, 18 * preprocess.TOKEN_BYTES)
            self.assertEqual(val.stat().st_size, 2 * preprocess.TOKEN_BYTES)
            self.assertFalse(preprocess.checkpoint_path(train).exists())

    def test_resume_does_not_split_twice(self):
        with TemporaryDirectory() as directory:
            # Interrupted after the split but before the checkpoint was removed.
            with patch.object(Path, "unlink", side_effect=OSError("interrupted")):
                returncode, train, val, _, _ = self.run_main(directory)
            self.assertEqual(returncode, 1)
            self.assertEqual(train.stat().st_size, 18 * preprocess.TOKEN_BYTES)

            returncode, train, val, _, _ = self.run_main(directory, "--resume")

            self.assertEqual(returncode, 0)
            self.assertEqual(train.stat().st_size, 18 * preprocess.TOKEN_BYTES)
            self.assertEqual(val.stat().st_size, 2 * preprocess.TOKEN_BYTES)
            self.assertFalse(preprocess.checkpoint_path(train).exists())

    def test_starting_over_warns_about_an_existing_checkpoint(self):
        with TemporaryDirectory() as directory:
            self.run_main(directory, rows=self.ROWS[:2] + [{"text": None}])

            returncode, train, val, stderr, _ = self.run_main(directory)

            self.assertEqual(returncode, 0)
            self.assertIn("pass --resume to continue it", stderr)
            self.assertEqual(train.stat().st_size, 18 * preprocess.TOKEN_BYTES)


if __name__ == "__main__":
    unittest.main()
