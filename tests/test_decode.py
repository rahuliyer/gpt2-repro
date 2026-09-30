from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from pretraining import decode


class FakeTokenizer:
    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class DecodeTests(unittest.TestCase):
    def test_output_filename_is_optional(self):
        args = decode.parse_args(["--input", "tokens.bin"])

        self.assertEqual(args.input, Path("tokens.bin"))
        self.assertIsNone(args.output)

    def test_parses_output_filename(self):
        args = decode.parse_args(
            ["--input", "tokens.bin", "--output", "decoded.txt"]
        )

        self.assertEqual(args.output, Path("decoded.txt"))

    def test_decodes_uint16_token_file(self):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            np.asarray([97, 98, 99], dtype=np.uint16).tofile(filename)

            text = decode.decode_file(filename, FakeTokenizer())

        self.assertEqual(text, "abc")

    @patch("pretraining.decode.tiktoken.get_encoding", return_value=FakeTokenizer())
    def test_main_writes_to_stdout_by_default(self, _get_encoding):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            np.asarray([97, 98, 99], dtype=np.uint16).tofile(filename)
            output = StringIO()

            with redirect_stdout(output):
                result = decode.main(["--input", str(filename)])

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "abc")

    @patch("pretraining.decode.tiktoken.get_encoding", return_value=FakeTokenizer())
    def test_main_writes_to_target_file(self, _get_encoding):
        with TemporaryDirectory() as directory:
            filename = Path(directory) / "tokens.bin"
            target_filename = Path(directory) / "decoded.txt"
            np.asarray([104, 233], dtype=np.uint16).tofile(filename)

            result = decode.main(
                ["--input", str(filename), "--output", str(target_filename)]
            )

            self.assertEqual(result, 0)
            self.assertEqual(target_filename.read_text(encoding="utf-8"), "hé")

    @patch("pretraining.decode.tiktoken.get_encoding", return_value=FakeTokenizer())
    def test_main_reports_read_errors(self, _get_encoding):
        error = StringIO()

        with redirect_stderr(error):
            result = decode.main(["--input", "missing.bin"])

        self.assertEqual(result, 1)
        self.assertIn("error:", error.getvalue())


if __name__ == "__main__":
    unittest.main()
