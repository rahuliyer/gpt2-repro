from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from pretraining import FineWebDataset


class FineWebDatasetTests(unittest.TestCase):
    def test_can_be_imported_and_read_tokens(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.asarray([10, 11, 12, 13, 14], dtype=np.uint16).tofile(path)

            dataset = FineWebDataset(path, context_len=2)

            self.assertEqual(len(dataset), 2)
            inputs, targets = dataset[1]
            self.assertEqual(inputs.tolist(), [12, 13])
            self.assertEqual(targets.tolist(), [13, 14])


if __name__ == "__main__":
    unittest.main()
