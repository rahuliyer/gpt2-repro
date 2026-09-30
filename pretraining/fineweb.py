import numpy as np
import torch
from torch.utils.data import Dataset


class FineWebDataset(Dataset):
    def __init__(self, path, context_len):
        self.context_len = context_len
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return (len(self.tokens) - 1) // self.context_len

    def __getitem__(self, idx):
        start = idx * self.context_len

        buf = np.asarray(
            self.tokens[start:start + self.context_len + 1],
            dtype=np.int64,
        )
        buf = torch.from_numpy(buf)

        return buf[:-1], buf[1:]
