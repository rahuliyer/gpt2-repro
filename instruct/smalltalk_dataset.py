import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .chat import encode_message
from .tokenizer import IGNORE_TOKEN_ID

class SmolTalkDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=1024):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.ds = load_dataset(
            "HuggingFaceTB/smol-smoltalk",
            split=split
        )

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, index):
        messages = self.ds[index]["messages"]

        inputs = []
        labels = []
        for message in messages:
            assert message["role"] in ("user", "assistant", "system")
            encoded_inputs, encoded_labels = encode_message(self.tokenizer, message)
            inputs += encoded_inputs
            labels += encoded_labels

        inputs = inputs[:-1][:self.max_length]
        labels = labels[1:][:self.max_length]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_fn(batch, pad_id):
    """Right-pad a batch to its longest sequence.

    Padded labels are IGNORE_TOKEN_ID so they add nothing to the loss, and
    causal attention means real tokens never attend to the padding after them.
    """
    max_len = max(len(x) for x, _ in batch)
    inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_TOKEN_ID, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        inputs[i, :len(x)] = x
        labels[i, :len(y)] = y

    return inputs, labels
