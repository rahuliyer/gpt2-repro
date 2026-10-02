import torch
from torch.utils.data import Dataset

from datasets import load_dataset

class AGNewsDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=128):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = load_dataset("sh0416/ag_news", split=split)

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["title"] + " " + item["description"]
        label = item["label"] - 1  # Adjust label to be 0-indexed

        input_ids = self.tokenizer.encode(text)
        input_ids = input_ids[:self.max_length]

        input_len = len(input_ids)
        if input_len < self.max_length:
            input_ids += [self.tokenizer.eot_token] * (self.max_length - len(input_ids))

        return torch.tensor(input_ids), input_len, torch.tensor(label)

def get_datasets(tokenizer, val_split=0.1, max_length=128):
    end_idx = int((1 - val_split) * 120000)

    train_set = AGNewsDataset(f"train[:{end_idx}]", tokenizer, max_length=max_length)
    val_set = AGNewsDataset(f"train[{end_idx}:]", tokenizer, max_length=max_length)
    test_set = AGNewsDataset("test", tokenizer, max_length=max_length)

    return train_set, val_set, test_set

