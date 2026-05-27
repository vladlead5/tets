import os
import re
import requests
import torch
from torch.utils.data import Dataset


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn"
    "/master/data/tinyshakespeare/input.txt"
)
DATA_PATH = "data/shakespeare.txt"


def download_data(save_path=DATA_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"Dataset already exists: {save_path}")
        return save_path
    print("Downloading dataset...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved: {save_path} ({len(response.text):,} chars)")
    return save_path


def load_text(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text):
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def train_val_split(text, val_fraction=0.1):
    split_idx = int(len(text) * (1 - val_fraction))
    return text[:split_idx], text[split_idx:]


class CharTextDataset(Dataset):
    def __init__(self, encoded, seq_len):
        self.data = torch.tensor(encoded, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class TokenTextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y
