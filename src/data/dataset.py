"""
Загрузка и предобработка датасета Shakespeare (tiny).
Источник: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import os
import re
import requests
import torch
from torch.utils.data import Dataset


DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/shakespeare.txt"


def download_data(save_path: str = DATA_PATH) -> str:
    """Скачиваем текст Shakespeare, если ещё не скачан."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"Датасет уже скачан: {save_path}")
        return save_path

    print(f"Скачиваем датасет из {DATA_URL} ...")
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"Сохранён: {save_path} ({len(response.text):,} символов)")
    return save_path


def load_text(path: str = DATA_PATH) -> str:
    """Читаем текст из файла."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """
    Базовая очистка текста:
    - нормализация пробелов и переносов строк,
    - удаление непечатаемых символов,
    - схлопывание множественных пустых строк.
    """
    # убираем непечатаемые символы кроме \n и пробела
    text = re.sub(r"[^\x20-\x7E\n]", "", text)

    # нормализуем пробелы внутри строк (но не переносы)
    text = re.sub(r"[ \t]+", " ", text)

    # убираем пробелы в начале/конце строк
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # схлопываем 3+ пустых строки в 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def train_val_split(text: str, val_fraction: float = 0.1):
    """
    Делим текст на train и val.
    Разбиение идёт по символам (для char-level).
    """
    split_idx = int(len(text) * (1 - val_fraction))
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    return train_text, val_text


# ──────────────────────────────────────────────────────
# Датасеты для PyTorch DataLoader
# ──────────────────────────────────────────────────────

class CharTextDataset(Dataset):
    """
    Датасет для char-level моделей.
    Каждый элемент — пара (вход, цель) длиной seq_len.
    """

    def __init__(self, encoded: list, seq_len: int):
        self.data = torch.tensor(encoded, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class TokenTextDataset(Dataset):
    """
    Датасет для word/BPE токенизации.
    Принцип тот же: скользящее окно по последовательности токенов.
    """

    def __init__(self, token_ids: list, seq_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def get_data_stats(text: str, tokenizer=None) -> dict:
    """Возвращает базовую статистику по датасету."""
    stats = {
        "num_chars": len(text),
        "num_lines": text.count("\n"),
        "num_words": len(text.split()),
        "unique_chars": len(set(text)),
    }
    if tokenizer is not None:
        tokens = tokenizer.encode(text)
        stats["num_tokens"] = len(tokens)
        stats["vocab_size"] = tokenizer.vocab_size
    return stats
