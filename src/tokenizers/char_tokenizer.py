"""
Посимвольная токенизация (char-level).
Самый простой вариант: каждый символ — отдельный токен.
"""

import json
import os


class CharTokenizer:
    """
    Char-level tokenizer.

    Vocabulary строится из всех уникальных символов текста.
    Порядок символов детерминирован (отсортирован), что важно для воспроизводимости.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build_vocab(self, text: str):
        """Строим словарь из всех символов текста."""
        unique_chars = sorted(set(text))
        self.char2idx = {ch: idx for idx, ch in enumerate(unique_chars)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.vocab_size = len(unique_chars)
        print(f"[CharTokenizer] vocab_size = {self.vocab_size}")
        return self

    def encode(self, text: str) -> list:
        """Строка → список индексов."""
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, indices: list) -> str:
        """Список индексов → строка."""
        return "".join(self.idx2char.get(idx, "?") for idx in indices)

    def save(self, path: str):
        """Сохраняем словарь в JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "char2idx": self.char2idx,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[CharTokenizer] сохранён → {path}")

    def load(self, path: str):
        """Загружаем словарь из JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char2idx = data["char2idx"]
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        self.vocab_size = data["vocab_size"]
        return self

    def get_vocab_info(self) -> dict:
        """Возвращает информацию о словаре для отображения."""
        chars = sorted(self.char2idx.keys())
        printable = [c for c in chars if c.isprintable()]
        return {
            "vocab_size": self.vocab_size,
            "sample_chars": printable[:20],
            "has_digits": any(c.isdigit() for c in chars),
            "has_upper": any(c.isupper() for c in chars),
            "has_lower": any(c.islower() for c in chars),
            "has_punct": any(not c.isalnum() and not c.isspace() for c in chars),
        }
