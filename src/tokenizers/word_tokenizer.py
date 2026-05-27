"""
Токенизация по словам (word-level).
Словарь строится из наиболее частых слов; редкие заменяются <UNK>.
"""

import re
import json
import os
from collections import Counter


class WordTokenizer:
    """
    Word-level tokenizer.

    Специальные токены:
        <PAD> = 0  — дополнение до длины батча
        <UNK> = 1  — неизвестные/редкие слова
        <SOS> = 2  — начало последовательности
        <EOS> = 3  — конец последовательности
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def _tokenize_text(self, text: str) -> list:
        """Разбиваем текст на токены: слова + знаки препинания отдельно."""
        # выделяем слова и знаки препинания как отдельные токены
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
        return tokens

    def build_vocab(self, text: str):
        """Строим словарь из текста."""
        tokens = self._tokenize_text(text)
        counter = Counter(tokens)

        # отбираем слова с достаточной частотой
        frequent = [(word, cnt) for word, cnt in counter.most_common() if cnt >= self.min_freq]

        # ограничиваем размер словаря (учитываем 4 спец. токена)
        max_words = self.max_vocab_size - 4
        frequent = frequent[:max_words]

        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        all_tokens = special + [word for word, _ in frequent]

        self.word2idx = {word: idx for idx, word in enumerate(all_tokens)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(all_tokens)

        unk_ratio = sum(cnt for word, cnt in counter.items() if word not in self.word2idx) / max(1, sum(counter.values()))
        print(f"[WordTokenizer] vocab_size = {self.vocab_size}, UNK rate ≈ {unk_ratio:.2%}")
        return self

    def encode(self, text: str, add_special: bool = False) -> list:
        """Текст → список индексов."""
        tokens = self._tokenize_text(text)
        unk_idx = self.word2idx[self.UNK_TOKEN]
        ids = [self.word2idx.get(t, unk_idx) for t in tokens]

        if add_special:
            ids = [self.word2idx[self.SOS_TOKEN]] + ids + [self.word2idx[self.EOS_TOKEN]]
        return ids

    def decode(self, indices: list, skip_special: bool = True) -> str:
        """Список индексов → текст."""
        special_ids = {
            self.word2idx.get(self.PAD_TOKEN, -1),
            self.word2idx.get(self.SOS_TOKEN, -1),
            self.word2idx.get(self.EOS_TOKEN, -1),
        }
        words = []
        for idx in indices:
            if skip_special and idx in special_ids:
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))

        # простая сборка: пробел перед знаком препинания не добавляем
        result = ""
        for i, word in enumerate(words):
            if i == 0:
                result += word
            elif re.match(r"[^\w]", word):
                result += word
            else:
                result += " " + word
        return result

    def save(self, path: str):
        """Сохраняем словарь."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "word2idx": self.word2idx,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[WordTokenizer] сохранён → {path}")

    def load(self, path: str):
        """Загружаем словарь."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.max_vocab_size = data.get("max_vocab_size", self.vocab_size)
        self.min_freq = data.get("min_freq", 2)
        return self

    @property
    def pad_id(self):
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_id(self):
        return self.word2idx[self.UNK_TOKEN]
