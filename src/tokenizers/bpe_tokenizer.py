"""
BPE (Byte Pair Encoding) токенизация с использованием библиотеки tokenizers (HuggingFace).
Обучаем BPE на нашем датасете с нуля.
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class BPETokenizer:
    """
    BPE tokenizer на основе HuggingFace tokenizers.

    Обучается на произвольном тексте, сохраняет и загружает модель.
    Поддерживает encode/decode как CharTokenizer и WordTokenizer.
    """

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train(self, text: str, save_path: str = None):
        """Обучаем BPE tokenizer на тексте."""
        # создаём временный файл для тренировки
        tmp_file = "/tmp/bpe_train_text.txt"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(text)

        # инициализируем tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"],
            show_progress=True,
        )

        tokenizer.train(files=[tmp_file], trainer=trainer)
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

        print(f"[BPETokenizer] обучен: vocab_size = {self.vocab_size}")

        if save_path:
            self.save(save_path)

        return self

    def encode(self, text: str) -> list:
        """Текст → список токен-ID."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer не обучен. Вызовите train() или load().")
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        """Список токен-ID → текст."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer не обучен. Вызовите train() или load().")
        return self.tokenizer.decode(ids)

    def token_to_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, idx: int) -> str:
        return self.tokenizer.id_to_token(idx)

    def save(self, path: str):
        """Сохраняем tokenizer в файл."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.tokenizer.save(path)
        print(f"[BPETokenizer] сохранён → {path}")

    def load(self, path: str):
        """Загружаем tokenizer из файла."""
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"[BPETokenizer] загружен: vocab_size = {self.vocab_size}")
        return self

    def get_vocab(self) -> dict:
        """Возвращает словарь {токен: id}."""
        return self.tokenizer.get_vocab()

    def show_sample_tokens(self, text: str, n: int = 50):
        """Показываем первые n токенов для примера."""
        encoding = self.tokenizer.encode(text[:500])
        tokens = encoding.tokens[:n]
        ids = encoding.ids[:n]
        return list(zip(tokens, ids))
