import json
import os


class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        chars = sorted(set(text))
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)
        print(f"CharTokenizer: vocab_size = {self.vocab_size}")
        return self

    def encode(self, text):
        return [self.char2idx[ch] for ch in text if ch in self.char2idx]

    def decode(self, indices):
        return "".join(self.idx2char.get(i, "?") for i in indices)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"char2idx": self.char2idx, "vocab_size": self.vocab_size}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.char2idx = data["char2idx"]
        self.idx2char = {int(v): k for k, v in self.char2idx.items()}
        self.vocab_size = data["vocab_size"]
        return self
