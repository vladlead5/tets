import re
import json
import os
from collections import Counter


class WordTokenizer:
    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self, max_vocab_size=10000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

    def build_vocab(self, text):
        tokens = self._tokenize(text)
        counter = Counter(tokens)

        frequent = [
            w for w, cnt in counter.most_common()
            if cnt >= self.min_freq
        ][:self.max_vocab_size - 4]

        vocab = [self.PAD, self.UNK, self.SOS, self.EOS] + frequent
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(vocab)

        total = sum(counter.values())
        in_vocab = sum(
            cnt for w, cnt in counter.items() if w in self.word2idx
        )
        unk_rate = 1 - in_vocab / max(1, total)
        print(f"WordTokenizer: vocab_size={self.vocab_size}, "
              f"UNK rate={unk_rate:.2%}")
        return self

    def encode(self, text, add_special=False):
        unk = self.word2idx[self.UNK]
        ids = [self.word2idx.get(t, unk) for t in self._tokenize(text)]
        if add_special:
            ids = (
                [self.word2idx[self.SOS]] + ids + [self.word2idx[self.EOS]]
            )
        return ids

    def decode(self, indices, skip_special=True):
        special = {
            self.word2idx.get(self.PAD, -1),
            self.word2idx.get(self.SOS, -1),
            self.word2idx.get(self.EOS, -1),
        }
        words = []
        for idx in indices:
            if skip_special and idx in special:
                continue
            words.append(self.idx2word.get(idx, self.UNK))

        result = ""
        for i, word in enumerate(words):
            if i == 0 or re.match(r"[^\w]", word):
                result += word
            else:
                result += " " + word
        return result

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "word2idx": self.word2idx,
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
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
        return self.word2idx[self.PAD]

    @property
    def unk_id(self):
        return self.word2idx[self.UNK]
