import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def train(self, text, save_path=None):
        tmp = "/tmp/bpe_train.txt"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)

        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"],
            show_progress=True,
        )
        tokenizer.train(files=[tmp], trainer=trainer)

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        print(f"BPETokenizer trained: vocab_size = {self.vocab_size}")

        if save_path:
            self.save(save_path)
        return self

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.tokenizer.save(path)

    def load(self, path):
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"BPETokenizer loaded: vocab_size = {self.vocab_size}")
        return self

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def show_sample_tokens(self, text, n=50):
        encoding = self.tokenizer.encode(text[:500])
        return list(zip(encoding.tokens[:n], encoding.ids[:n]))
