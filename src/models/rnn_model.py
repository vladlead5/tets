"""
Simple RNN модель для генерации текста.

Архитектура:
  Embedding → RNN → Linear (language model head)

Используется ванильный torch.nn.RNN без LSTM ячеек.
"""

import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    """
    Простая RNN для языкового моделирования.

    Args:
        vocab_size: размер словаря
        embed_dim:  размерность эмбеддингов
        hidden_size: размер скрытого состояния RNN
        num_layers:  количество слоёв
        dropout:     dropout между слоями (если num_layers > 1)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN — базовая ванильная ячейка
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x:      (batch, seq_len) — индексы токенов
        hidden: начальное скрытое состояние (None = нули)

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: (num_layers, batch, hidden_size)
        """
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        emb = self.dropout(emb)

        out, hidden = self.rnn(emb, hidden)   # out: (batch, seq_len, hidden)
        out = self.dropout(out)

        logits = self.fc(out)            # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Инициализация нулевого скрытого состояния."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
