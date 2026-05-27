"""
LSTM модель для генерации текста.

Поддерживает:
  - однослойную LSTM (num_layers=1)
  - многослойную LSTM (num_layers>1)

Архитектура:
  Embedding → LSTM → Dropout → Linear
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM language model.

    Args:
        vocab_size:  размер словаря
        embed_dim:   размерность эмбеддингов
        hidden_size: размер скрытого состояния
        num_layers:  количество LSTM слоёв
        dropout:     dropout (применяется между слоями и перед fc)
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

        self.lstm = nn.LSTM(
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
        x:      (batch, seq_len)
        hidden: tuple (h, c) или None

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: tuple (h, c)
        """
        emb = self.dropout(self.embedding(x))       # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(emb, hidden)         # out: (batch, seq_len, hidden)
        out = self.dropout(out)
        logits = self.fc(out)                        # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Нулевые начальные состояния (h_0, c_0)."""
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c
