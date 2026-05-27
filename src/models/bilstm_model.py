"""
Bidirectional LSTM модель для генерации текста.

Важная особенность: двунаправленный LSTM видит будущий контекст,
что не соответствует стандартному языковому моделированию.
В этой реализации BiLSTM используется следующим образом:

  - На обучении — BiLSTM обрабатывает последовательность,
    выходные состояния конкатенируются и проецируются в словарь.
  - На генерации — используем только forward-направление (unidirectional inference),
    что является стандартным компромиссом для BiLSTM в LM.

Это позволяет сравнить качество представлений BiLSTM vs LSTM
при том же числе параметров.
"""

import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM language model.

    Args:
        vocab_size:  размер словаря
        embed_dim:   размерность эмбеддингов
        hidden_size: размер скрытого состояния (на каждое направление)
        num_layers:  количество слоёв
        dropout:     dropout
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2  # bidirectional

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # выход LSTM: hidden_size * 2 (прямое + обратное направления)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        """
        x:      (batch, seq_len)
        hidden: tuple (h, c) с формой (num_layers*2, batch, hidden) или None

        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: tuple (h, c)
        """
        emb = self.dropout(self.embedding(x))        # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(emb, hidden)          # out: (batch, seq_len, hidden*2)
        out = self.dropout(out)
        logits = self.fc(out)                         # (batch, seq_len, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        """Нулевые начальные состояния для двунаправленного LSTM."""
        h = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=device)
        return h, c

    def get_forward_hidden(self, hidden):
        """
        Для генерации берём только forward-направление из скрытого состояния.
        h shape: (num_layers*2, batch, hidden) → (num_layers, batch, hidden)
        """
        h, c = hidden
        # чётные индексы — forward, нечётные — backward
        h_forward = h[0::2]
        c_forward = c[0::2]
        return h_forward, c_forward
