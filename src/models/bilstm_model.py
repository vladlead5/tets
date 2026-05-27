"""
Bidirectional LSTM модель для генерации текста.

Архитектурные особенности и ограничения:
──────────────────────────────────────────
ОГРАНИЧЕНИЕ MULTI-LAYER BiLSTM:
  В двуслойном BiLSTM backward-представление первого слоя попадает
  во входы forward-LSTM второго слоя, что создаёт утечку будущего контекста
  даже в «forward» половине выхода. Для честного causal LM используем
  num_layers=1 по умолчанию.

ОДНОСЛОЙНЫЙ BiLSTM:
  - Forward-направление: видит только x[0..t] при предсказании t+1 ✓
  - Backward-направление: видит x[t..T] — не участвует в LM-голове
  - LM-голова использует только forward-часть выхода → нет leakage

  Это означает, что backward LSTM не вносит прямого вклада в loss.
  Однако backward-обучение идёт через shared embedding и помогает
  на раннем обучении как дополнительный регуляризирующий сигнал.

ГЕНЕРАЦИЯ:
  При генерации токен за токеном backward-направление за каждый шаг
  получает на вход только текущий токен (без будущих), что делает его
  бессмысленным. Фактически на инференсе BiLSTM = однонаправленный LSTM.
  Это подтверждает, что BiLSTM подходит для задач классификации/NER,
  а не для авторегрессивной генерации.
"""

import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM language model.

    LM-голова использует только forward-половину вывода BiLSTM,
    исключая утечку будущего контекста при вычислении loss.

    Args:
        vocab_size:  размер словаря
        embed_dim:   размерность эмбеддингов
        hidden_size: размер скрытого состояния (на каждое направление)
        num_layers:  количество слоёв (рекомендуется 1 для causal LM)
        dropout:     dropout
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

        # LM-голова использует только forward-половину (hidden_size),
        # а не hidden_size*2, чтобы избежать утечки будущего контекста.
        # В однослойном BiLSTM forward-выход[:,t,:hidden_size] зависит
        # только от x[0..t] — строго каузальный.
        self.fc = nn.Linear(hidden_size, vocab_size)

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

        # Берём только forward-половину: первые hidden_size каналов
        # (PyTorch кладёт forward-направление первым в конкатенации)
        forward_out = out[:, :, :self.hidden_size]    # (batch, seq_len, hidden_size)

        logits = self.fc(forward_out)                 # (batch, seq_len, vocab_size)
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

        Чётные индексы (0, 2, ...) — forward, нечётные (1, 3, ...) — backward.
        """
        h, c = hidden
        h_forward = h[0::2]
        c_forward = c[0::2]
        return h_forward, c_forward
