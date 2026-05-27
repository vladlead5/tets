import torch
import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        forward_out = out[:, :, :self.hidden_size]
        logits = self.fc(forward_out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size,
                        device=device)
        c = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size,
                        device=device)
        return h, c

    def get_forward_hidden(self, hidden):
        h, c = hidden
        return h[0::2], c[0::2]
