import torch
import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.rnn(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=device
        )
