"""
GPT-подобный Transformer, реализованный с нуля на PyTorch.

Архитектура (decoder-only, как GPT):
  Token Embedding + Positional Embedding
  → N × TransformerBlock (masked self-attention + FFN)
  → LayerNorm
  → Linear (lm head)

Causal mask гарантирует, что позиция i видит только j <= i.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Многоголовое self-attention с causal mask.

    Реализация "руками": разбиваем embed_dim на num_heads голов,
    считаем scaled dot-product attention для каждой, конкатенируем.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim должен делиться на num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # проекции Q, K, V объединены в одну матрицу для эффективности
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch, seq_len, embed_dim)
        Returns: (batch, seq_len, embed_dim)
        """
        B, T, C = x.shape

        # считаем Q, K, V
        qkv = self.qkv_proj(x)                        # (B, T, 3*C)
        q, k, v = qkv.split(self.embed_dim, dim=-1)   # по (B, T, C)

        # разбиваем на головы: (B, num_heads, T, head_dim)
        def reshape(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale      # (B, num_heads, T, T)

        # causal mask: позиция i не видит j > i
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # взвешенная сумма
        out = attn @ v                                 # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.out_proj(out)
        return out


class FeedForward(nn.Module):
    """
    FFN: Linear → GELU → Linear.
    Промежуточная размерность = 4 * embed_dim (как в оригинальном GPT).
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Один блок Transformer (pre-norm вариант, как в GPT-2).

    Порядок: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # attention с residual
        x = x + self.resid_dropout(self.attn(self.ln1(x)))
        # FFN с residual
        x = x + self.resid_dropout(self.ffn(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    """
    GPT-like decoder-only Transformer, реализованный с нуля.

    Args:
        vocab_size:  размер словаря
        embed_dim:   размерность эмбеддингов
        num_heads:   количество голов в attention
        num_layers:  количество transformer блоков
        max_seq_len: максимальная длина последовательности
        dropout:     dropout
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # токен-эмбеддинги
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        # позиционные эмбеддинги (обучаемые, как в GPT)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        self.emb_dropout = nn.Dropout(dropout)

        # стек трансформер блоков
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # финальная нормализация
        self.ln_final = nn.LayerNorm(embed_dim)

        # языковая модельная голова
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # tied weights: эмбеддинги и голова разделяют веса
        # это стандартная техника, уменьшает число параметров
        self.lm_head.weight = self.token_emb.weight

        # инициализация весов
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов как в GPT-2 paper."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        x: (batch, seq_len) — индексы токенов
        Returns: logits (batch, seq_len, vocab_size)
        """
        B, T = x.shape
        assert T <= self.max_seq_len, f"Последовательность длиной {T} > max_seq_len={self.max_seq_len}"

        # позиционные индексы
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)

        # суммируем токен- и позиционные эмбеддинги
        x = self.emb_dropout(self.token_emb(x) + self.pos_emb(positions))

        # проходим через блоки
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
