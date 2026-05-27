"""
Метрики оценки языковых моделей.

Основная метрика — perplexity (PPL):
  PPL = exp(cross_entropy_loss)

Чем меньше перплексия, тем лучше модель предсказывает текст.
"""

import math
import torch
import torch.nn.functional as F


def compute_perplexity(loss: float) -> float:
    """
    Перплексия из значения loss (cross-entropy).
    Обрезаем снизу чтобы избежать бесконечности при очень плохих моделях.
    """
    return min(math.exp(loss), 9999.0)


def compute_loss_on_loader(model, data_loader, device, model_type: str = "rnn"):
    """
    Вычисляем средний loss на всём data_loader.

    model_type: "rnn"/"lstm"/"bilstm" или "transformer"
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            if model_type == "transformer":
                logits = model(x)
            else:
                logits, _ = model(x)

            # logits: (batch, seq_len, vocab_size)
            # y:      (batch, seq_len)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))
            total_loss += loss.item() * B * T
            total_tokens += B * T

    avg_loss = total_loss / total_tokens
    return avg_loss


def evaluate_model(model, data_loader, device, model_type: str = "rnn"):
    """
    Возвращает loss и perplexity для датасета.
    """
    loss = compute_loss_on_loader(model, data_loader, device, model_type)
    ppl = compute_perplexity(loss)
    return {"loss": loss, "perplexity": ppl}


def bits_per_char(loss: float) -> float:
    """
    Bits per character (BPC) — популярная метрика для char-level моделей.
    BPC = loss / ln(2)
    """
    return loss / math.log(2)
