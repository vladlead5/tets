"""
Генерация текста для всех типов моделей.

Стратегии:
  - greedy:       всегда выбираем самый вероятный токен
  - temperature:  смягчаем/обостряем распределение температурой
  - top-k:        сэмплируем только из top-k самых вероятных токенов
"""

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────
# RNN / LSTM / BiLSTM генерация
# ──────────────────────────────────────────────────────

def generate_rnn(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    strategy: str = "temperature",
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device = None,
) -> str:
    """
    Генерация текста авторегрессивно для RNN/LSTM/BiLSTM.

    prompt   — начало текста (затравка)
    strategy — "greedy", "temperature", "top_k"
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # кодируем prompt
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]

    # прогоняем prompt через модель чтобы получить начальное hidden state
    with torch.no_grad():
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits, hidden = model(x)
        # берём logits последнего токена
        last_logits = logits[0, -1, :]  # (vocab_size,)

    generated_ids = list(input_ids)

    for _ in range(max_new_tokens):
        next_id = sample_next_token(last_logits, strategy, temperature, top_k)
        generated_ids.append(next_id)

        # следующий шаг
        with torch.no_grad():
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            last_logits = logits[0, -1, :]

    return tokenizer.decode(generated_ids)


# ──────────────────────────────────────────────────────
# Transformer (GPT-like) генерация
# ──────────────────────────────────────────────────────

def generate_transformer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    strategy: str = "temperature",
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device = None,
) -> str:
    """
    Авторегрессивная генерация для GPT-like Transformer.

    Transformer работает с фиксированным окном контекста (max_seq_len).
    На каждом шаге передаём весь накопленный контекст.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]

    generated_ids = list(input_ids)
    max_ctx = model.max_seq_len

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # обрезаем контекст до max_seq_len
            ctx = generated_ids[-max_ctx:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)

            logits = model(x)               # (1, T, vocab_size)
            last_logits = logits[0, -1, :]  # (vocab_size,)

            next_id = sample_next_token(last_logits, strategy, temperature, top_k)
            generated_ids.append(next_id)

    return tokenizer.decode(generated_ids)


# ──────────────────────────────────────────────────────
# Вспомогательная функция сэмплирования
# ──────────────────────────────────────────────────────

def sample_next_token(
    logits: torch.Tensor,
    strategy: str = "temperature",
    temperature: float = 0.8,
    top_k: int = 40,
) -> int:
    """
    Выбор следующего токена по логитам.

    strategy:
      "greedy"      — argmax
      "temperature" — softmax(logits / T) и сэмплируем
      "top_k"       — обнуляем все кроме top-k, потом temperature
    """
    if strategy == "greedy":
        return int(logits.argmax(dim=-1).item())

    # применяем температуру
    if temperature <= 0:
        temperature = 1e-8
    logits = logits / temperature

    if strategy == "top_k":
        # обнуляем всё кроме top-k
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = logits.masked_fill(logits < min_val, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return int(next_id)


# ──────────────────────────────────────────────────────
# Удобный интерфейс: генерируем сразу несколько промптов
# ──────────────────────────────────────────────────────

def generate_samples(
    model,
    tokenizer,
    prompts: list,
    model_type: str = "rnn",  # "rnn" или "transformer"
    max_new_tokens: int = 150,
    strategies: list = None,
    device: torch.device = None,
) -> dict:
    """
    Генерируем текст для нескольких промптов и стратегий.

    Returns:
        dict: {prompt: {strategy: generated_text}}
    """
    if strategies is None:
        strategies = ["greedy", "temperature", "top_k"]

    generate_fn = generate_rnn if model_type == "rnn" else generate_transformer

    results = {}
    for prompt in prompts:
        results[prompt] = {}
        for strategy in strategies:
            try:
                text = generate_fn(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                    strategy=strategy,
                    device=device,
                )
                results[prompt][strategy] = text
            except Exception as e:
                results[prompt][strategy] = f"[Ошибка генерации: {e}]"

    return results
