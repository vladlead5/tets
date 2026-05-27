import torch
import torch.nn.functional as F


def generate_rnn(model, tokenizer, prompt, max_new_tokens=200,
                 strategy="temperature", temperature=0.8, top_k=40,
                 device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_ids = tokenizer.encode(prompt) or [0]

    with torch.no_grad():
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits, hidden = model(x)
        last_logits = logits[0, -1, :]

    generated = list(input_ids)
    for _ in range(max_new_tokens):
        next_id = _sample(last_logits, strategy, temperature, top_k)
        generated.append(next_id)
        with torch.no_grad():
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            last_logits = logits[0, -1, :]

    return tokenizer.decode(generated)


def generate_transformer(model, tokenizer, prompt, max_new_tokens=200,
                         strategy="temperature", temperature=0.8, top_k=40,
                         device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_ids = tokenizer.encode(prompt) or [0]
    generated = list(input_ids)
    max_ctx = model.max_seq_len

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx = generated[-max_ctx:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(x)
            last_logits = logits[0, -1, :]
            next_id = _sample(last_logits, strategy, temperature, top_k)
            generated.append(next_id)

    return tokenizer.decode(generated)


def generate_samples(model, tokenizer, prompts, model_type="rnn",
                     max_new_tokens=150, strategies=None, device=None):
    if strategies is None:
        strategies = ["greedy", "temperature", "top_k"]

    gen_fn = generate_rnn if model_type == "rnn" else generate_transformer
    results = {}
    for prompt in prompts:
        results[prompt] = {}
        for strategy in strategies:
            try:
                text = gen_fn(model, tokenizer, prompt,
                              max_new_tokens=max_new_tokens,
                              strategy=strategy, device=device)
                results[prompt][strategy] = text
            except Exception as e:
                results[prompt][strategy] = f"[Error: {e}]"
    return results


def _sample(logits, strategy, temperature=0.8, top_k=40):
    if strategy == "greedy":
        return int(logits.argmax().item())

    logits = logits / max(temperature, 1e-8)

    if strategy == "top_k":
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())
