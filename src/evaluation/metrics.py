import math
import torch


def compute_perplexity(loss):
    if loss > 20.0:
        return 9999.0
    try:
        return min(math.exp(loss), 9999.0)
    except OverflowError:
        return 9999.0


def compute_loss_on_loader(model, data_loader, device, model_type="rnn"):
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
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))
            total_loss += loss.item() * B * T
            total_tokens += B * T

    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


def evaluate_model(model, data_loader, device, model_type="rnn"):
    loss = compute_loss_on_loader(model, data_loader, device, model_type)
    return {"loss": loss, "perplexity": compute_perplexity(loss)}


def bits_per_char(loss):
    return loss / math.log(2)
