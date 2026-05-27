import os
import time
import torch
import torch.nn as nn
from src.utils.utils import save_checkpoint
from src.evaluation.metrics import compute_perplexity


def _train_epoch(model, loader, optimizer, criterion, device,
                 clip_grad, is_transformer, scaler=None):
    model.train()
    total_loss, total_tokens = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        B, T = x.shape
        optimizer.zero_grad()

        if is_transformer:
            logits = model(x)
        else:
            logits, _ = model(x)

        loss = criterion(logits.view(B * T, -1), y.view(B * T))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        total_loss += loss.item() * B * T
        total_tokens += B * T

    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


def _eval_epoch(model, loader, criterion, device, is_transformer):
    model.eval()
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B, T = x.shape
            if is_transformer:
                logits = model(x)
            else:
                logits, _ = model(x)
            loss = criterion(logits.view(B * T, -1), y.view(B * T))
            total_loss += loss.item() * B * T
            total_tokens += B * T

    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


def train_model(model, train_loader, val_loader, model_name="model",
                model_type="rnn", num_epochs=10, lr=1e-3, clip_grad=1.0,
                patience=5, checkpoint_dir="outputs/checkpoints",
                device=None, use_amp=False):
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    is_transformer = model_type == "transformer"
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = (
        torch.cuda.GradScaler()
        if use_amp and device.type == "cuda"
        else None
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_ppl": [], "val_ppl": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_ckpt = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"Training: {model_name}  device: {device}  epochs: {num_epochs}")
    print(sep)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = _train_epoch(model, train_loader, optimizer, criterion,
                                  device, clip_grad, is_transformer, scaler)
        val_loss = _eval_epoch(model, val_loader, criterion, device,
                               is_transformer)
        elapsed = time.time() - t0

        train_ppl = compute_perplexity(train_loss)
        val_ppl = compute_perplexity(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)
        history["epoch_times"].append(elapsed)

        print(
            f"Epoch {epoch:02d}/{num_epochs}  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"ppl={val_ppl:.1f}  time={elapsed:.1f}s"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  early stopping after {patience} epochs")
                break

    print(f"\nBest val_loss: {best_val_loss:.4f}  →  {best_ckpt}\n")
    return history


def train_epoch_rnn(model, loader, optimizer, criterion,
                    device, clip_grad=1.0):
    return _train_epoch(model, loader, optimizer, criterion, device,
                        clip_grad, is_transformer=False)


def eval_epoch_rnn(model, loader, criterion, device):
    return _eval_epoch(model, loader, criterion, device, is_transformer=False)


def train_epoch_transformer(model, loader, optimizer, criterion,
                            device, clip_grad=1.0):
    return _train_epoch(model, loader, optimizer, criterion, device,
                        clip_grad, is_transformer=True)


def eval_epoch_transformer(model, loader, criterion, device):
    return _eval_epoch(model, loader, criterion, device, is_transformer=True)
