"""
Универсальный training loop для всех моделей.

Поддерживает:
  - RNN / LSTM / BiLSTM (имеют hidden state)
  - Transformer (нет hidden state)
  - gradient clipping
  - checkpoint saving (лучшая модель по val loss)
  - early stopping
  - learning rate scheduler
  - mixed precision (если доступна CUDA)
  - логирование метрик
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.utils import save_checkpoint
from src.evaluation.metrics import compute_perplexity


def train_epoch_rnn(model, loader, optimizer, criterion, device, clip_grad=1.0, scaler=None):
    """Один эпох обучения для RNN/LSTM/BiLSTM."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        B, T = x.shape

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=(scaler is not None)):
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

    return total_loss / total_tokens


def eval_epoch_rnn(model, loader, criterion, device):
    """Валидация для RNN/LSTM/BiLSTM."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B, T = x.shape
            logits, _ = model(x)
            loss = criterion(logits.view(B * T, -1), y.view(B * T))
            total_loss += loss.item() * B * T
            total_tokens += B * T

    return total_loss / total_tokens


def train_epoch_transformer(model, loader, optimizer, criterion, device, clip_grad=1.0, scaler=None):
    """Один эпох обучения для Transformer."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        B, T = x.shape

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type if device.type != "mps" else "cpu", enabled=(scaler is not None)):
            logits = model(x)
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

    return total_loss / total_tokens


def eval_epoch_transformer(model, loader, criterion, device):
    """Валидация для Transformer."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B, T = x.shape
            logits = model(x)
            loss = criterion(logits.view(B * T, -1), y.view(B * T))
            total_loss += loss.item() * B * T
            total_tokens += B * T

    return total_loss / total_tokens


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str = "model",
    model_type: str = "rnn",   # "rnn" или "transformer"
    num_epochs: int = 10,
    lr: float = 1e-3,
    clip_grad: float = 1.0,
    patience: int = 5,
    checkpoint_dir: str = "outputs/checkpoints",
    device: torch.device = None,
    use_amp: bool = False,
) -> dict:
    """
    Полный training loop с early stopping и checkpoint saving.

    Returns:
        history: dict с ключами train_loss, val_loss, train_ppl, val_ppl, epoch_times
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # mixed precision только для CUDA
    scaler = torch.cuda.GradScaler() if (use_amp and device.type == "cuda") else None

    if model_type == "transformer":
        train_fn = train_epoch_transformer
        eval_fn = eval_epoch_transformer
    else:
        train_fn = train_epoch_rnn
        eval_fn = eval_epoch_rnn

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
        "epoch_times": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    print(f"\n{'='*50}")
    print(f"Обучаем модель: {model_name}")
    print(f"Устройство: {device} | Эпох: {num_epochs} | LR: {lr}")
    print(f"{'='*50}")

    for epoch in range(1, num_epochs + 1):
        t_start = time.time()

        train_loss = train_fn(model, train_loader, optimizer, criterion, device, clip_grad, scaler)
        val_loss = eval_fn(model, val_loader, criterion, device)

        epoch_time = time.time() - t_start
        train_ppl = compute_perplexity(train_loss)
        val_ppl = compute_perplexity(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)
        history["epoch_times"].append(epoch_time)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_ppl={val_ppl:.1f} | time={epoch_time:.1f}s"
        )

        scheduler.step(val_loss)

        # сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [early stopping] val_loss не улучшается {patience} эпох подряд")
                break

    print(f"\nЛучший val_loss: {best_val_loss:.4f}")
    print(f"Чекпоинт сохранён: {best_path}\n")

    return history
