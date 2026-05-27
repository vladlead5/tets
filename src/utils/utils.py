"""
Вспомогательные функции: seed, устройство, сохранение/загрузка.
"""

import os
import random
import json
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Фиксируем все источники случайности для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # детерминированные алгоритмы (немного медленнее, зато reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Возвращает лучшее доступное устройство."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Количество обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, val_loss, path: str):
    """Сохраняем чекпоинт модели."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }, path)
    print(f"  [checkpoint] сохранён → {path}")


def load_checkpoint(model, optimizer, path: str):
    """Загружаем чекпоинт модели."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["val_loss"]


def save_metrics(metrics: dict, path: str):
    """Сохраняем метрики в JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def plot_loss_curves(
    train_losses: list,
    val_losses: list,
    title: str = "Loss",
    save_path: str = None,
):
    """Строим и сохраняем кривые обучения."""
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    ax.plot(epochs, val_losses, "r-o", markersize=4, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  [plot] сохранён → {save_path}")
    plt.show()
    plt.close()


def plot_perplexity_curves(
    train_ppls: list,
    val_ppls: list,
    title: str = "Perplexity",
    save_path: str = None,
):
    """Строим кривые перплексии."""
    epochs = range(1, len(train_ppls) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_ppls, "b-o", markersize=4, label="Train PPL")
    ax.plot(epochs, val_ppls, "r-o", markersize=4, label="Val PPL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexity")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  [plot] сохранён → {save_path}")
    plt.show()
    plt.close()


def plot_comparison(
    model_names: list,
    metric_values: list,
    metric_name: str = "Val Loss",
    save_path: str = None,
):
    """Столбчатая диаграмма для сравнения моделей."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
    bars = ax.bar(model_names, metric_values, color=colors[:len(model_names)], edgecolor="black", linewidth=0.7)

    for bar, val in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(metric_values),
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_ylabel(metric_name)
    ax.set_title(f"Сравнение моделей по {metric_name}")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close()
