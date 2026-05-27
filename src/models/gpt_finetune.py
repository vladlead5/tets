"""
Fine-tuning предобученной GPT-2 модели (distilgpt2) на нашем датасете.

Используем HuggingFace Transformers.
Дообучаем все слои (full fine-tuning), т.к. датасет небольшой
и нам важно адаптировать стиль генерации.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from tqdm import tqdm


class GPT2TextDataset(Dataset):
    """Датасет для fine-tuning GPT-2: sliding window по токенам."""

    def __init__(self, token_ids: list, seq_len: int = 128):
        self.data = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_pretrained_gpt2(model_name: str = "distilgpt2"):
    """Загружаем предобученный GPT-2 и tokenizer."""
    print(f"Загружаем {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # GPT-2 не имеет pad token — используем eos
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Параметров: {n_params:,}")
    return model, tokenizer


def finetune_gpt2(
    text: str,
    output_dir: str = "outputs/checkpoints",
    model_name: str = "distilgpt2",
    seq_len: int = 128,
    batch_size: int = 8,
    num_epochs: int = 3,
    lr: float = 5e-5,
    device: torch.device = None,
):
    """
    Дообучаем GPT-2 на нашем тексте.

    Returns:
        model, tokenizer, history (dict с потерями)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_pretrained_gpt2(model_name)
    model = model.to(device)

    # токенизируем текст GPT-2 tokenizer'ом
    print("Токенизируем текст...")
    token_ids = tokenizer.encode(text)
    print(f"  Токенов всего: {len(token_ids):,}")

    # разбиваем на train/val
    split = int(len(token_ids) * 0.9)
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    train_dataset = GPT2TextDataset(train_ids, seq_len)
    val_dataset = GPT2TextDataset(val_ids, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": [], "train_ppl": [], "val_ppl": []}

    for epoch in range(1, num_epochs + 1):
        # обучение
        model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(train_loader, desc=f"GPT2 Epoch {epoch}/{num_epochs} [train]")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": f"{total_loss / steps:.4f}"})

        train_loss = total_loss / steps
        train_ppl = min(torch.exp(torch.tensor(train_loss)).item(), 9999.0)

        # валидация
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(input_ids=x, labels=y)
                val_loss_total += outputs.loss.item()
                val_steps += 1

        val_loss = val_loss_total / max(val_steps, 1)
        val_ppl = min(torch.exp(torch.tensor(val_loss)).item(), 9999.0)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f}")

        # сохраняем чекпоинт
        ckpt_path = os.path.join(output_dir, f"gpt2_finetune_epoch{epoch}.pt")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    return model, tokenizer, history


def generate_gpt2(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = None,
) -> str:
    """Генерируем текст fine-tuned GPT-2."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated
