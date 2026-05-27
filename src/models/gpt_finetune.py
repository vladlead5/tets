import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2TextDataset(Dataset):
    def __init__(self, token_ids, seq_len=128):
        self.data = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_pretrained_gpt2(model_name="distilgpt2"):
    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    return model, tokenizer


def finetune_gpt2(text, output_dir="outputs/checkpoints",
                  model_name="distilgpt2", seq_len=128,
                  batch_size=8, num_epochs=3, lr=5e-5, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_pretrained_gpt2(model_name)
    model = model.to(device)

    print("Tokenizing text...")
    token_ids = tokenizer.encode(text)
    print(f"  Total tokens: {len(token_ids):,}")

    split = int(len(token_ids) * 0.9)
    train_dataset = GPT2TextDataset(token_ids[:split], seq_len)
    val_dataset = GPT2TextDataset(token_ids[split:], seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_ppl": [], "val_ppl": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = model(input_ids=x, labels=y).loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        train_loss = total_loss / steps
        train_ppl = min(torch.exp(torch.tensor(train_loss)).item(), 9999.0)

        model.eval()
        val_total, val_steps = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_total += model(input_ids=x, labels=y).loss.item()
                val_steps += 1
        val_loss = val_total / max(val_steps, 1)
        val_ppl = min(torch.exp(torch.tensor(val_loss)).item(), 9999.0)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}")

        ckpt_path = os.path.join(output_dir, f"gpt2_epoch{epoch}.pt")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

    return model, tokenizer, history


def generate_gpt2(model, tokenizer, prompt, max_new_tokens=100,
                  temperature=0.8, top_k=50, device=None):
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
    return tokenizer.decode(output[0], skip_special_tokens=True)
