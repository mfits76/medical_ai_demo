"""Dataset helpers and train/eval loops."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from .model import SpecialtyTextCNN
from .tokenizer import Vocab, tokenize


class NoteDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocab, max_len: int) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.vocab.encode(tokenize(self.texts[idx]), self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    best_val_acc: float


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def evaluate(model: SpecialtyTextCNN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    criterion = torch.nn.CrossEntropyLoss()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        bs = batch_y.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, batch_y) * bs
        n += bs

    return {"loss": total_loss / n, "acc": total_acc / n}


def train_model(
    model: SpecialtyTextCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 8,
    lr: float = 1e-3,
) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            bs = batch_y.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy(logits, batch_y) * bs
            n += bs

        train_metrics = {"loss": running_loss / n, "acc": running_acc / n}
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={row['train_loss']:.3f} train_acc={row['train_acc']:.3f} | "
            f"val_loss={row['val_loss']:.3f} val_acc={row['val_acc']:.3f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(history=history, best_val_acc=best_val_acc)
