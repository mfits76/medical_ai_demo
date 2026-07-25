"""Train a clinical-note specialty triage model (medical-style PoC)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data import ID2LABEL, SPECIALTIES, generate_notes
from src.model import SpecialtyTextCNN
from src.tokenizer import build_vocab
from src.train_utils import NoteDataset, evaluate, train_model


def plot_confusion(cm: list[list[int]], labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Specialty triage — confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clinical note triage model")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=48)
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    notes = generate_notes(n_per_class=args.n_per_class, seed=args.seed)
    texts = [n.text for n in notes]
    labels = [n.label_id for n in notes]

    x_train, x_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=0.3, random_state=args.seed, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    vocab = build_vocab(x_train, min_freq=2)
    train_ds = NoteDataset(x_train, y_train, vocab, args.max_len)
    val_ds = NoteDataset(x_val, y_val, vocab, args.max_len)
    test_ds = NoteDataset(x_test, y_test, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = SpecialtyTextCNN(
        vocab_size=len(vocab),
        num_classes=len(SPECIALTIES),
        pad_id=vocab.pad_id,
    )

    result = train_model(model, train_loader, val_loader, device, epochs=args.epochs)
    test_metrics = evaluate(model, test_loader, device)
    print(f"\nBest val accuracy: {result.best_val_acc:.3f}")
    print(f"Test accuracy:     {test_metrics['acc']:.3f}")

    model.eval()
    all_preds: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(batch_y.tolist())

    report = classification_report(
        all_true,
        all_preds,
        target_names=list(SPECIALTIES),
        digits=3,
    )
    print("\nClassification report:\n", report)

    cm = confusion_matrix(all_true, all_preds).tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, list(SPECIALTIES), args.out_dir / "confusion_matrix.png")

    checkpoint = {
        "model_state": model.state_dict(),
        "vocab_stoi": vocab.stoi,
        "vocab_itos": vocab.itos,
        "max_len": args.max_len,
        "specialties": list(SPECIALTIES),
        "id2label": ID2LABEL,
        "metrics": {
            "best_val_acc": result.best_val_acc,
            "test_acc": test_metrics["acc"],
            "history": result.history,
        },
    }
    ckpt_path = args.out_dir / "model.pt"
    torch.save(checkpoint, ckpt_path)

    meta = {
        "device": str(device),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "vocab_size": len(vocab),
        "test_accuracy": test_metrics["acc"],
        "best_val_accuracy": result.best_val_acc,
        "checkpoint": str(ckpt_path),
        "confusion_matrix": str(args.out_dir / "confusion_matrix.png"),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (args.out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved artifacts to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
