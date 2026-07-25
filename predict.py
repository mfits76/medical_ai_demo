"""Interactive / CLI inference for the trained triage model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model import SpecialtyTextCNN
from src.tokenizer import Vocab, tokenize


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab(stoi=ckpt["vocab_stoi"], itos=ckpt["vocab_itos"])
    model = SpecialtyTextCNN(
        vocab_size=len(vocab),
        num_classes=len(ckpt["specialties"]),
        pad_id=vocab.pad_id,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, vocab, ckpt


@torch.no_grad()
def predict_note(
    text: str,
    model: SpecialtyTextCNN,
    vocab: Vocab,
    max_len: int,
    specialties: list[str],
    device: torch.device,
) -> dict:
    ids = vocab.encode(tokenize(text), max_len)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().item())
    ranking = sorted(
        (
            {"specialty": specialties[i], "probability": float(probs[i].item())}
            for i in range(len(specialties))
        ),
        key=lambda r: r["probability"],
        reverse=True,
    )
    return {
        "predicted_specialty": specialties[pred_id],
        "confidence": ranking[0]["probability"],
        "ranking": ranking,
        "disclaimer": (
            "Demo only - synthetic training data, not a medical device. "
            "No real patient data should be submitted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict specialty for a clinical note")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/model.pt"),
    )
    parser.add_argument(
        "--text",
        type=str,
        default=(
            "Patient berichtet ueber belastungsabhaengige Brustschmerzen und Dyspnoe. "
            "Troponin leicht erhoeht."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device)
    result = predict_note(
        args.text,
        model,
        vocab,
        ckpt["max_len"],
        ckpt["specialties"],
        device,
    )

    print(f"Note: {args.text}\n")
    print(f"Predicted: {result['predicted_specialty']} ({result['confidence']:.1%})")
    print("Ranking:")
    for row in result["ranking"]:
        print(f"  {row['specialty']:16s} {row['probability']:.1%}")
    print(f"\n{result['disclaimer']}")


if __name__ == "__main__":
    main()
