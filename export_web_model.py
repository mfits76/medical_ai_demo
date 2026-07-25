"""Export artifacts/model.pt to static/model.json for browser inference.

Training stays offline in PyTorch. The web demo loads this JSON only —
no server-side PyTorch required on GitHub Pages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def export(checkpoint: Path, out_path: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]

    convs = []
    i = 0
    while f"convs.{i}.weight" in state:
        convs.append(
            {
                "weight": state[f"convs.{i}.weight"].tolist(),
                "bias": state[f"convs.{i}.bias"].tolist(),
            }
        )
        i += 1

    payload = {
        "specialties": ckpt["specialties"],
        "max_len": ckpt["max_len"],
        "pad_id": ckpt["vocab_stoi"]["<pad>"],
        "unk_id": ckpt["vocab_stoi"]["<unk>"],
        "stoi": ckpt["vocab_stoi"],
        "embed_dim": state["embedding.weight"].shape[1],
        "num_filters": state["convs.0.weight"].shape[0],
        "kernel_sizes": [int(state[f"convs.{i}.weight"].shape[-1]) for i in range(len(convs))],
        "embedding": state["embedding.weight"].tolist(),
        "convs": convs,
        "fc": {
            "weight": state["fc.weight"].tolist(),
            "bias": state["fc.bias"].tolist(),
        },
        "disclaimer": (
            "Demo only - synthetic training data, not a medical device. "
            "No real patient data should be submitted."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model for static web demo")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--out", type=Path, default=Path("static/model.json"))
    args = parser.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
