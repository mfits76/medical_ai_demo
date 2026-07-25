"""Tkinter UI for clinical note specialty triage demo."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import torch

from predict import load_checkpoint, predict_note

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "model.pt"

EXAMPLES = {
    "Kardiologie": (
        "Patient berichtet ueber belastungsabhaengige Brustschmerzen und Dyspnoe. "
        "Troponin leicht erhoeht."
    ),
    "Neurologie": (
        "Akute Halbseitenschwaeche rechts und Sprachstoerung seit heute morgen."
    ),
    "Orthopaedie": (
        "Chronische Rueckenschmerzen lumbal mit Ausstrahlung ins Bein."
    ),
    "Innere Medizin": (
        "Seit Wochen unklare Gewichtsabnahme, Nachtschweiss und Muedigkeit."
    ),
    "Notfallmedizin": (
        "Akute Atemnot und Zyanose, Verdacht auf Lungenembolie. Vitalparameter instabil."
    ),
}


class TriageApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Clinical Note Triage — medical PoC")
        self.geometry("780x620")
        self.minsize(700, 560)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.ckpt = None

        self._build()
        self.after(50, self._load_model)

    def _build(self) -> None:
        pad = {"padx": 14, "pady": 8}

        header = ttk.Frame(self)
        header.pack(fill="x", **pad)
        ttk.Label(
            header,
            text="Specialty Triage Demo",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Synthetic German clinical notes only — not for real patient data.",
            foreground="#555555",
        ).pack(anchor="w")

        self.status_var = tk.StringVar(value="Loading model...")
        ttk.Label(header, textvariable=self.status_var).pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, **pad)

        ttk.Label(body, text="Clinical note").pack(anchor="w")
        self.text = tk.Text(body, height=8, wrap="word", font=("Consolas", 11))
        self.text.pack(fill="x", pady=(4, 8))
        self.text.insert("1.0", EXAMPLES["Kardiologie"])

        controls = ttk.Frame(body)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, text="Example:").pack(side="left")
        self.example_var = tk.StringVar(value="Kardiologie")
        example_box = ttk.Combobox(
            controls,
            textvariable=self.example_var,
            values=list(EXAMPLES.keys()),
            state="readonly",
            width=18,
        )
        example_box.pack(side="left", padx=(6, 12))
        example_box.bind("<<ComboboxSelected>>", self._on_example)

        self.predict_btn = ttk.Button(
            controls, text="Predict specialty", command=self._predict, state="disabled"
        )
        self.predict_btn.pack(side="left")

        result_frame = ttk.LabelFrame(body, text="Result")
        result_frame.pack(fill="both", expand=True)

        self.prediction_var = tk.StringVar(value="—")
        ttk.Label(
            result_frame,
            textvariable=self.prediction_var,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 4))

        columns = ("specialty", "probability", "bar")
        self.tree = ttk.Treeview(
            result_frame, columns=columns, show="headings", height=6
        )
        self.tree.heading("specialty", text="Specialty")
        self.tree.heading("probability", text="Probability")
        self.tree.heading("bar", text="")
        self.tree.column("specialty", width=160, anchor="w")
        self.tree.column("probability", width=100, anchor="center")
        self.tree.column("bar", width=280, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.disclaimer_var = tk.StringVar(value="")
        ttk.Label(
            result_frame,
            textvariable=self.disclaimer_var,
            wraplength=700,
            foreground="#666666",
        ).pack(anchor="w", padx=10, pady=(0, 10))

    def _load_model(self) -> None:
        if not CHECKPOINT.exists():
            self.status_var.set("Model missing — run train.py first.")
            messagebox.showerror(
                "Model not found",
                f"Expected checkpoint:\n{CHECKPOINT}\n\nRun train.py before opening the UI.",
            )
            return
        try:
            self.model, self.vocab, self.ckpt = load_checkpoint(CHECKPOINT, self.device)
            self.status_var.set(f"Model ready ({self.device})")
            self.predict_btn.configure(state="normal")
            self._predict()
        except Exception as exc:  # noqa: BLE001 — show any load failure in UI
            self.status_var.set("Failed to load model")
            messagebox.showerror("Load error", str(exc))

    def _on_example(self, _event=None) -> None:
        key = self.example_var.get()
        self.text.delete("1.0", "end")
        self.text.insert("1.0", EXAMPLES[key])

    def _predict(self) -> None:
        if self.model is None:
            return
        note = self.text.get("1.0", "end").strip()
        if len(note) < 10:
            messagebox.showwarning("Input too short", "Please enter a longer clinical note.")
            return

        result = predict_note(
            note,
            self.model,
            self.vocab,
            self.ckpt["max_len"],
            self.ckpt["specialties"],
            self.device,
        )

        self.prediction_var.set(
            f"{result['predicted_specialty']}  ({result['confidence']:.1%})"
        )
        self.disclaimer_var.set(result["disclaimer"])

        for row in self.tree.get_children():
            self.tree.delete(row)
        for item in result["ranking"]:
            pct = item["probability"]
            bar = "█" * int(round(pct * 20))
            self.tree.insert(
                "",
                "end",
                values=(item["specialty"], f"{pct:.1%}", bar),
            )


def main() -> None:
    app = TriageApp()
    app.mainloop()


if __name__ == "__main__":
    main()
