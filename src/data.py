"""Synthetic German clinical notes for specialty triage.

Loads expandable templates from data/specialty_dictionary.json.
Uses fictional phrases only — no real PHI.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DICTIONARY_PATH = Path(__file__).resolve().parents[1] / "data" / "specialty_dictionary.json"


@lru_cache(maxsize=1)
def load_dictionary(path: str | None = None) -> dict:
    dict_path = Path(path) if path else DICTIONARY_PATH
    payload = json.loads(dict_path.read_text(encoding="utf-8"))

    specialties = payload["specialties"]
    templates = payload["templates"]
    noise = payload.get("noise", [])

    missing = [name for name in specialties if name not in templates]
    if missing:
        raise ValueError(f"Dictionary missing templates for: {missing}")

    empty = [name for name in specialties if not templates[name]]
    if empty:
        raise ValueError(f"Dictionary has empty template lists for: {empty}")

    extra = sorted(set(templates) - set(specialties))
    if extra:
        raise ValueError(
            f"Dictionary has templates for specialties not listed in "
            f"'specialties': {extra}"
        )

    return {
        "specialties": tuple(specialties),
        "templates": {k: list(v) for k, v in templates.items()},
        "noise": list(noise),
    }


def _dictionary() -> dict:
    return load_dictionary()


def get_specialties() -> tuple[str, ...]:
    return _dictionary()["specialties"]


# Eager defaults for existing imports (train.py uses SPECIALTIES at import time).
SPECIALTIES: tuple[str, ...] = get_specialties()
LABEL2ID = {name: i for i, name in enumerate(SPECIALTIES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}


def reload_dictionary(path: str | None = None) -> None:
    """Reload JSON and refresh module-level specialty maps (useful after edits)."""
    global SPECIALTIES, LABEL2ID, ID2LABEL
    load_dictionary.cache_clear()
    data = load_dictionary(path)
    SPECIALTIES = data["specialties"]
    LABEL2ID = {name: i for i, name in enumerate(SPECIALTIES)}
    ID2LABEL = {i: name for name, i in LABEL2ID.items()}


@dataclass(frozen=True)
class ClinicalNote:
    text: str
    specialty: str

    @property
    def label_id(self) -> int:
        return LABEL2ID[self.specialty]


def generate_notes(
    n_per_class: int = 80,
    seed: int = 42,
    dictionary_path: str | Path | None = None,
) -> list[ClinicalNote]:
    """Build a balanced synthetic corpus from the JSON dictionary."""
    if dictionary_path is not None:
        load_dictionary.cache_clear()
        data = load_dictionary(str(dictionary_path))
    else:
        data = _dictionary()

    templates: dict[str, list[str]] = data["templates"]
    noise: list[str] = data["noise"]
    rng = random.Random(seed)
    notes: list[ClinicalNote] = []

    for specialty, specialty_templates in templates.items():
        for _ in range(n_per_class):
            base = rng.choice(specialty_templates)
            extras = (
                rng.sample(noise, k=min(len(noise), rng.randint(1, 3)))
                if noise
                else []
            )
            if rng.random() < 0.35:
                fragment = rng.choice(specialty_templates)
                text = f"{base} {fragment} {' '.join(extras)}".strip()
            else:
                text = f"{base} {' '.join(extras)}".strip()
            notes.append(ClinicalNote(text=text, specialty=specialty))

    rng.shuffle(notes)
    return notes
