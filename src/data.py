"""Synthetic German clinical notes for specialty triage.

Uses fictional patients and template phrases only — no real PHI.
This mirrors a hospital documentation / routing use case while staying
DSGVO-safe for demos and interviews.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SPECIALTIES = (
    "Kardiologie",
    "Neurologie",
    "Orthopaedie",
    "Innere_Medizin",
    "Notfallmedizin",
)

LABEL2ID = {name: i for i, name in enumerate(SPECIALTIES)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

_TEMPLATES: dict[str, list[str]] = {
    "Kardiologie": [
        "Patient berichtet ueber belastungsabhaengige Brustschmerzen und Dyspnoe.",
        "Bekannte koronare Herzkrankheit, aktuell instabile Angina pectoris.",
        "EKG zeigt ST-Senkungen, Troponin leicht erhoeht, Verdacht auf ACS.",
        "Palpitationen und unregelmaessiger Puls, V. a. Vorhofflimmern.",
        "Nach Herzinsuffizienz-Dekompensation: Beinoedeme und Orthoopnoe.",
        "Belastungs-EKG pathologisch, weiter kardiologische Abklaerung noetig.",
        "Z. n. Stentimplantation, heute erneut thorakales Engegefuehl.",
        "Blutdruckkrise mit Kopfschmerzen, RR 210/110 mmHg.",
    ],
    "Neurologie": [
        "Akute Halbseitenschwaeche rechts und Sprachstoerung seit heute morgen.",
        "Wiederholte Kopfschmerzen mit Sehstoerungen, V. a. Migraine avec aura.",
        "Schwindel, Doppelbilder und Gangunsicherheit seit drei Tagen.",
        "Epileptischer Anfall in der Vorgeschichte, heute erneut Konvulsion.",
        "Taubheitsgefuehl in beiden Haenden, V. a. Polyneuropathie.",
        "Gedaechtnisstoerungen und Orientierungsprobleme, dementielle Abklaerung.",
        "Tremor der rechten Hand und Bradykinese, V. a. Parkinson-Syndrom.",
        "Plötzliche Gesichtslähmung links, V. a. Fazialisparese.",
    ],
    "Orthopaedie": [
        "Chronische Rueckenschmerzen lumbal mit Ausstrahlung ins Bein.",
        "Kniegelenksschwellung nach Distorsion, Bewegung stark eingeschraenkt.",
        "Hueftgelenkschmerzen rechts, belastungsabhaengig, V. a. Koxarthrose.",
        "Schulterluxation reponiert, nun Instabilitaetsgefuehl und Schmerzen.",
        "Frakturverdacht distal Radius nach Sturz auf die Hand.",
        "Bandscheibenvorfall bekannt, jetzt progrediente Beinschwaeche.",
        "Achillessehnenruptur nach Sportunfall, starke Druckschmerzhaftigkeit.",
        "Gonarthrose beidseits, Indikation zur Gelenkersatz-Evaluation.",
    ],
    "Innere_Medizin": [
        "Seit Wochen unklare Gewichtsabnahme, Nachtschweiss und Muedigkeit.",
        "Neu diagnostizierter Diabetes mellitus Typ 2, Blutzucker entgleist.",
        "Oberbauchschmerzen und Erbrechen, V. a. Gastritis oder Ulkus.",
        "Erhoehete Leberwerte, Abklaerung einer hepatischen Ursache.",
        "Rezidivierende Harnwegsinfekte, aktuell Fieber und Dysurie.",
        "Anämie unklarer Genese, weiter internistische Diagnostik geplant.",
        "Hypothyreose unter Substitution, aktuell weiterhin Antriebslosigkeit.",
        "Unklarer Bauchschmerz diffus, Differenzialdiagnose breit.",
    ],
    "Notfallmedizin": [
        "Polytrauma nach Verkehrsunfall, mehrfach verletzt, vital bedroht.",
        "Akute Atemnot und Zyanose, Verdacht auf Lungenembolie.",
        "Bewusstlosigkeit unklarer Ursache, GCS 8 bei Aufnahme.",
        "Schwere anaphylaktische Reaktion nach Medikamenteneinnahme.",
        "Massive gastrointestinale Blutung mit Kreislaufinstabilitaet.",
        "Status epilepticus, bislang nicht durchbrechbar.",
        "Akutes Abdomen mit Abwehrspannung, dringende Abklaerung.",
        "Sepsisverdacht: Fieber, Tachykardie und Hypotonie.",
    ],
}

_NOISE = [
    "Vorgeschichte unauffaellig.",
    "Allergien nicht bekannt.",
    "Medikation wird erhoben.",
    "Vitalparameter stabil.",
    "Laborwerte ausstehend.",
    "Angehoerige informiert.",
    "Dokumentation vorlaeufig.",
]


@dataclass(frozen=True)
class ClinicalNote:
    text: str
    specialty: str

    @property
    def label_id(self) -> int:
        return LABEL2ID[self.specialty]


def generate_notes(n_per_class: int = 80, seed: int = 42) -> list[ClinicalNote]:
    """Build a balanced synthetic corpus of German clinical notes."""
    rng = random.Random(seed)
    notes: list[ClinicalNote] = []

    for specialty, templates in _TEMPLATES.items():
        for _ in range(n_per_class):
            base = rng.choice(templates)
            extras = rng.sample(_NOISE, k=rng.randint(1, 3))
            # Light paraphrasing via shuffle + optional second template fragment
            if rng.random() < 0.35:
                fragment = rng.choice(templates)
                text = f"{base} {fragment} {' '.join(extras)}"
            else:
                text = f"{base} {' '.join(extras)}"
            notes.append(ClinicalNote(text=text, specialty=specialty))

    rng.shuffle(notes)
    return notes
