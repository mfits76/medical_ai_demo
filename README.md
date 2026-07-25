# Clinical Note Triage — PyTorch PoC for medical AI Engineer

[![Demo](https://img.shields.io/badge/demo-live-blue)](https://mfits76.github.io/medical_ai_demo/)

Interview / portfolio example for the **AI Engineer (w/m/d)** role at **medical Service IT GmbH**: a small but complete path from **prototype → trained model → integrable API**, focused on a hospital-relevant use case (routing documentation to the right specialty).

> **Not a medical device.** Trained only on **synthetic German notes**. Do not send real patient data.

**[Live web demo](https://mfits76.github.io/medical_ai_demo/)** — no install, uses pretrained weights in the browser

## How it works

### Pipeline

1. **Offline (PyTorch):** `train.py` trains the model and writes `artifacts/` (`model.pt`, metrics, plots).
2. **Export:** `export_web_model.py` (also run at the end of training) writes `static/model.json`.
3. **Web app:** static HTML/JS loads `model.json` and runs inference in the browser (GitHub Pages or `start_web.bat`).

No PyTorch is needed to *use* the web demo — only to (re)train offline.

### Training data

All training text is **synthetic** (no real PHI). The source of truth is:

`data/specialty_dictionary.json`

| Field | Role |
| --- | --- |
| `specialties` | Ordered list of class names (e.g. `Kardiologie`) |
| `templates` | Specialty-specific example sentences for each class |
| `noise` | Generic filler phrases shared across classes |

At train time, `src/data.py` builds notes by:

1. Picking a specialty template as the clinical signal  
2. Appending **1–3 random `noise` phrases** (boilerplate such as “Vitalparameter stabil.”)  
3. Sometimes (35%) also appending a second specialty template  

Noise is **not** its own label — it only makes notes look more like real documentation so the model learns to focus on clinical wording, not memorize short clean lines.

### Prediction

There is **no exact string lookup** against the dictionary. Every input goes through the same path:

1. Tokenize the note and map words to a learned vocabulary (`unknown` → `<unk>`)  
2. Run the TextCNN (embeddings + short phrase patterns)  
3. Softmax → a **probability for each specialty** (the ranking in the UI)

Exact or near-template wording usually scores high because those phrases were common in training. New wording can still work if it shares enough medical terms; unfamiliar vocabulary tends to lower confidence.

## How to expand

Edit `data/specialty_dictionary.json`, then retrain.

### Add more example phrases for an existing specialty

Append strings to that specialty’s list under `templates`:

```json
"Kardiologie": [
  "Patient berichtet ueber belastungsabhaengige Brustschmerzen und Dyspnoe.",
  "Your new example sentence here."
]
```

### Add more filler (noise)

Append to the `noise` array:

```json
"noise": [
  "Vorgeschichte unauffaellig.",
  "Your new boilerplate sentence here."
]
```

### Add a new specialty

1. Add the name to `specialties`  
2. Add a matching non-empty list under `templates`  
3. Retrain (class count changes, so a new model is required)

```json
"specialties": ["Kardiologie", "Neurologie", "Orthopaedie", "Innere_Medizin", "Notfallmedizin", "Dermatologie"],
"templates": {
  "Dermatologie": [
    "Juckender Hautausschlag seit einer Woche, V. a. Ekzem."
  ]
}
```

### Retrain after edits

```powershell
.\start_train.bat
```

This refreshes `artifacts/` and `static/model.json`. Optional: more generated notes per class without new phrases:

```powershell
.\start_train.bat --n-per-class 200
```

Then commit/push if you want GitHub Pages and the repo artifacts updated.

## Why this example maps to the job

| medical expectation | What this PoC shows |
| --- | --- |
| KI for Klinikalltag / Dokumentation | Specialty triage from free-text notes |
| PoC → produktiver Einsatz | `train.py` → artifacts → static web / FastAPI |
| Python + PyTorch | TextCNN classifier end-to-end |
| Systemintegration (APIs) | Optional FastAPI `/triage` microservice |
| DSGVO / sensible Daten | Synthetic corpus + explicit no-PHI checkbox |
| Modellqualität & Monitoring mindset | Val/test metrics, classification report, confusion matrix |

## Architecture

```
offline PyTorch train → artifacts/model.pt
                         ↓ export_web_model.py
                    static/model.json
                         ↓
              GitHub Pages / start_web.bat (browser inference)
```

Specialties: `Kardiologie`, `Neurologie`, `Orthopaedie`, `Innere_Medizin`, `Notfallmedizin`.

## Setup (Windows) — only needed to retrain or use the local API

```powershell
cd c:\Users\mfits\Documents\ai_example
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train offline (PyTorch → artifacts/)

After editing `data/specialty_dictionary.json` (see **How to expand** above):

```powershell
.\start_train.bat
```

Or:

```powershell
python train.py
```

Outputs:

- `artifacts/model.pt` — weights + vocab + metadata  
- `artifacts/metrics.json` — accuracies  
- `artifacts/classification_report.txt`  
- `artifacts/confusion_matrix.png`  
- `static/model.json` — browser export for the web demo  

Re-export only:

```powershell
python export_web_model.py
```

## Web UI

### Live on GitHub Pages (no local server)

Open **https://mfits76.github.io/medical_ai_demo/**

### Local static / API host

Double-click `start_web.bat` (opens http://127.0.0.1:8000), or:

```powershell
.\start_web.bat
```

The page still does browser inference from `model.json`. FastAPI also exposes `/health` and `/triage` if you want an API demo.

## Desktop UI

Double-click `start_ui.bat` (Tkinter; loads `artifacts/model.pt` with PyTorch).

## Predict (CLI)

```powershell
python predict.py --text "Akute Halbseitenschwaeche rechts und Sprachstoerung seit heute morgen."
```

## Project layout

```
medical_ai_demo/
  data/
    specialty_dictionary.json  # expandable training phrases
  train.py              # offline PyTorch training → artifacts/
  export_web_model.py   # artifacts/model.pt → static/model.json
  predict.py            # CLI inference
  ui.py                 # Tkinter desktop UI
  api.py                # optional FastAPI host for static UI + /triage
  start_train.bat
  start_ui.bat
  start_web.bat
  static/               # GitHub Pages web app
  artifacts/            # pretrained model + metrics
  src/
```
