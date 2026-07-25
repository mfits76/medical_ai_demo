# Clinical Note Triage — PyTorch PoC for medical AI Engineer

[![Demo](https://img.shields.io/badge/demo-live-blue)](https://mfits76.github.io/ai_example/)

Interview / portfolio example for the **AI Engineer (w/m/d)** role at **medical Service IT GmbH**: a small but complete path from **prototype → trained model → integrable API**, focused on a hospital-relevant use case (routing documentation to the right specialty).

> **Not a medical device.** Trained only on **synthetic German notes**. Do not send real patient data.

**[Live web demo](https://mfits76.github.io/ai_example/)** — no install, uses pretrained weights in the browser

## How it works

1. **Offline (PyTorch):** `train.py` trains the model and writes `artifacts/` (`model.pt`, metrics, plots).
2. **Export:** `export_web_model.py` (also run at the end of training) writes `static/model.json`.
3. **Web app:** static HTML/JS loads `model.json` and runs inference in the browser (GitHub Pages or `start_web.bat`).

No PyTorch is needed to *use* the web demo — only to (re)train offline.

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

Open **https://mfits76.github.io/ai_example/**

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
ai_example/
  train.py              # offline PyTorch training → artifacts/
  export_web_model.py   # artifacts/model.pt → static/model.json
  predict.py            # CLI inference
  ui.py                 # Tkinter desktop UI
  api.py                # optional FastAPI host for static UI + /triage
  start_ui.bat
  start_web.bat
  static/               # GitHub Pages web app
  artifacts/            # pretrained model + metrics
  src/
```
