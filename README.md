# Clinical Note Triage — PyTorch PoC for medical AI Engineer

[![Open in GitHub Codespaces](https://img.shields.io/badge/Codespaces-Open-blue?logo=github)](https://codespaces.new/mfits76/ai_example?quickstart=1)

Interview / portfolio example for the **AI Engineer (w/m/d)** role at **medical Service IT GmbH**: a small but complete path from **prototype → trained model → integrable API**, focused on a hospital-relevant use case (routing documentation to the right specialty).

> **Not a medical device.** Trained only on **synthetic German notes**. Do not send real patient data.

**[Open in Codespaces](https://codespaces.new/mfits76/ai_example?quickstart=1)** (runs the Python web server in the cloud) · or use `start_web.bat` locally

> A local/cloud **Python server is required** for live triage. GitHub Pages only serves static files and cannot run PyTorch inference.

## Why this example maps to the job

| medical expectation | What this PoC shows |
| --- | --- |
| KI for Klinikalltag / Dokumentation | Specialty triage from free-text notes |
| PoC → produktiver Einsatz | `train.py` → artifacts → FastAPI `/triage` |
| Python + PyTorch | TextCNN classifier end-to-end |
| Systemintegration (APIs) | REST microservice with health check |
| DSGVO / sensible Daten | Synthetic corpus + explicit `confirm_no_phi` gate |
| Modellqualität & Monitoring mindset | Val/test metrics, classification report, confusion matrix |

## Architecture

```
synthetic notes → tokenizer/vocab → TextCNN (PyTorch)
                                      ↓
                         artifacts/model.pt + metrics
                                      ↓
         CLI predict.py  |  Tkinter ui.py  |  Web UI (api.py)
```

Specialties: `Kardiologie`, `Neurologie`, `Orthopaedie`, `Innere_Medizin`, `Notfallmedizin`.

## Setup (Windows)

```powershell
cd c:\Users\mfits\Documents\ai_example
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train

```powershell
python train.py
```

Outputs in `artifacts/`:

- `model.pt` — weights + vocab + metadata  
- `metrics.json` — accuracies  
- `classification_report.txt`  
- `confusion_matrix.png` — slide-ready figure  

## Desktop UI

Double-click `start_ui.bat`, or:

```powershell
.\start_ui.bat
```

## Web UI

### Option A — GitHub Codespaces (no local install)

1. Open **[Codespaces](https://codespaces.new/mfits76/ai_example?quickstart=1)**
2. In the terminal: `.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000`
3. Open the forwarded port **8000** in the browser

### Option B — Local (`start_web.bat`)

Double-click `start_web.bat` (opens http://127.0.0.1:8000), or:

```powershell
.\start_web.bat
```

Paste / pick an example note → **Predict specialty** → ranked probabilities.

## Predict (CLI)

```powershell
python predict.py --text "Akute Halbseitenschwaeche rechts und Sprachstoerung seit heute morgen."
```

## API only

```powershell
uvicorn api:app --reload --port 8000
```

Example request:

```powershell
curl -X POST http://127.0.0.1:8000/triage `
  -H "Content-Type: application/json" `
  -d '{"text":"EKG zeigt ST-Senkungen, Troponin leicht erhoeht.","confirm_no_phi":true}'
```

Interactive docs: http://127.0.0.1:8000/docs

## Talking points for the interview

1. **Use case** — triage supports documentation workflows and faster routing (Entscheidungsunterstützung), without claiming diagnosis.
2. **Modeling** — TextCNN is a deliberate PoC choice: fast, inspectable, good baseline before LLMs / fine-tuning.
3. **Metrics honesty** — near-perfect accuracy here is expected because notes are template-based and separable. On real clinical text you would expect noise, class imbalance, domain shift, and the need for calibration, error analysis, and human review.
4. **Next steps toward production** — replace synthetic data with de-identified hospital text under ethics/DSGVO; evaluate calibration & fairness; add MLOps (model registry, drift monitoring, canary deploy); optionally swap backbone for a German clinical encoder or LLM + retrieval.
5. **Governance** — PHI gate in the API, clear disclaimer, human-in-the-loop for clinical decisions.

## Project layout

```
ai_example/
  train.py          # train + evaluate + save artifacts
  predict.py        # CLI inference
  ui.py             # Tkinter desktop UI
  api.py            # FastAPI + web UI
  start_ui.bat      # launch desktop UI
  start_web.bat     # launch web UI in browser
  static/           # web frontend (HTML/CSS/JS)
  requirements.txt
  src/
    data.py         # synthetic German notes
    tokenizer.py
    model.py        # SpecialtyTextCNN
    train_utils.py
  artifacts/        # trained model + metrics (included in repo)
```
