"""FastAPI service + web UI for clinical note specialty triage."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from predict import load_checkpoint, predict_note

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "artifacts" / "model.pt"
STATIC = ROOT / "static"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"Missing {CHECKPOINT}. Run `python train.py` before starting the API."
        )
    model, vocab, ckpt = load_checkpoint(CHECKPOINT, DEVICE)
    _state.update(model=model, vocab=vocab, ckpt=ckpt)
    yield
    _state.clear()


app = FastAPI(
    title="Clinical Note Triage",
    description=(
        "Interview PoC for hospital AI workflows: route synthetic clinical notes "
        "to a specialty. Not for clinical use; no real PHI."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC), name="static")


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Clinical note text (synthetic/demo)")
    confirm_no_phi: bool = Field(
        ...,
        description="Must be true — confirms the payload contains no real patient data",
    )


class TriageResponse(BaseModel):
    predicted_specialty: str
    confidence: float
    ranking: list[dict]
    disclaimer: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest) -> TriageResponse:
    if not req.confirm_no_phi:
        raise HTTPException(
            status_code=400,
            detail="Set confirm_no_phi=true. This demo must not process real PHI.",
        )
    result = predict_note(
        req.text,
        _state["model"],
        _state["vocab"],
        _state["ckpt"]["max_len"],
        _state["ckpt"]["specialties"],
        DEVICE,
    )
    return TriageResponse(**result)
