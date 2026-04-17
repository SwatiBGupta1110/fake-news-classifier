"""
app/api.py
----------
FastAPI REST API for the Fake vs Real News classifier.

REQUIRES: models must be trained first.
    python src/train.py

Run:
    uvicorn app.api:app --reload
    uvicorn app.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /              → health + model status
    GET  /health        → health check
    POST /predict       → classify a single article
    POST /predict/batch → classify up to 50 articles
    GET  /metrics       → last saved Track 1 metrics
"""
import os
import sys
import json
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
sys.path.insert(0, BASE_DIR)

# ── Load models at startup ────────────────────────────────────────────────────
def _load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run 'python src/train.py' first to train and save models."
        )
    return joblib.load(path)


print("[api] Loading models ...")
try:
    LR_PIPELINE  = _load_model("lr_pipeline.joblib")
    SVC_PIPELINE = _load_model("svc_pipeline.joblib")
    MODELS_LOADED = True
    print("[api] Models loaded successfully.")
except FileNotFoundError as e:
    print(f"[api] WARNING: {e}")
    MODELS_LOADED = False
    LR_PIPELINE = SVC_PIPELINE = None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake vs Real News Classifier",
    description=(
        "Production-grade API for classifying news articles as Fake or Real. "
        "Supports a fast classical ML baseline (LinearSVC) and an optional "
        "hybrid LLM mode (GPT-4o-mini) for low-confidence articles."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Article text to classify.")
    title: Optional[str] = Field(None, description="Optional article title.")
    model: Optional[str] = Field(
        "svc",
        description="Which model to use: 'lr' (Logistic Regression) or 'svc' (LinearSVC).",
    )
    use_llm: Optional[bool] = Field(
        False,
        description=(
            "If True, low-confidence predictions are escalated to GPT-4o-mini. "
            "Requires OPENAI_API_KEY in environment."
        ),
    )
    llm_threshold: Optional[float] = Field(
        0.80,
        ge=0.5, le=1.0,
        description="Confidence threshold below which LLM is invoked (when use_llm=True).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Senate passes $1.9 trillion relief bill",
                "text": "WASHINGTON (Reuters) - The Senate passed the landmark $1.9 trillion economic relief bill on Friday ...",
                "model": "svc",
                "use_llm": False,
            }
        }


class PredictResponse(BaseModel):
    label: str
    label_int: int
    confidence: float
    source: str
    baseline_confidence: float
    llm_confidence: Optional[float]
    reasons: List[str]
    needs_human_review: bool
    latency_ms: float
    model_used: str


class BatchPredictRequest(BaseModel):
    articles: List[PredictRequest] = Field(..., max_length=50)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    latency_ms: float


# ── Helper ────────────────────────────────────────────────────────────────────
def _check_models():
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run 'python src/train.py' first.",
        )


def _build_content(title: Optional[str], text: str) -> str:
    prefix = (title.strip() + " ") if title else ""
    return (prefix + text.strip()).lower()


def _classify(req: PredictRequest) -> PredictResponse:
    _check_models()

    pipeline   = LR_PIPELINE if req.model == "lr" else SVC_PIPELINE
    model_name = "Logistic Regression" if req.model == "lr" else "LinearSVC (calibrated)"
    content    = _build_content(req.title, req.text)

    t0 = time.time()

    if req.use_llm:
        from src.llm_inference import hybrid_classify
        result = hybrid_classify(
            article_text=content,
            baseline_pipeline=pipeline,
            threshold=req.llm_threshold,
            use_llm=True,
        )
    else:
        proba  = pipeline.predict_proba([content])[0]
        conf   = float(max(proba))
        pred   = int(proba.argmax())
        result = {
            "final_label"        : "Real" if pred == 1 else "Fake",
            "source"             : "baseline",
            "baseline_confidence": round(conf, 4),
            "llm_confidence"     : None,
            "reasons"            : [],
            "needs_human_review" : False,
            "latency_s"          : 0.0,
            "prompt_tokens"      : 0,
            "completion_tokens"  : 0,
        }

    total_ms = (time.time() - t0) * 1000

    return PredictResponse(
        label               = result["final_label"],
        label_int           = 1 if result["final_label"] == "Real" else 0,
        confidence          = result.get("llm_confidence") or result["baseline_confidence"],
        source              = result["source"],
        baseline_confidence = result["baseline_confidence"],
        llm_confidence      = result.get("llm_confidence"),
        reasons             = result.get("reasons", []),
        needs_human_review  = result.get("needs_human_review", False),
        latency_ms          = round(total_ms, 2),
        model_used          = model_name,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "service"       : "Fake vs Real News Classifier",
        "version"       : "1.0.0",
        "models_loaded" : MODELS_LOADED,
        "docs"          : "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status"        : "ok" if MODELS_LOADED else "degraded",
        "models_loaded" : MODELS_LOADED,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest):
    """
    Classify a single article as Fake or Real.

    - **model**: `svc` (default, fastest) or `lr` (calibrated probabilities)
    - **use_llm**: escalate low-confidence predictions to GPT-4o-mini
    """
    return _classify(req)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    """
    Classify up to 50 articles in a single request.
    """
    t0 = time.time()
    results = [_classify(r) for r in req.articles]
    total_ms = (time.time() - t0) * 1000

    return BatchPredictResponse(
        results   = results,
        total     = len(results),
        latency_ms= round(total_ms, 2),
    )


@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """Return the last saved Track 1 evaluation metrics."""
    path = os.path.join(METRICS_DIR, "track1_metrics.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run 'python src/train.py' first.",
        )
    df = pd.read_csv(path, index_col=0)
    return df.to_dict(orient="index")


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
