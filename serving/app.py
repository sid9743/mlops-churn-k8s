"""Churn prediction API with Prometheus metrics and live drift detection."""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from serving.drift import PsiMonitor

PREDICTIONS = Counter("churn_predictions_total", "Predictions served", ["label"])
LATENCY = Histogram("churn_prediction_latency_seconds", "Prediction latency")
DRIFT = Gauge("churn_feature_psi", "PSI of monthly_charges vs training data")

state: dict = {}


def load_model() -> None:
    model_dir = Path(os.getenv("MODEL_DIR", "artifacts"))
    state["model"] = joblib.load(model_dir / "model.joblib")
    state["metrics"] = json.loads((model_dir / "metrics.json").read_text())
    reference = json.loads((model_dir / "reference.json").read_text())
    state["drift"] = PsiMonitor(reference["monthly_charges_bin_edges"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield
    state.clear()


app = FastAPI(title="churn-api", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0)
    total_charges: float = Field(ge=0)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal["upi", "credit-card", "bank-transfer", "cash"]
    support_calls: int = Field(ge=0)
    senior_citizen: int = Field(ge=0, le=1)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    model_run_id: str


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_run_id": state["metrics"]["run_id"]}


@app.get("/model-info")
def model_info():
    return state["metrics"]


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    start = time.perf_counter()
    row = pd.DataFrame([customer.model_dump()])
    proba = float(state["model"].predict_proba(row)[0, 1])
    LATENCY.observe(time.perf_counter() - start)

    will_churn = proba >= 0.5
    PREDICTIONS.labels(label=str(will_churn).lower()).inc()

    monitor: PsiMonitor = state["drift"]
    monitor.add(customer.monthly_charges)
    psi = monitor.psi()
    if psi is not None:
        DRIFT.set(psi)

    return Prediction(
        churn_probability=round(proba, 4),
        will_churn=will_churn,
        model_run_id=state["metrics"]["run_id"],
    )
