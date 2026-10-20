from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client

# Load backend/.env. Keep Supabase secrets on the server only.
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tsunamiguard")

MODEL_PATH = BASE_DIR / "tsunami_model.pkl"
FEATURES = ["Year", "dmin", "longitude", "nst", "latitude", "depth", "gap", "sig"]

app = FastAPI(
    title="TsunamiGuard AI API",
    description="Educational API for a trained tsunami-classification model.",
    version="1.0.0",
)

frontend_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,https://tsunamiguard-ai001.mhabibulie02.workers.dev",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in frontend_origins if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model = None
model_load_error: Optional[str] = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
        # This is a warning, not a replacement for validating the original training setup.
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is not None and list(model_features) != FEATURES:
            logger.warning(
                "Model feature_names_in_ is %s; app feature list is %s",
                list(model_features), FEATURES
            )
    except Exception as exc:
        model_load_error = str(exc)
        logger.exception("Could not load the trained model")
else:
    model_load_error = f"Model file not found at {MODEL_PATH}"

supabase: Optional[Client] = None
supabase_error: Optional[str] = None
supabase_url = os.getenv("SUPABASE_URL", "").strip()
supabase_secret = os.getenv("SUPABASE_SECRET_KEY", "").strip()
if supabase_url and supabase_secret:
    try:
        supabase = create_client(supabase_url, supabase_secret)
    except Exception as exc:
        supabase_error = str(exc)
        logger.exception("Could not initialize Supabase")
else:
    supabase_error = "Set SUPABASE_URL and SUPABASE_SECRET_KEY in backend/.env to enable history."


class PredictionInput(BaseModel):
    Year: int = Field(..., ge=1800, le=2200)
    dmin: float = Field(..., ge=0)
    longitude: float = Field(..., ge=-180, le=180)
    nst: int = Field(..., ge=0)
    latitude: float = Field(..., ge=-90, le=90)
    depth: float = Field(..., ge=0)
    gap: float = Field(..., ge=0)
    sig: float = Field(..., ge=0)


def require_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The trained model could not be loaded. Check that backend/tsunami_model.pkl "
                "is present and that scikit-learn/joblib versions match the training environment. "
                f"Details: {model_load_error}"
            ),
        )


def as_json_value(value: Any):
    if hasattr(value, "item"):
        return value.item()
    return value


def save_prediction(input_values: dict, predicted_class: Any):
    """Save a result when Supabase is configured. DB failure must not hide a model prediction."""
    if supabase is None:
        return False, supabase_error or "Supabase is not configured."
    row = {key.lower(): value for key, value in input_values.items()}
    row["predicted_class"] = str(predicted_class)
    try:
        supabase.table("tsunami_predictions").insert(row).execute()
        return True, None
    except Exception as exc:
        logger.exception("Prediction worked, but saving to Supabase failed")
        return False, str(exc)


@app.get("/")
def root():
    return {
        "app": "TsunamiGuard AI",
        "status": "online",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "supabase_configured": supabase is not None,
        "supabase_message": None if supabase is not None else supabase_error,
    }


@app.get("/health")
def health():
    return {"api": "ok", "model_loaded": model is not None, "supabase_configured": supabase is not None}


@app.post("/predict")
def predict(payload: PredictionInput):
    require_model()
    input_values = payload.model_dump()
    # Use a DataFrame so pipelines trained with named columns can receive the same names.
    frame = pd.DataFrame([[input_values[name] for name in FEATURES]], columns=FEATURES)
    try:
        prediction = model.predict(frame)
        predicted_class = as_json_value(prediction[0])
    except Exception as exc:
        logger.exception("Model prediction failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "The model rejected the supplied features. Confirm the original training "
                f"feature names, order, and preprocessing. Details: {exc}"
            ),
        )

    probabilities = None
    classes = getattr(model, "classes_", None)
    if hasattr(model, "predict_proba"):
        try:
            probability_values = model.predict_proba(frame)[0]
            probabilities = [
                {"class": str(as_json_value(cls)), "probability": float(prob)}
                for cls, prob in zip(classes if classes is not None else range(len(probability_values)), probability_values)
            ]
        except Exception:
            logger.exception("Probability scores are unavailable; returning the predicted class only")

    saved, save_error = save_prediction(input_values, predicted_class)
    return {
        "predicted_class": str(predicted_class),
        "probabilities": probabilities,
        "inputs": input_values,
        "saved_to_history": saved,
        "history_warning": save_error,
        "disclaimer": "Educational model output only; not an official tsunami warning.",
    }


@app.get("/history")
def history(limit: int = Query(default=20, ge=1, le=100)):
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail=supabase_error or "Supabase is not configured.",
        )
    try:
        result = (
            supabase.table("tsunami_predictions")
            .select("id,year,dmin,longitude,nst,latitude,depth,gap,sig,predicted_class,created_at")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"items": result.data or []}
    except Exception as exc:
        logger.exception("Could not retrieve prediction history")
        raise HTTPException(status_code=500, detail=f"Could not retrieve history: {exc}")
