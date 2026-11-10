"""
FastAPI Inference Service for Hospital Appointment No-Show Prediction.
Provides real-time decision-support scoring, risk stratification, and local explainability.
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    CALIBRATED_MODEL_PATH,
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_PATH,
    EXPLAINER_PATH,
    METADATA_PATH,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    BENEFIT_TRUE_POSITIVE,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    INTERVENTIONS,
)
from src.explainability import explain_instance
from src.schemas import (
    AppointmentInput,
    AppointmentPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureContribution,
    ModelMetadataResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# State container for loaded artifacts
ml_assets: Dict[str, Any] = {}


def load_assets():
    """Load model artifacts and preprocessor into ml_assets dictionary."""
    if ml_assets.get("model") is not None and ml_assets.get("preprocessor") is not None:
        return

    logger.info("Loading ML artifacts into memory...")
    if not PREPROCESSOR_PATH.exists() or not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Please run `python src/train.py` first.")

    ml_assets["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    
    # Load calibrated model if present, fallback to raw model
    if CALIBRATED_MODEL_PATH.exists():
        ml_assets["model"] = joblib.load(CALIBRATED_MODEL_PATH)
        logger.info("Loaded Calibrated Model.")
    else:
        ml_assets["model"] = joblib.load(MODEL_ARTIFACT_PATH)
        logger.info("Loaded Base Model.")

    if EXPLAINER_PATH.exists():
        ml_assets["explainer_dict"] = joblib.load(EXPLAINER_PATH)
        logger.info("Loaded SHAP Explainer.")
    else:
        ml_assets["explainer_dict"] = None

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            ml_assets["metadata"] = json.load(f)
    else:
        ml_assets["metadata"] = {
            "pipeline_version": "1.0.0",
            "champion_model": "Calibrated LightGBM",
            "optimal_threshold": 0.12,
        }

    logger.info("All ML assets successfully initialized.")


# Automatically load upon module import if files exist
if PREPROCESSOR_PATH.exists() and (MODEL_ARTIFACT_PATH.exists() or CALIBRATED_MODEL_PATH.exists()):
    try:
        load_assets()
    except Exception as e:
        logger.warning("Auto-load of assets on import skipped: %s", str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    load_assets()
    yield
    # Keep assets or cleanup if needed


app = FastAPI(
    title="Hospital Appointment No-Show Prediction API",
    description=(
        "Production ML Decision-Support Service for Outpatient Appointment Operations. "
        "Provides calibrated risk scoring, operational reminder triage, and SHAP explainability. "
        "Strictly decision support; no clinical/diagnostic inputs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _determine_risk_tier(prob: float) -> str:
    if prob < RISK_THRESHOLD_LOW:
        return "Low"
    elif prob < RISK_THRESHOLD_HIGH:
        return "Medium"
    else:
        return "High"


def _score_single_record(item_dict: Dict[str, Any], threshold: float) -> AppointmentPredictionResponse:
    """Internal helper to score a single appointment dictionary."""
    df_single = pd.DataFrame([item_dict])
    preprocessor = ml_assets["preprocessor"]
    model = ml_assets["model"]

    # Transform
    X_trans = preprocessor.transform(df_single)

    # Predict calibrated probability
    prob = float(model.predict_proba(X_trans)[0, 1])
    is_at_risk = bool(prob >= threshold)
    risk_tier = _determine_risk_tier(prob)
    recommended_action = INTERVENTIONS[risk_tier]

    # Explainability
    factors: List[FeatureContribution] = []
    if ml_assets.get("explainer_dict") is not None:
        try:
            raw_contribs = explain_instance(ml_assets["explainer_dict"], X_trans[0], top_k=5)
            factors = [
                FeatureContribution(
                    feature=c["feature"],
                    shap_value=c["shap_value"],
                    direction=c["direction"],
                )
                for c in raw_contribs
            ]
        except Exception as ex:
            logger.warning("SHAP explanation error: %s", str(ex))

    return AppointmentPredictionResponse(
        appointment_id=item_dict.get("appointment_id"),
        patient_id=item_dict.get("patient_id"),
        no_show_probability=round(prob, 4),
        is_at_risk=is_at_risk,
        risk_tier=risk_tier,
        decision_threshold=round(threshold, 4),
        recommended_action=recommended_action,
        top_factors=factors,
    )


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Hospital Appointment No-Show Prediction Service",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["General"])
def health_check():
    """Verify service readiness and asset integrity."""
    is_ready = bool(ml_assets.get("model") is not None and ml_assets.get("preprocessor") is not None)
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "champion_model": ml_assets.get("metadata", {}).get("champion_model", "Unknown"),
        "version": ml_assets.get("metadata", {}).get("pipeline_version", "1.0.0"),
    }


@app.get("/model/metadata", response_model=ModelMetadataResponse, tags=["Model Governance"])
def get_model_metadata():
    """Retrieve model governance details, acceptance test metrics, and cost matrix."""
    meta = ml_assets.get("metadata", {})
    return ModelMetadataResponse(
        pipeline_version=meta.get("pipeline_version", "1.0.0"),
        champion_model=meta.get("champion_model", "Calibrated LightGBM"),
        optimal_threshold=meta.get("optimal_threshold", 0.12),
        test_metrics=meta.get("champion_test_metrics_optimal", {}),
        feature_count=meta.get("feature_count", 45),
        cost_matrix={
            "cost_false_negative_usd": COST_FALSE_NEGATIVE,
            "cost_false_positive_usd": COST_FALSE_POSITIVE,
            "benefit_true_positive_usd": BENEFIT_TRUE_POSITIVE,
        },
    )


@app.post("/predict", response_model=AppointmentPredictionResponse, tags=["Inference"])
def predict_single_appointment(payload: AppointmentInput):
    """
    Score a single hospital appointment.
    Loads the trained model, applies data transformations, and returns
    calibrated probability, risk tier, action recommendation, and SHAP drivers.
    """
    threshold = ml_assets.get("metadata", {}).get("optimal_threshold", 0.12)
    try:
        return _score_single_record(payload.model_dump(), threshold=threshold)
    except Exception as e:
        logger.error("Inference failure: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_appointments(payload: BatchPredictionRequest):
    """
    Score multiple upcoming appointments for daily operations triage.
    """
    threshold = ml_assets.get("metadata", {}).get("optimal_threshold", 0.12)
    predictions = []
    high_count, med_count, low_count = 0, 0, 0

    try:
        for appt in payload.appointments:
            res = _score_single_record(appt.model_dump(), threshold=threshold)
            predictions.append(res)
            if res.risk_tier == "High":
                high_count += 1
            elif res.risk_tier == "Medium":
                med_count += 1
            else:
                low_count += 1

        return BatchPredictionResponse(
            total_appointments=len(predictions),
            high_risk_count=high_count,
            medium_risk_count=med_count,
            low_risk_count=low_count,
            predictions=predictions,
        )
    except Exception as e:
        logger.error("Batch inference failure: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")
