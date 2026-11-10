"""
Unit and integration tests for real model artifact loading, inference probability, and explainability.
Demonstrates acceptance criteria: real predictions from loaded artifacts, non-hardcoded outputs.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    CALIBRATED_MODEL_PATH,
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_PATH,
    EXPLAINER_PATH,
    METADATA_PATH,
)
from src.explainability import explain_instance


def test_saved_artifacts_exist():
    """Verify all required production artifacts are serialized and readable."""
    assert PREPROCESSOR_PATH.exists(), "preprocessor.joblib is missing."
    assert (MODEL_ARTIFACT_PATH.exists() or CALIBRATED_MODEL_PATH.exists()), "Model artifacts missing."
    assert EXPLAINER_PATH.exists(), "explainer.joblib is missing."
    assert METADATA_PATH.exists(), "metadata.json is missing."


def test_real_inference_not_hardcoded():
    """
    Acceptance Test: Submit sample inputs and prove the system returns
    dynamic, genuine probabilities computed by the loaded artifact.
    """
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(CALIBRATED_MODEL_PATH if CALIBRATED_MODEL_PATH.exists() else MODEL_ARTIFACT_PATH)

    # Input A: Very Low Risk Profile (Senior, same-day visit, walk-in reception, SMS confirmed, spotless history)
    input_low_risk = pd.DataFrame([{
        "appointment_id": "APT-TEST-LOW",
        "patient_id": "PAT-LOW",
        "booking_date": "2025-06-10",
        "appointment_date": "2025-06-10",
        "lead_time_days": 0,
        "patient_age": 72,
        "distance_miles": 1.5,
        "financial_assistance": 0,
        "clinic_department": "General Medicine",
        "booking_channel": "Walk-in Reception",
        "appointment_time_slot": "Late Morning (10:00-12:00)",
        "appointment_day_of_week": "Tuesday",
        "sms_reminder_sent": 1,
        "prior_appointments_count": 8,
        "prior_no_shows_count": 0,
        "prior_no_show_rate": 0.05,
        "is_first_appointment": 0,
    }])

    # Input B: High Risk Profile (Young adult, 45-day lead time, Friday late afternoon, far distance, prior missed visits)
    input_high_risk = pd.DataFrame([{
        "appointment_id": "APT-TEST-HIGH",
        "patient_id": "PAT-HIGH",
        "booking_date": "2025-05-01",
        "appointment_date": "2025-06-15",
        "lead_time_days": 45,
        "patient_age": 22,
        "distance_miles": 28.5,
        "financial_assistance": 1,
        "clinic_department": "Orthopedics",
        "booking_channel": "Online Portal",
        "appointment_time_slot": "Late Afternoon (14:00-17:00)",
        "appointment_day_of_week": "Friday",
        "sms_reminder_sent": 0,
        "prior_appointments_count": 4,
        "prior_no_shows_count": 3,
        "prior_no_show_rate": 0.65,
        "is_first_appointment": 0,
    }])

    X_low = preprocessor.transform(input_low_risk)
    X_high = preprocessor.transform(input_high_risk)

    prob_low = float(model.predict_proba(X_low)[0, 1])
    prob_high = float(model.predict_proba(X_high)[0, 1])

    # Assertions
    assert 0.0 <= prob_low <= 1.0, f"Low risk probability out of bounds: {prob_low}"
    assert 0.0 <= prob_high <= 1.0, f"High risk probability out of bounds: {prob_high}"

    # Genuine responsiveness check: high risk input must produce significantly higher probability
    assert prob_high > prob_low, f"Model outputs not sensitive to risk features! High: {prob_high}, Low: {prob_low}"
    assert prob_high - prob_low > 0.15, "Model probability delta between extreme cases is too small."


def test_shap_local_explainability_instance():
    """Acceptance Test: Explain one prediction with local feature attribution."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    explainer_dict = joblib.load(EXPLAINER_PATH)

    input_sample = pd.DataFrame([{
        "appointment_id": "APT-EXPLAIN",
        "patient_id": "PAT-EXPLAIN",
        "booking_date": "2025-06-01",
        "appointment_date": "2025-06-25",
        "lead_time_days": 24,
        "patient_age": 24,
        "distance_miles": 19.0,
        "financial_assistance": 1,
        "clinic_department": "Dental Care",
        "booking_channel": "Online Portal",
        "appointment_time_slot": "Early Morning (08:00-10:00)",
        "appointment_day_of_week": "Monday",
        "sms_reminder_sent": 0,
        "prior_appointments_count": 1,
        "prior_no_shows_count": 1,
        "prior_no_show_rate": 0.50,
        "is_first_appointment": 0,
    }])

    X_trans = preprocessor.transform(input_sample)
    contributions = explain_instance(explainer_dict, X_trans[0], top_k=5)

    assert len(contributions) == 5, "Expected top 5 feature contributions."
    for item in contributions:
        assert "feature" in item
        assert "shap_value" in item
        assert "direction" in item
        assert item["direction"] in ["Increases Risk", "Decreases Risk"]

