"""
API Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_api_root():
    """Verify API root welcomes with links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "/docs" in data["docs_url"]


def test_api_health_check():
    """Verify health endpoint reports model loaded status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "LightGBM" in data["champion_model"]


def test_api_model_metadata():
    """Verify governance metadata endpoint returns correct thresholds and cost matrix."""
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["pipeline_version"] == "1.0.0"
    assert "optimal_threshold" in data
    assert data["cost_matrix"]["cost_false_negative_usd"] == 120.0
    assert data["cost_matrix"]["cost_false_positive_usd"] == 15.0


def test_api_predict_single_appointment():
    """Acceptance Test: Send realistic JSON payload to POST /predict and receive decision-support response."""
    payload = {
        "appointment_id": "APT-API-TEST",
        "patient_id": "PAT-9912",
        "booking_date": "2025-06-01",
        "appointment_date": "2025-06-20",
        "lead_time_days": 19,
        "patient_age": 29,
        "distance_miles": 14.2,
        "financial_assistance": 1,
        "clinic_department": "Orthopedics",
        "booking_channel": "Online Portal",
        "appointment_time_slot": "Late Afternoon (14:00-17:00)",
        "appointment_day_of_week": "Friday",
        "sms_reminder_sent": 0,
        "prior_appointments_count": 2,
        "prior_no_shows_count": 1,
        "prior_no_show_rate": 0.40,
        "is_first_appointment": 0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["appointment_id"] == "APT-API-TEST"
    assert 0.0 <= data["no_show_probability"] <= 1.0
    assert data["risk_tier"] in ["Low", "Medium", "High"]
    assert isinstance(data["is_at_risk"], bool)
    assert len(data["recommended_action"]) > 10
    assert len(data["top_factors"]) > 0
    assert "DECISION-SUPPORT NOTICE" in data["disclaimer"]


def test_api_predict_batch():
    """Test batch prediction endpoint with multiple appointments."""
    payload = {
        "appointments": [
            {
                "appointment_id": "APT-BATCH-1",
                "patient_id": "PAT-1",
                "booking_date": "2025-06-01",
                "appointment_date": "2025-06-02",
                "lead_time_days": 1,
                "patient_age": 70,
                "distance_miles": 2.0,
                "financial_assistance": 0,
                "clinic_department": "General Medicine",
                "booking_channel": "Walk-in Reception",
                "appointment_time_slot": "Late Morning (10:00-12:00)",
                "appointment_day_of_week": "Monday",
                "sms_reminder_sent": 1,
                "prior_appointments_count": 5,
                "prior_no_shows_count": 0,
                "prior_no_show_rate": 0.05,
                "is_first_appointment": 0,
            },
            {
                "appointment_id": "APT-BATCH-2",
                "patient_id": "PAT-2",
                "booking_date": "2025-06-01",
                "appointment_date": "2025-07-01",
                "lead_time_days": 30,
                "patient_age": 21,
                "distance_miles": 25.0,
                "financial_assistance": 1,
                "clinic_department": "Dental Care",
                "booking_channel": "Online Portal",
                "appointment_time_slot": "Early Morning (08:00-10:00)",
                "appointment_day_of_week": "Friday",
                "sms_reminder_sent": 0,
                "prior_appointments_count": 2,
                "prior_no_shows_count": 2,
                "prior_no_show_rate": 0.60,
                "is_first_appointment": 0,
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_appointments"] == 2
    assert len(data["predictions"]) == 2
    assert (data["high_risk_count"] + data["medium_risk_count"] + data["low_risk_count"]) == 2


def test_api_input_validation_negative_lead_time():
    """Verify Pydantic input validation blocks impossible negative lead times."""
    invalid_payload = {
        "lead_time_days": -5,  # Invalid negative lead time
        "patient_age": 30,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422, "API should return 422 Unprocessable Entity for negative lead time."

