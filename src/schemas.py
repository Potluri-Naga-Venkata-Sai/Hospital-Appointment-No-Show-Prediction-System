"""
Pydantic Schemas for Hospital Appointment No-Show Inference API.
Validates operational inputs and standardizes decision-support responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AppointmentInput(BaseModel):
    """Input payload representing a single scheduled hospital appointment."""
    appointment_id: Optional[str] = Field(default="APT-SAMPLE", description="Unique appointment tracking ID")
    patient_id: Optional[str] = Field(default="PAT-SAMPLE", description="Anonymized patient identifier")
    booking_date: str = Field(default="2025-06-01", description="Date the booking was made (YYYY-MM-DD)")
    appointment_date: str = Field(default="2025-06-15", description="Date of appointment (YYYY-MM-DD)")
    lead_time_days: int = Field(default=14, ge=0, description="Days between booking and appointment")
    patient_age: int = Field(default=42, ge=0, le=115, description="Patient age in years")
    distance_miles: float = Field(default=6.5, ge=0.0, description="Approximate travel distance in miles")
    financial_assistance: int = Field(default=0, ge=0, le=1, description="1 if patient receives subsidy/welfare, 0 otherwise")
    clinic_department: str = Field(
        default="General Medicine",
        description="Administrative department (e.g., General Medicine, Orthopedics, Ophthalmology, etc.)",
    )
    booking_channel: str = Field(
        default="Call Center",
        description="Scheduling channel: Online Portal, Call Center, Walk-in Reception, Mobile App",
    )
    appointment_time_slot: str = Field(
        default="Late Morning (10:00-12:00)",
        description="Time block of scheduled appointment",
    )
    appointment_day_of_week: Optional[str] = Field(
        default="Wednesday",
        description="Day of week (Monday-Saturday)",
    )
    sms_reminder_sent: int = Field(default=1, ge=0, le=1, description="1 if reminder was dispatched, 0 otherwise")
    prior_appointments_count: int = Field(default=2, ge=0, description="Total historical appointments prior to this booking")
    prior_no_shows_count: int = Field(default=0, ge=0, description="Total historical no-shows prior to this booking")
    prior_no_show_rate: float = Field(default=0.20, ge=0.0, le=1.0, description="Historical no-show rate")
    is_first_appointment: int = Field(default=0, ge=0, le=1, description="1 if first-time hospital patient, 0 otherwise")

    model_config = {
        "json_schema_extra": {
            "example": {
                "appointment_id": "APT-99201",
                "patient_id": "PAT-4021",
                "booking_date": "2025-06-01",
                "appointment_date": "2025-06-20",
                "lead_time_days": 19,
                "patient_age": 27,
                "distance_miles": 18.5,
                "financial_assistance": 1,
                "clinic_department": "Orthopedics",
                "booking_channel": "Online Portal",
                "appointment_time_slot": "Late Afternoon (14:00-17:00)",
                "appointment_day_of_week": "Friday",
                "sms_reminder_sent": 0,
                "prior_appointments_count": 3,
                "prior_no_shows_count": 2,
                "prior_no_show_rate": 0.50,
                "is_first_appointment": 0,
            }
        }
    }


class FeatureContribution(BaseModel):
    """Explains a single feature's direction and magnitude of influence on no-show risk."""
    feature: str
    shap_value: float
    direction: str


class AppointmentPredictionResponse(BaseModel):
    """Decision-support output for operations staff triage."""
    appointment_id: Optional[str]
    patient_id: Optional[str]
    no_show_probability: float
    is_at_risk: bool
    risk_tier: str
    decision_threshold: float
    recommended_action: str
    top_factors: List[FeatureContribution]
    disclaimer: str = (
        "DECISION-SUPPORT NOTICE: This prediction is for operations planning (reminders, scheduling buffers) only. "
        "It must NEVER be used to withhold care, cancel slots, or alter clinical treatment."
    )


class BatchPredictionRequest(BaseModel):
    """Batch scoring request payload."""
    appointments: List[AppointmentInput]


class BatchPredictionResponse(BaseModel):
    """Batch scoring response summary."""
    total_appointments: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    predictions: List[AppointmentPredictionResponse]


class ModelMetadataResponse(BaseModel):
    """Current model deployment metadata and acceptance test metrics."""
    pipeline_version: str
    champion_model: str
    optimal_threshold: float
    test_metrics: Dict[str, Any]
    feature_count: int
    cost_matrix: Dict[str, float]
