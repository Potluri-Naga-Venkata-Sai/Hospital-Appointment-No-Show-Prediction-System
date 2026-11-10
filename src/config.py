"""
Configuration settings for Hospital Appointment No-Show Prediction system.
Maintains operational parameters, business costs, paths, and feature schemas.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "appointments.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "appointments_cleaned.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

MODEL_ARTIFACT_PATH = ARTIFACTS_DIR / "model.joblib"
CALIBRATED_MODEL_PATH = ARTIFACTS_DIR / "calibrated_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
EXPLAINER_PATH = ARTIFACTS_DIR / "explainer.joblib"

# Reproducibility
RANDOM_SEED = 42

# Target Variable
TARGET_COL = "no_show"

# Feature definitions - strictly non-clinical operational features
NUMERIC_FEATURES = [
    "lead_time_days",
    "patient_age",
    "distance_miles",
    "prior_appointments_count",
    "prior_no_shows_count",
    "prior_no_show_rate",
]

CATEGORICAL_FEATURES = [
    "booking_channel",
    "clinic_department",
    "appointment_day_of_week",
    "appointment_time_slot",
    "financial_assistance",
    "sms_reminder_sent",
    "is_first_appointment",
]

DATE_COLS = [
    "booking_date",
    "appointment_date",
]

# Business Operations Cost Matrix (in USD)
# Cost of False Negative (Predict Show, but misses): Doctor idle time, wasted clinic slot
COST_FALSE_NEGATIVE = 120.0

# Cost of False Positive (Predict No-Show, but attends): Redundant phone reminder, scheduling buffer overhead
COST_FALSE_POSITIVE = 15.0

# Benefit of True Positive (Correctly anticipated No-Show): Proactive phone intervention or buffer slot recovery
BENEFIT_TRUE_POSITIVE = 90.0

# Operational Risk Tiers
RISK_THRESHOLD_LOW = 0.20
RISK_THRESHOLD_HIGH = 0.45

# Recommended Interventions per Risk Tier
INTERVENTIONS = {
    "Low": "Automated standard SMS reminder 48 hours prior. Standard slot allocation.",
    "Medium": "Two-way confirmation SMS with 24hr response window. Automated reminder call if unconfirmed.",
    "High": "Priority phone outreach by hospital scheduling coordinator 3 days prior. Eligible for flexible scheduling buffer.",
}
