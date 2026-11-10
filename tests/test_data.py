"""
Unit and integration tests for dataset integrity, leakage prevention, and chronological splitting.
"""

import pandas as pd
import pytest
from src.config import RAW_DATA_PATH, TARGET_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.data_generator import generate_synthetic_appointments
from src.data_loader import chronological_split, load_raw_data, validate_and_clean_data


def test_synthetic_data_generator_constraints():
    """Verify generated dataset adheres to all operational and safety requirements."""
    df = generate_synthetic_appointments(n_appointments=1500, n_patients=500, random_seed=42)

    # 1. Volume constraint
    assert len(df) >= 1000, "Dataset must have at least 1,000 records per assignment brief."

    # 2. Strict non-clinical check: Ensure no diagnosis, disease, or treatment columns exist
    forbidden_terms = ["diagnosis", "treatment", "disease", "icd", "prescription", "symptom", "urgency", "vital"]
    for col in df.columns:
        for term in forbidden_terms:
            assert term not in col.lower(), f"Prohibited clinical column detected: {col}"

    # 3. Non-negative lead time
    assert (df["lead_time_days"] >= 0).all(), "Found impossible negative lead times."

    # 4. Realistic age bounds
    assert (df["patient_age"] >= 0).all() and (df["patient_age"] <= 115).all(), "Patient ages out of realistic bounds."

    # 5. Distance sanity
    assert (df["distance_miles"] > 0).all(), "Distance must be strictly positive."

    # 6. Binary target
    unique_targets = set(df[TARGET_COL].unique())
    assert unique_targets.issubset({0, 1}), f"Target must be binary 0 or 1, got {unique_targets}"


def test_chronological_split_leakage_safety():
    """Verify chronological split avoids temporal leakage between train, val, and test."""
    raw_df = load_raw_data()
    cleaned_df = validate_and_clean_data(raw_df)
    train_df, val_df, test_df, meta = chronological_split(cleaned_df, train_ratio=0.70, val_ratio=0.15)

    # Parse dates
    train_max = pd.to_datetime(train_df["appointment_date"]).max()
    val_min = pd.to_datetime(val_df["appointment_date"]).min()
    val_max = pd.to_datetime(val_df["appointment_date"]).max()
    test_min = pd.to_datetime(test_df["appointment_date"]).min()

    # Temporal boundary verification
    assert train_max <= val_min, f"Temporal leakage: Train max date ({train_max}) > Val min date ({val_min})"
    assert val_max <= test_min, f"Temporal leakage: Val max date ({val_max}) > Test min date ({test_min})"

    # Disjoint record verification
    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned_df)

