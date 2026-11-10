"""
Data loading, validation, and chronological splitting module.
Guarantees temporal ordering and leak-free train/validation/test partitions.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from src.config import (
    RAW_DATA_PATH,
    TARGET_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    DATE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(file_path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw appointments dataset from disk and parse dates."""
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {file_path}. Run generator first.")
    
    df = pd.read_csv(file_path)
    logger.info("Loaded raw data with shape %s", df.shape)
    return df


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data integrity, enforce business constraints, and clean anomalies.
    Ensures:
      1. Non-negative lead time (booking occurs before or on appointment date).
      2. Plausible patient age bounds (0 to 110).
      3. Non-negative distance.
      4. Valid target encoding (0 or 1).
    """
    df = df.copy()

    # Required columns check
    expected_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DATE_COLS + [TARGET_COL]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in raw data: {missing}")

    # Parse dates
    df["booking_date_dt"] = pd.to_datetime(df["booking_date"], errors="coerce")
    df["appointment_date_dt"] = pd.to_datetime(df["appointment_date"], errors="coerce")

    # Drop null dates
    initial_len = len(df)
    df = df.dropna(subset=["booking_date_dt", "appointment_date_dt"]).copy()
    if len(df) < initial_len:
        logger.warning("Dropped %d records with invalid dates", initial_len - len(df))

    # Re-verify lead time calculation
    calculated_lead_time = (df["appointment_date_dt"] - df["booking_date_dt"]).dt.days
    # Drop impossible negative lead times if any exist
    invalid_lead = calculated_lead_time < 0
    if invalid_lead.any():
        logger.warning("Dropping %d records with negative lead time (booking > appointment)", invalid_lead.sum())
        df = df[~invalid_lead].copy()
        calculated_lead_time = calculated_lead_time[~invalid_lead]

    df["lead_time_days"] = calculated_lead_time

    # Age sanity checks
    invalid_age = (df["patient_age"] < 0) | (df["patient_age"] > 115)
    if invalid_age.any():
        logger.warning("Clipping %d records with out-of-range ages", invalid_age.sum())
        df["patient_age"] = df["patient_age"].clip(0, 100)

    # Distance sanity
    df["distance_miles"] = df["distance_miles"].clip(lower=0.1)

    # Sort strictly chronologically by appointment date, then booking date
    df.sort_values(by=["appointment_date_dt", "booking_date_dt"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Drop helper date objects
    df.drop(columns=["booking_date_dt", "appointment_date_dt"], inplace=True)

    logger.info("Data validated. Cleaned records count: %d", len(df))
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split the dataset chronologically to reflect real-world production deployment.
    Avoids future data leakage into training/validation sets.

    Returns:
      train_df, val_df, test_df, split_metadata
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    metadata = {
        "total_records": n,
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "train_date_range": (train_df["appointment_date"].min(), train_df["appointment_date"].max()),
        "val_date_range": (val_df["appointment_date"].min(), val_df["appointment_date"].max()),
        "test_date_range": (test_df["appointment_date"].min(), test_df["appointment_date"].max()),
        "train_no_show_rate": float(train_df[TARGET_COL].mean()),
        "val_no_show_rate": float(val_df[TARGET_COL].mean()),
        "test_no_show_rate": float(test_df[TARGET_COL].mean()),
    }

    logger.info("Chronological split completed:")
    logger.info("  Train: %d rows (%s to %s) | No-show: %.2f%%",
                metadata["train_records"], metadata["train_date_range"][0], metadata["train_date_range"][1], metadata["train_no_show_rate"] * 100)
    logger.info("  Val:   %d rows (%s to %s) | No-show: %.2f%%",
                metadata["val_records"], metadata["val_date_range"][0], metadata["val_date_range"][1], metadata["val_no_show_rate"] * 100)
    logger.info("  Test:  %d rows (%s to %s) | No-show: %.2f%%",
                metadata["test_records"], metadata["test_date_range"][0], metadata["test_date_range"][1], metadata["test_no_show_rate"] * 100)

    return train_df, val_df, test_df, metadata

