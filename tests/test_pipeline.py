"""
Unit and integration tests for feature engineering, preprocessor transformation, and model training.
"""

import numpy as np
import pandas as pd
import pytest
from src.features import build_preprocessor, get_feature_names
from src.data_loader import load_raw_data, validate_and_clean_data, chronological_split


def test_preprocessor_pipeline_transformation():
    """Verify that preprocessing converts raw tabular inputs into a complete numerical matrix."""
    raw_df = load_raw_data()
    cleaned_df = validate_and_clean_data(raw_df)
    train_df, _, _, _ = chronological_split(cleaned_df)

    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(train_df)

    # Check transformed array shape
    assert isinstance(X_trans, np.ndarray), "Transformed output must be a numpy ndarray."
    assert X_trans.shape[0] == len(train_df), "Transformed row count must match input row count."
    assert X_trans.shape[1] > 20, "Transformed features must be properly expanded via one-hot encoding."

    # Check for NaN / Infs
    assert not np.isnan(X_trans).any(), "Transformed feature matrix contains unexpected NaN values."
    assert not np.isinf(X_trans).any(), "Transformed feature matrix contains unexpected Inf values."

    # Verify feature names extraction
    feature_names = get_feature_names(preprocessor)
    assert len(feature_names) == X_trans.shape[1], "Extracted feature names length does not match column count."
