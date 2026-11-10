"""
Feature engineering and preprocessing pipeline module.
Builds scikit-learn transformers that handle imputation, encoding, and scaling
without data leakage.
"""

import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to generate domain-specific non-clinical features:
      - Lead time categorizations
      - Age grouping (Pediatric, Young Adult, Middle Age, Senior)
      - Transit hardship proxy (distance x financial assistance)
      - Risk profile index
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Lead time risk bucketing
        if "lead_time_days" in df.columns:
            df["lead_time_bucket"] = pd.cut(
                df["lead_time_days"],
                bins=[-1, 0, 3, 14, 30, 999],
                labels=["Same-Day", "Short (1-3d)", "Medium (4-14d)", "Long (15-30d)", "Extended (>30d)"],
            ).astype(str)

        # 2. Age group classification
        if "patient_age" in df.columns:
            df["age_group"] = pd.cut(
                df["patient_age"],
                bins=[-1, 17, 35, 64, 150],
                labels=["Pediatric", "Young Adult", "Middle Aged", "Senior"],
            ).astype(str)

        # 3. Transit hardship interaction (Distance > 15 miles AND Financial Assistance == 1)
        if "distance_miles" in df.columns and "financial_assistance" in df.columns:
            df["transit_barrier_flag"] = (
                (df["distance_miles"] > 15.0) & (df["financial_assistance"] == 1)
            ).astype(int)

        # 4. Same day booking indicator
        if "lead_time_days" in df.columns:
            df["is_same_day"] = (df["lead_time_days"] == 0).astype(int)

        return df


def build_preprocessor() -> Pipeline:
    """
    Construct the complete end-to-end preprocessing pipeline.
    Combines domain feature engineering, imputation, one-hot encoding, and scaling.
    """
    # Base categorical features + newly created domain categories
    categorical_cols = CATEGORICAL_FEATURES + ["lead_time_bucket", "age_group"]
    # Base numeric features + engineered binary flags
    numeric_cols = NUMERIC_FEATURES + ["transit_barrier_flag", "is_same_day"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor_core = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    full_pipeline = Pipeline(
        steps=[
            ("domain_features", DomainFeatureEngineer()),
            ("preprocessor", preprocessor_core),
        ]
    )

    return full_pipeline


def get_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """Extract human-readable feature names from fitted ColumnTransformer."""
    col_transformer = fitted_pipeline.named_steps["preprocessor"]
    feature_names = []

    for name, trans, cols in col_transformer.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            onehot = trans.named_steps["onehot"]
            cat_names = onehot.get_feature_names_out(cols)
            feature_names.extend(cat_names.tolist())

    return feature_names

