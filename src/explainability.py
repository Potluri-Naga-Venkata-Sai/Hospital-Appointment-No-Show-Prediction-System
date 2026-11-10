"""
Model Explainability and Interpretability Module using SHAP.
Generates global feature importance and local instance-level risk attribution.
"""

import logging
from typing import Dict, Any, List
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import EXPLAINER_PATH, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_and_save_explainer(model, X_sample: np.ndarray, feature_names: List[str]) -> shap.TreeExplainer:
    """
    Fit SHAP TreeExplainer on tree-based model (LightGBM/XGBoost) using background sample.
    """
    logger.info("Initializing SHAP TreeExplainer...")
    # Use TreeExplainer for tree models
    explainer = shap.TreeExplainer(model, data=X_sample[:300])
    
    # Save explainer artifact
    EXPLAINER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"explainer": explainer, "feature_names": feature_names}, EXPLAINER_PATH)
    logger.info("Saved SHAP explainer artifact to %s", EXPLAINER_PATH)
    return explainer


def plot_global_shap_summary(
    explainer: shap.TreeExplainer,
    X_test_transformed: np.ndarray,
    feature_names: List[str],
    output_path=FIGURES_DIR / "shap_summary.png",
    max_display: int = 15,
) -> None:
    """Generate and save global SHAP summary plot (beeswarm)."""
    logger.info("Generating global SHAP summary plot...")
    shap_values = explainer.shap_values(X_test_transformed[:1000])

    # Handle binary classification output format differences in SHAP versions
    if isinstance(shap_values, list):
        # shap_values[1] is for positive class (No-Show)
        vals_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        vals_to_plot = shap_values[:, :, 1]
    else:
        vals_to_plot = shap_values

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        vals_to_plot,
        features=X_test_transformed[:1000],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("Global Feature Attribution (SHAP Values for No-Show)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved global SHAP summary plot to %s", output_path)


def explain_instance(
    explainer_dict: Dict[str, Any],
    transformed_row: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Compute local feature contributions for a single transformed appointment input.
    Returns sorted list of top influential factors (positive and negative).
    """
    explainer = explainer_dict["explainer"]
    feature_names = explainer_dict["feature_names"]

    # Ensure 2D shape (1, n_features)
    row_2d = transformed_row.reshape(1, -1)
    shap_vals = explainer.shap_values(row_2d)

    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        vals = shap_vals[0, :, 1]
    else:
        vals = shap_vals[0]

    contributions = []
    for name, val in zip(feature_names, vals):
        contributions.append({
            "feature": name,
            "shap_value": float(val),
            "abs_impact": float(abs(val)),
            "direction": "Increases Risk" if val > 0 else "Decreases Risk",
        })

    # Sort by absolute impact
    contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
    return contributions[:top_k]
