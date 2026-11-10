"""
Model Training, Cross-Model Comparison, Hyperparameter Calibration, and Artifact Serialization.
Follows strict chronological partitioning and generates all evaluation deliverables.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")

import json
import logging
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import (
    CALIBRATED_MODEL_PATH,
    MODEL_ARTIFACT_PATH,
    PREPROCESSOR_PATH,
    METADATA_PATH,
    RANDOM_SEED,
    TARGET_COL,
)
from src.data_loader import chronological_split, load_raw_data, validate_and_clean_data
from src.evaluate import (
    calculate_metrics,
    evaluate_subgroup_fairness,
    find_optimal_threshold,
    plot_calibration_curves,
    plot_confusion_matrix_comparison,
    plot_cost_threshold_analysis,
    plot_roc_pr_curves,
)
from src.explainability import build_and_save_explainer, plot_global_shap_summary
from src.features import build_preprocessor, get_feature_names

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate_pipeline():
    """Execute complete end-to-end training and evaluation workflow."""
    logger.info("=== Starting Hospital Appointment No-Show ML Pipeline ===")

    # 1. Ingestion & Validation
    raw_df = load_raw_data()
    cleaned_df = validate_and_clean_data(raw_df)

    # 2. Chronological Split (Train 70%, Val 15%, Test 15%)
    train_df, val_df, test_df, split_meta = chronological_split(cleaned_df)

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    # 3. Fit Preprocessing Pipeline strictly on Train
    logger.info("Fitting preprocessing pipeline on training data...")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(train_df)
    X_val_trans = preprocessor.transform(val_df)
    X_test_trans = preprocessor.transform(test_df)

    feature_names = get_feature_names(preprocessor)
    logger.info("Total transformed features: %d", len(feature_names))

    # 4. Model Candidates Definition
    models = {
        "Dummy Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression (L2)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "LightGBM (Champion)": LGBMClassifier(
            n_estimators=220,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            min_child_samples=30,
            class_weight="balanced",
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED,
            verbose=-1,
        ),
    }

    # 5. Train and Evaluate Models on Validation Set
    val_probs = {}
    test_probs = {}
    experiment_results = {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train_trans, y_train)

        # Predict probabilities (positive class = No-Show)
        p_val = model.predict_proba(X_val_trans)[:, 1]
        p_test = model.predict_proba(X_test_trans)[:, 1]

        val_probs[name] = (y_val, p_val)
        test_probs[name] = (y_test, p_test)

        val_metrics = calculate_metrics(y_val, p_val, threshold=0.50)
        test_metrics = calculate_metrics(y_test, p_test, threshold=0.50)

        experiment_results[name] = {
            "validation_metrics_default": val_metrics,
            "test_metrics_default": test_metrics,
        }
        logger.info("%s Test ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f",
                    name, test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["f1"])

    # 6. Model Calibration (Isotonic calibration on Validation set for champion LightGBM)
    logger.info("Applying Isotonic Probability Calibration to LightGBM...")
    champion_lgbm = models["LightGBM (Champion)"]
    calibrated_lgbm = CalibratedClassifierCV(estimator=champion_lgbm, method="isotonic", cv="prefit")
    calibrated_lgbm.fit(X_val_trans, y_val)

    p_test_calibrated = calibrated_lgbm.predict_proba(X_test_trans)[:, 1]
    p_val_calibrated = calibrated_lgbm.predict_proba(X_val_trans)[:, 1]

    test_probs["Calibrated LightGBM"] = (y_test, p_test_calibrated)
    val_probs["Calibrated LightGBM"] = (y_val, p_val_calibrated)

    # 7. Threshold Optimization on Validation Set
    logger.info("Optimizing decision threshold based on hospital operational cost matrix...")
    optimal_thresh, cost_df = find_optimal_threshold(y_val, p_val_calibrated)

    # Evaluate Calibrated LightGBM on Held-Out Test at optimal threshold
    champion_test_default = calculate_metrics(y_test, p_test_calibrated, threshold=0.50)
    champion_test_optimal = calculate_metrics(y_test, p_test_calibrated, threshold=optimal_thresh)

    logger.info("Champion Test Performance Comparison:")
    logger.info("  Default Threshold (0.50): F1=%.4f, Precision=%.4f, Recall=%.4f, Cost=$%.2f",
                champion_test_default["f1"], champion_test_default["precision"], champion_test_default["recall"],
                champion_test_default["financial_impact"]["total_cost_usd"])
    logger.info("  Optimal Threshold (%.2f): F1=%.4f, Precision=%.4f, Recall=%.4f, Cost=$%.2f",
                optimal_thresh, champion_test_optimal["f1"], champion_test_optimal["precision"], champion_test_optimal["recall"],
                champion_test_optimal["financial_impact"]["total_cost_usd"])
    
    cost_savings = champion_test_default["financial_impact"]["total_cost_usd"] - champion_test_optimal["financial_impact"]["total_cost_usd"]
    logger.info("  Operational Cost Savings from Threshold Optimization: $%.2f (%.2f%% reduction)",
                cost_savings, (cost_savings / champion_test_default["financial_impact"]["total_cost_usd"]) * 100)

    # 8. Visualizations and Artifact Generation
    logger.info("Generating evaluation curves and diagnostics...")
    plot_roc_pr_curves(test_probs)
    plot_calibration_curves(test_probs)
    plot_cost_threshold_analysis(cost_df, optimal_thresh)
    plot_confusion_matrix_comparison(y_test, p_test_calibrated, optimal_thresh)
    subgroup_df = evaluate_subgroup_fairness(test_df, p_test_calibrated, optimal_thresh)

    # 9. Explainability with SHAP
    explainer = build_and_save_explainer(champion_lgbm, X_train_trans, feature_names)
    plot_global_shap_summary(explainer, X_test_trans, feature_names)

    # 10. Serialize Artifacts
    logger.info("Serializing model, preprocessor, and metadata...")
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(champion_lgbm, MODEL_ARTIFACT_PATH)
    joblib.dump(calibrated_lgbm, CALIBRATED_MODEL_PATH)

    metadata = {
        "pipeline_version": "1.0.0",
        "created_at": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "split_summary": split_meta,
        "champion_model": "Calibrated LightGBM (Isotonic)",
        "optimal_threshold": round(optimal_thresh, 4),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "model_comparison_test": {
            k: {
                "roc_auc": round(v["test_metrics_default"]["roc_auc"], 4),
                "pr_auc": round(v["test_metrics_default"]["pr_auc"], 4),
                "f1": round(v["test_metrics_default"]["f1"], 4),
                "precision": round(v["test_metrics_default"]["precision"], 4),
                "recall": round(v["test_metrics_default"]["recall"], 4),
                "brier_score": round(v["test_metrics_default"]["brier_score"], 4),
                "total_cost_usd": v["test_metrics_default"]["financial_impact"]["total_cost_usd"],
            }
            for k, v in experiment_results.items()
        },
        "champion_test_metrics_optimal": champion_test_optimal,
        "champion_test_metrics_default": champion_test_default,
        "operational_cost_savings_usd": round(cost_savings, 2),
        "subgroup_fairness_summary": subgroup_df.to_dict(orient="records"),
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved metadata to %s", METADATA_PATH)
    logger.info("=== Pipeline Training & Evaluation Completed Successfully ===")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_pipeline()
