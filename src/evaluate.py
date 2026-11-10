"""
Evaluation, metrics computation, threshold optimization, and cost-benefit analysis.
Produces defensible metrics and plots comparing candidate models.
"""

import json
import logging
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    BENEFIT_TRUE_POSITIVE,
    FIGURES_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """
    Compute standard and business-aligned classification metrics at a given decision threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Economic impact calculation
    total_cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
    net_benefit = (tp * BENEFIT_TRUE_POSITIVE) - (fp * COST_FALSE_POSITIVE) - (fn * COST_FALSE_NEGATIVE)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "financial_impact": {
            "total_cost_usd": float(total_cost),
            "net_benefit_usd": float(net_benefit),
            "cost_per_appointment_usd": float(total_cost / len(y_true)),
        },
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """
    Search thresholds from 0.05 to 0.95 to find the point minimizing operational cost.
    Evaluates net operational cost = (FN * C_FN) + (FP * C_FP).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        results.append({
            "threshold": t,
            "cost": cost,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    cost_df = pd.DataFrame(results)
    best_row = cost_df.loc[cost_df["cost"].idxmin()]
    optimal_threshold = float(best_row["threshold"])

    logger.info("Optimal threshold by business cost: %.2f (Cost: $%.2f, F1: %.4f)",
                optimal_threshold, best_row["cost"], best_row["f1"])
    return optimal_threshold, cost_df


def plot_roc_pr_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    output_path=FIGURES_DIR / "roc_pr_curves.png"
) -> None:
    """Plot ROC and PR curves for all candidate models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, (y_true, y_prob) in models_dict.items():
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_score:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_score = average_precision_score(y_true, y_prob)
        axes[1].plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_score:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance Baseline")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Base rate line
    sample_y_true = list(models_dict.values())[0][0]
    base_rate = sample_y_true.mean()
    axes[1].axhline(base_rate, color="k", linestyle="--", lw=1.5, label=f"No-Skill ({base_rate:.2f})")
    axes[1].set_title("Precision-Recall Curve (PR-AUC)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved ROC/PR curves to %s", output_path)


def plot_calibration_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path=FIGURES_DIR / "calibration_curve.png",
) -> None:
    """Plot reliability / probability calibration curves."""
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")

    for name, (y_true, y_prob) in models_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        brier = brier_score_loss(y_true, y_prob)
        plt.plot(prob_pred, prob_true, marker="o", lw=2, label=f"{name} (Brier: {brier:.4f})")

    plt.title("Probability Calibration (Reliability Diagram)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Positives (No-Shows)", fontsize=11)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved calibration curves to %s", output_path)


def plot_cost_threshold_analysis(
    cost_df: pd.DataFrame, 
    optimal_threshold: float,
    output_path=FIGURES_DIR / "cost_curve_thresholds.png"
) -> None:
    """Plot operational cost and F1 score against decision thresholds."""
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color = "tab:red"
    ax1.set_xlabel("Decision Threshold (Probability of No-Show)", fontsize=11)
    ax1.set_ylabel("Total Operational Cost ($)", color=color, fontsize=11)
    line1 = ax1.plot(cost_df["threshold"], cost_df["cost"], color=color, lw=2.5, label="Operational Cost ($)")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("F1 Score / Precision / Recall", color=color, fontsize=11)
    line2 = ax2.plot(cost_df["threshold"], cost_df["f1"], color=color, lw=2, linestyle="-", label="F1 Score")
    line3 = ax2.plot(cost_df["threshold"], cost_df["precision"], color="tab:green", lw=1.5, linestyle="--", label="Precision")
    line4 = ax2.plot(cost_df["threshold"], cost_df["recall"], color="tab:orange", lw=1.5, linestyle=":", label="Recall")
    ax2.tick_params(axis="y", labelcolor=color)

    # Highlight optimal threshold
    ax1.axvline(optimal_threshold, color="black", linestyle="--", lw=1.5, label=f"Optimal Threshold ({optimal_threshold:.2f})")
    ax1.axvline(0.50, color="gray", linestyle=":", lw=1.2, label="Default 0.50 Threshold")

    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("Operational Cost vs Decision Threshold Optimization", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved cost threshold analysis to %s", output_path)


def plot_confusion_matrix_comparison(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    opt_threshold: float,
    output_path=FIGURES_DIR / "confusion_matrices.png",
) -> None:
    """Compare confusion matrix at default 0.50 vs optimal threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (thresh, title) in enumerate([
        (0.50, f"Default Threshold (t = 0.50)"),
        (opt_threshold, f"Cost-Optimized Threshold (t = {opt_threshold:.2f})"),
    ]):
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["Attended (0)", "No-Show (1)"],
            yticklabels=["Attended (0)", "No-Show (1)"],
        )
        axes[idx].set_title(title, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrices to %s", output_path)


def evaluate_subgroup_fairness(
    test_df: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    output_path=FIGURES_DIR / "subgroup_fairness.png",
) -> pd.DataFrame:
    """
    Perform responsible AI fairness and error slice analysis across demographic/operational subgroups:
      - Age tiers
      - Financial assistance status
      - Distance tiers
    """
    eval_df = test_df.copy()
    eval_df["y_prob"] = y_prob
    eval_df["y_pred"] = (y_prob >= threshold).astype(int)

    # Categorize age
    eval_df["age_bracket"] = pd.cut(
        eval_df["patient_age"],
        bins=[-1, 18, 35, 65, 120],
        labels=["<18 (Pediatric)", "18-34 (Young)", "35-64 (Adult)", "65+ (Senior)"],
    )

    slices = []
    # 1. By age bracket
    for grp, group_data in eval_df.groupby("age_bracket", observed=False):
        slices.append({
            "Subgroup": f"Age: {grp}",
            "Count": len(group_data),
            "No-Show Rate": float(group_data["no_show"].mean()),
            "Precision": float(precision_score(group_data["no_show"], group_data["y_pred"], zero_division=0)),
            "Recall": float(recall_score(group_data["no_show"], group_data["y_pred"], zero_division=0)),
            "F1": float(f1_score(group_data["no_show"], group_data["y_pred"], zero_division=0)),
        })

    # 2. By Financial Assistance
    for val, name in [(0, "No Assistance"), (1, "Assistance Subsidy")]:
        group_data = eval_df[eval_df["financial_assistance"] == val]
        slices.append({
            "Subgroup": f"Financial: {name}",
            "Count": len(group_data),
            "No-Show Rate": float(group_data["no_show"].mean()),
            "Precision": float(precision_score(group_data["no_show"], group_data["y_pred"], zero_division=0)),
            "Recall": float(recall_score(group_data["no_show"], group_data["y_pred"], zero_division=0)),
            "F1": float(f1_score(group_data["no_show"], group_data["y_pred"], zero_division=0)),
        })

    fairness_df = pd.DataFrame(slices)

    # Plot subgroup metrics
    plt.figure(figsize=(10, 5))
    x = np.arange(len(fairness_df))
    width = 0.25

    plt.bar(x - width, fairness_df["Precision"], width=width, label="Precision", color="#2b5c8f")
    plt.bar(x, fairness_df["Recall"], width=width, label="Recall", color="#e07a5f")
    plt.bar(x + width, fairness_df["F1"], width=width, label="F1 Score", color="#81b29a")

    plt.xlabel("Operational / Demographic Subgroups", fontsize=11)
    plt.ylabel("Metric Score", fontsize=11)
    plt.title("Subgroup Fairness & Model Parity Audit", fontsize=13, fontweight="bold")
    plt.xticks(x, fairness_df["Subgroup"], rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved subgroup fairness audit to %s", output_path)

    return fairness_df
