"""
Evaluation module — comprehensive metrics for imbalanced classification.

Provides:
    - accuracy, precision, recall, F1 (macro & per-class)
    - ROC-AUC
    - Confusion Matrix values
    - Minority-class-specific metrics (recall, precision for the "1" class)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    target_name: str,
) -> dict:
    """
    Compute all relevant metrics for a binary classification task.

    Parameters
    ----------
    y_true : Ground truth (0 = Normal, 1 = Failure)
    y_pred : Predicted labels
    y_prob : Predicted probability of class 1 (failure)
    target_name : e.g. 'TWF'

    Returns
    -------
    dict with all metrics.
    """
    metrics = {"target": target_name}

    # ── Core metrics ──────────────────────────────────────────────────
    metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)

    # Macro-averaged (treats both classes equally → good for imbalance)
    metrics["f1_macro"] = round(
        f1_score(y_true, y_pred, average="macro", zero_division=0), 4
    )
    metrics["precision_macro"] = round(
        precision_score(y_true, y_pred, average="macro", zero_division=0), 4
    )
    metrics["recall_macro"] = round(
        recall_score(y_true, y_pred, average="macro", zero_division=0), 4
    )

    # ── Per-class (minority = failure = class 1) ──────────────────────
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["confusion_matrix"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    else:
        metrics["confusion_matrix"] = None

    # Minority class (1 = failure) metrics
    metrics["precision_minority"] = round(
        precision_score(y_true, y_pred, labels=[1], zero_division=0), 4
    )
    metrics["recall_minority"] = round(
        recall_score(y_true, y_pred, labels=[1], zero_division=0), 4
    )
    metrics["f1_minority"] = round(
        f1_score(y_true, y_pred, labels=[1], zero_division=0), 4
    )

    # ── ROC-AUC ───────────────────────────────────────────────────────
    try:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
    except Exception:
        metrics["roc_auc"] = 0.0

    # ── Support ───────────────────────────────────────────────────────
    metrics["n_normal"] = int((y_true == 0).sum())
    metrics["n_failure"] = int((y_true == 1).sum())

    return metrics


def print_classification_report(metrics: dict) -> str:
    """Format metrics as a human-readable string (like sklearn's report)."""
    lines = [
        f"  Accuracy       : {metrics['accuracy']:.4f}",
        f"  F1-macro       : {metrics['f1_macro']:.4f}",
        f"  Precision-macro: {metrics['precision_macro']:.4f}",
        f"  Recall-macro   : {metrics['recall_macro']:.4f}",
        f"  ROC-AUC        : {metrics['roc_auc']:.4f}",
        "",
        f"  ── Per-class Minority (Failure) ──",
        f"  Precision (Gagal) : {metrics['precision_minority']:.4f}",
        f"  Recall (Gagal)    : {metrics['recall_minority']:.4f}",
        f"  F1 (Gagal)        : {metrics['f1_minority']:.4f}",
    ]

    if metrics.get("confusion_matrix"):
        cm = metrics["confusion_matrix"]
        lines += [
            "",
            f"  ── Confusion Matrix ──",
            f"  TN={cm['tn']}  FP={cm['fp']}",
            f"  FN={cm['fn']}  TP={cm['tp']}",
        ]

    lines += [
        "",
        f"  Support: Normal={metrics['n_normal']}, Failure={metrics['n_failure']}",
    ]

    return "\n".join(lines)
