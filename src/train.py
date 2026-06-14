"""
Training module — builds a unified sklearn Pipeline per target.

Pipeline steps:
    StandardScaler  →  SMOTE (on train only, via imblearn.Pipeline)
    →  SelectKBest (Feature Reduction)  →  RandomForestClassifier

Each of the 5 binary failure types gets its own pipeline saved as a .joblib file.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.evaluate import evaluate_all_metrics
from src.preprocess import (
    load_data,
    preprocess,
    get_X_y,
    FEATURE_COLS,
    TARGETS,
    TARGET_LABELS,
    RANDOM_STATE,
)

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────
MODELS_DIR = "models"
N_FEATURES_KEEP = 6  # SelectKBest: keep all 6 (no reduction) or lower
N_ESTIMATORS = 100


def build_pipeline(n_features: int = N_FEATURES_KEEP) -> ImbPipeline:
    """
    Build an imblearn Pipeline:
        StandardScaler → SMOTE → SelectKBest → RandomForestClassifier

    Using imblearn's Pipeline ensures SMOTE is applied correctly
    during cross-validation / train-test splits (only transforms
    training folds, never test data).
    """
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            (
                "select",
                SelectKBest(score_func=f_classif, k=min(n_features, 6)),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_one_target(
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = N_FEATURES_KEEP,
) -> tuple[ImbPipeline, dict]:
    """
    Train a single binary classifier for one failure type.

    Pipeline per target:
        1. Stratified train-test split (80-20)
        2. Fit pipeline on train (scaler → smote → select → rf)
        3. Predict on test
        4. Evaluate with full metrics

    Returns
    -------
    pipeline : fitted ImbPipeline
    metrics : dict of evaluation results
    """
    # Stratified split — preserves class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Adjust SMOTE k_neighbors if minority class is very small
    n_minor = int(y_train.sum())
    k_nn = min(5, max(1, n_minor - 1))

    pipeline = ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "smote",
                SMOTE(random_state=RANDOM_STATE, k_neighbors=k_nn),
            ),
            (
                "select",
                SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1])),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics = evaluate_all_metrics(y_test, y_pred, y_prob, target)

    # Store extra info
    metrics["n_minority_train"] = n_minor
    metrics["n_minority_test"] = int(y_test.sum())

    return pipeline, metrics


def train_all(
    force_rerun: bool = False,
) -> pd.DataFrame:
    """
    Train all 5 binary classifiers and save each .joblib.

    Parameters
    ----------
    force_rerun : bool
        If True, retrain even if .joblib files exist.

    Returns
    -------
    results_df : pd.DataFrame
        Metrics for all targets and models (for comparison with NB later).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Load & preprocess ──────────────────────────────────────────────
    df_raw = load_data()
    df_clean, _ = preprocess(df_raw)
    X, y_dict = get_X_y(df_clean)

    # Also compute overall X scaled for fallback comparison
    scaler_global = StandardScaler()
    X_scaled_global = scaler_global.fit_transform(X)

    results = []

    print("=" * 70)
    print("TRAINING — 5 BINARY CLASSIFIERS (Pipeline: Scaler→SMOTE→SelectKBest→RF)")
    print("=" * 70)

    for target in TARGETS:
        model_path = os.path.join(MODELS_DIR, f"{target}_rf.pkl")

        # Check if already exists
        if os.path.exists(model_path) and not force_rerun:
            print(f"  ⏭️  {target} — model already exists, loading...")
            pipeline = joblib.load(model_path)
            # Re-run eval for summary
            _, y_dict_local = get_X_y(df_clean)
            y = y_dict_local[target]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
            )
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate_all_metrics(y_test, y_pred, y_prob, target)
            results.append(metrics)
            continue

        y = y_dict[target]
        print(f"\n── {target} ({TARGET_LABELS[target]}) ────────────────────────")

        pipeline, metrics = train_one_target(target, X, y)
        results.append(metrics)

        # Save
        joblib.dump(pipeline, model_path)
        print(f"  ✅ Model saved → {model_path}")

        # Show key metrics
        print(
            f"     Acc={metrics['accuracy']:.4f}  |  "
            f"F1-macro={metrics['f1_macro']:.4f}  |  "
            f"Recall-macro={metrics['recall_macro']:.4f}  |  "
            f"Recall-minority={metrics['recall_minority']:.4f}  |  "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    # Also save a reference LabelEncoder for Type in Streamlit
    _, le = preprocess(df_raw)
    joblib.dump(le, os.path.join(MODELS_DIR, "type_encoder.pkl"))

    print("\n" + "=" * 70)
    print("✅ All models trained and saved!")
    print(f"   Models → {MODELS_DIR}/")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    df_results = train_all(force_rerun=True)
    print("\n📊 Summary:\n")
    print(df_results.to_string(index=False))
