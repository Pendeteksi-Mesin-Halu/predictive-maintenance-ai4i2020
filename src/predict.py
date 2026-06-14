"""
Prediction / Inference module.

Loads a trained pipeline from .joblib and predicts failure probabilities
for a single sample (e.g. from Streamlit form input).
"""

import os

import joblib
import numpy as np
import pandas as pd

from src.preprocess import FEATURE_COLS, TARGETS

MODELS_DIR = "models"


def load_model(target: str):
    """Load a trained pipeline for one failure type."""
    path = os.path.join(MODELS_DIR, f"{target}_rf.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_type_encoder():
    """Load the fitted LabelEncoder for 'Type' column."""
    path = os.path.join(MODELS_DIR, "type_encoder.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Type encoder not found: {path}. Run training first."
        )
    return joblib.load(path)


def predict_single(
    type_val: int | str,
    air_temp: float,
    proc_temp: float,
    rot_speed: float,
    torque: float,
    tool_wear: float,
) -> dict:
    """
    Predict failure probabilities for a single machine sample.

    Parameters
    ----------
    type_val : int (0/1/2) or str (L/M/H) — product type
    air_temp : Air temperature [K]
    proc_temp : Process temperature [K]
    rot_speed : Rotational speed [rpm]
    torque : Torque [Nm]
    tool_wear : Tool wear [min]

    Returns
    -------
    dict of {target: {prediction: int, probability: float}} for all 5 targets.
    """
    # Encode Type if given as string
    if isinstance(type_val, str):
        le = load_type_encoder()
        type_val = int(le.transform([type_val])[0])

    features = np.array([[type_val, air_temp, proc_temp, rot_speed, torque, tool_wear]])

    results = {}
    for target in TARGETS:
        pipe = load_model(target)
        pred = int(pipe.predict(features)[0])
        prob = float(pipe.predict_proba(features)[0, 1])
        results[target] = {
            "prediction": pred,
            "probability": round(prob, 4),
        }

    return results


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict failure probabilities for a batch of samples.

    Parameters
    ----------
    df : DataFrame with columns matching FEATURE_COLS

    Returns
    -------
    DataFrame with original features + prediction columns.
    """
    result = df.copy()
    for target in TARGETS:
        pipe = load_model(target)
        preds = pipe.predict(result[FEATURE_COLS].values)
        probs = pipe.predict_proba(result[FEATURE_COLS].values)[:, 1]
        result[f"{target}_pred"] = preds
        result[f"{target}_prob"] = np.round(probs, 4)
    return result
