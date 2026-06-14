"""
Preprocessing module for AI4I 2020 Predictive Maintenance dataset.

Loads raw data, drops non-informative columns, encodes categorical 'Type',
and defines feature / target columns.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Constants ──────────────────────────────────────────────────────────────
DATA_PATH = "data/ai4i2020.csv"
RANDOM_STATE = 42

# Feature columns (6 inputs)
FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Target columns (5 binary failure types)
TARGETS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

TARGET_LABELS = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}

# ── Functions ──────────────────────────────────────────────────────────────


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV dataset."""
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Clean and encode the raw DataFrame.

    Steps:
        1. Drop identifier columns (UDI, Product ID, Machine failure).
        2. Label-encode the 'Type' column (L/M/H → 0/1/2).

    Returns
    -------
    df_clean : pd.DataFrame
        Cleaned DataFrame with encoded Type.
    le : LabelEncoder
        Fitted encoder for 'Type' (useful for Streamlit inference).
    """
    df_clean = df.drop(columns=["UDI", "Product ID", "Machine failure"])

    le = LabelEncoder()
    df_clean["Type"] = le.fit_transform(df_clean["Type"])

    return df_clean, le


def get_X_y(df_clean: pd.DataFrame):
    """
    Split cleaned DataFrame into feature matrix X and target dict.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 6)
    y_dict : dict[str, np.ndarray]
        One binary array per failure type.
    """
    X = df_clean[FEATURE_COLS].values
    y_dict = {t: df_clean[t].values for t in TARGETS}
    return X, y_dict
