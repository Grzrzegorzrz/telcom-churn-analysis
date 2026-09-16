"""Data loading, cleaning, feature engineering and train/test split."""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from telco_churn.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

# Columns that get binary-encoded (Yes->1 / No->0, Male->1 / Female->0)
BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]
ONE_HOT_COLS = ["InternetService", "Contract", "PaymentMethod"]
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TENURE_BINS = [0, 12, 24, 36, 48, 60, 72]
TENURE_LABELS = ["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]

Dataset = namedtuple(
    "Dataset",
    ["X_train", "X_test", "y_train", "y_test", "scaler", "test_ids"],
)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 6 engineered features described in the README."""
    df = df.copy()
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True
    )
    df["AverageMonthlySpend"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1)
    df["HasTechSupport"] = (df["TechSupport"] == "Yes").astype(int)
    df["HasInternet"] = (df["InternetService"] != "No").astype(int)
    df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y, feature_names) with all-numeric, encoded features."""
    df = clean(df)
    if df["TotalCharges"].isna().any():
        missing = df["TotalCharges"].isna()
        df.loc[missing, "TotalCharges"] = (
            df.loc[missing, "MonthlyCharges"] * df.loc[missing, "tenure"]
        )
    df = engineer_features(df)
    y = df["Churn"].to_numpy()
    df = df.drop(columns=["customerID", "Churn"]).copy()

    binaries = (df[BINARY_COLS] == "Yes").astype(int)
    binaries["gender"] = (df["gender"] == "Male").astype(int)
    one_hot = pd.get_dummies(
        df[ONE_HOT_COLS], prefix=ONE_HOT_COLS, drop_first=True, dtype=int
    )
    tenure_oh = pd.get_dummies(df["TenureGroup"], prefix="TenureGroup", drop_first=True, dtype=int)
    engineered = df[
        [
            "AverageMonthlySpend",
            "ServiceCount",
            "HasTechSupport",
            "HasInternet",
            "IsMonthToMonth",
        ]
    ]
    numerics = df[NUMERIC_COLS]

    X = pd.concat([numerics, one_hot, tenure_oh, binaries, engineered], axis=1)
    X = X.astype(float)
    return X, y, list(X.columns)


def get_train_test(
    data_path=DATA_PATH,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
) -> Dataset:
    df = load_data(data_path)
    X, y, _ = build_feature_matrix(df)
    customer_ids = df["customerID"].to_numpy()

    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X,
        y,
        customer_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = pd.DataFrame(X_train_s, columns=X_train.columns)
    X_test_s = pd.DataFrame(X_test_s, columns=X_test.columns)

    return Dataset(
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        test_ids=idx_te,
    )