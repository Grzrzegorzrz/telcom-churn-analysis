"""Fit the final RF + SMOTE model from the predefined config; write artifacts.

Usage:
    .venv/bin/python scripts/train_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telco_churn.config import (
    BEST_RF_CONFIG,
    CAMPAIGN_TARGET_COUNT,
    OUTPUTS_DIR,
    RF_PARAM_KEYS,
)
from telco_churn.models import evaluate_topk, fit_smote_predict, make_rf
from telco_churn.preprocessing import get_train_test


def main() -> int:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    dataset = get_train_test()

    rf = make_rf(random_state=BEST_RF_CONFIG["rf_seed"], **{
        k: BEST_RF_CONFIG[k] for k in RF_PARAM_KEYS
    })
    _, y_proba = fit_smote_predict(
        rf,
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        sampling_strategy=BEST_RF_CONFIG["sampling_strategy"],
        k_neighbors=BEST_RF_CONFIG["k_neighbors"],
        seed=42,
    )
    metrics = evaluate_topk(dataset.y_test, y_proba, k=CAMPAIGN_TARGET_COUNT)
    metrics["params"] = dict(BEST_RF_CONFIG)

    order = pd.Series(y_proba).sort_values(ascending=False).index
    preds = pd.DataFrame({
        "customerID": dataset.test_ids[order],
        "actual_churn": dataset.y_test[order],
        "churn_probability": y_proba[order],
    })
    preds["predicted_churn"] = 0
    preds.iloc[:CAMPAIGN_TARGET_COUNT, preds.columns.get_loc("predicted_churn")] = 1
    preds["annual_value"] = preds["customerID"].map(
        {cid: row for cid, row in _annual_value_map().items()}
    ).fillna(0).astype(int)

    joblib.dump(rf, OUTPUTS_DIR / "rf_smote_model.joblib")
    joblib.dump(dataset.scaler, OUTPUTS_DIR / "scaler.joblib")
    (OUTPUTS_DIR / "feature_names.json").write_text(
        json.dumps(list(dataset.X_train.columns), indent=2)
    )
    (OUTPUTS_DIR / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    preds.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    sys.stdout.write(
        "FINAL RF+SMOTE: acc={:.1f}% prec={:.1f}% rec={:.1f}% f1={:.3f} "
        "auc={:.3f} pp={} tp={} threshold={:.4f}\n".format(
            metrics["accuracy"] * 100, metrics["precision"] * 100,
            metrics["recall"] * 100, metrics["f1"], metrics["auc"],
            metrics["pp"], metrics["tp"], metrics["threshold"],
        )
    )
    return 0


def _annual_value_map():
    from telco_churn.preprocessing import load_data

    df = load_data()
    return {row.customerID: int(row.MonthlyCharges * 12) for row in df.itertuples()}


if __name__ == "__main__":
    raise SystemExit(main())