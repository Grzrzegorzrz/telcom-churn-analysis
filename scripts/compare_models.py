"""Compare Logistic Regression, Random Forest and XGBoost (all with SMOTE).

Hyper-parameters are predefined in src/telco_churn/config.py
(SMOTE_CONFIG, LR_CONFIG, XGB_CONFIG, and BEST_RF_CONFIG for the RF).

Usage:
    .venv/bin/python scripts/compare_models.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telco_churn.config import (
    BEST_RF_CONFIG,
    CAMPAIGN_TARGET_COUNT,
    LR_CONFIG,
    OUTPUTS_DIR,
    RF_PARAM_KEYS,
    SMOTE_CONFIG,
    XGB_CONFIG,
)
from telco_churn.models import (
    evaluate_topk,
    fit_smote_predict,
    make_lr,
    make_rf,
    make_xgb,
)
from telco_churn.preprocessing import get_train_test


def _smote_params():
    return {
        "sampling_strategy": SMOTE_CONFIG["sampling_strategy"],
        "k_neighbors": SMOTE_CONFIG["k_neighbors"],
    }


def main() -> int:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    dataset = get_train_test()

    models = {
        "Logistic Regression": make_lr(**LR_CONFIG),
        "Random Forest": make_rf(
            random_state=BEST_RF_CONFIG["rf_seed"],
            **{k: BEST_RF_CONFIG[k] for k in RF_PARAM_KEYS},
        ),
        "XGBoost": make_xgb(**XGB_CONFIG),
    }

    results = {}
    for name, clf in models.items():
        _, y_proba = fit_smote_predict(
            clf,
            dataset.X_train,
            dataset.y_train,
            dataset.X_test,
            seed=42,
            **_smote_params(),
        )
        m = evaluate_topk(dataset.y_test, y_proba, k=CAMPAIGN_TARGET_COUNT)
        results[name] = {k: m[k] for k in ("accuracy", "precision", "recall", "f1", "auc")}
        sys.stdout.write(
            f"{name:20s} acc={m['accuracy']:.1%} prec={m['precision']:.1%} "
            f"rec={m['recall']:.1%} f1={m['f1']:.3f} auc={m['auc']:.3f} "
            f"pp={m['pp']} tp={m['tp']}\n"
        )

    results["_settings"] = {
        "smote": SMOTE_CONFIG,
        "logistic_regression": LR_CONFIG,
        "xgboost": XGB_CONFIG,
        "random_forest": "BEST_RF_CONFIG in src/telco_churn/config.py",
    }
    (OUTPUTS_DIR / "model_comparison.json").write_text(
        json.dumps(results, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())