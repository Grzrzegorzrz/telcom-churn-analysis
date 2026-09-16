"""Compute and print the retention-campaign ROI from test-set predictions.

Usage:
    .venv/bin/python scripts/roi_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telco_churn.business import compute_roi
from telco_churn.config import OUTPUTS_DIR


def format_money(x):
    return f"${x:,}"


def main() -> int:
    preds_path = OUTPUTS_DIR / "predictions.csv"
    if not preds_path.exists():
        sys.stdout.write("no outputs/predictions.csv -- run scripts/train_final.py first\n")
        return 1

    preds = pd.read_csv(preds_path)
    roi = compute_roi(preds)

    (OUTPUTS_DIR / "roi.json").write_text(json.dumps(roi, indent=2))

    lines = [
        "Retention Campaign -- Projected Impact",
        "-------------------------------------",
        f"Customers tested on holdout set ........ {len(preds):,}",
        f"Campaign cost per customer ............. {format_money(roi['cost_per_customer'])}",
        f"Customer lifetime value (CLV) .......... {format_money(roi['clv'])}",
        f"High-risk customers targeted ........... {roi['targeted_customers']:,}",
        f"Actual churners among targets .......... {roi['at_risk_in_targets']:,}",
        f"Retained (assumed {roi['retention_rate_assumption']:.0%} of at-risk) ...... {roi['retained_customers']:,}",
        f"Campaign cost .......................... {format_money(roi['campaign_cost'])}",
        f"Revenue saved .......................... {format_money(roi['revenue_saved'])}",
        f"Net benefit ............................ {format_money(roi['net_benefit'])}",
        f"ROI .................................... {roi['roi_pct']}%",
        "",
    ]
    sys.stdout.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())