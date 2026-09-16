"""Retention-campaign economics (matched to the README numbers)."""
from __future__ import annotations

import numpy as np

from telco_churn.config import (
    CAMPAIGN_COST_PER_CUSTOMER,
    CLV,
    RETENTION_RATE,
)


def top_k_ids(predictions, k):
    """Return the k customer rows with the highest churn probability."""
    return predictions.sort_values("churn_probability", ascending=False).head(k)


def compute_roi(predictions, k=470, retention_rate=RETENTION_RATE):
    """Compute the README ROI table from per-customer predictions.

    predictions: DataFrame with columns actual_churn (0/1) and churn_probability.
    """
    targeted = top_k_ids(predictions, k)
    at_risk = int(np.sum(targeted["actual_churn"].to_numpy() == 1))
    retained = int(at_risk * retention_rate)

    campaign_cost = k * CAMPAIGN_COST_PER_CUSTOMER
    revenue_saved = retained * CLV
    net_benefit = revenue_saved - campaign_cost
    roi = net_benefit / campaign_cost if campaign_cost else 0.0

    return {
        "targeted_customers": int(k),
        "at_risk_in_targets": at_risk,
        "retained_customers": retained,
        "campaign_cost": campaign_cost,
        "revenue_saved": revenue_saved,
        "net_benefit": net_benefit,
        "roi_pct": round(roi * 100, 1),
        "roi_x": round(roi, 3),
        "retention_rate_assumption": retention_rate,
        "cost_per_customer": CAMPAIGN_COST_PER_CUSTOMER,
        "clv": CLV,
    }