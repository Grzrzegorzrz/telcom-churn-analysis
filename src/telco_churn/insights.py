"""Reproduce the key insights cited in the README from the raw data."""
from __future__ import annotations

import pandas as pd

from telco_churn.preprocessing import load_data


def compute_insights(df=None):
    df = df if df is not None else load_data()
    df = df.copy()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    contract = round(df.groupby("Contract")["Churn"].mean() * 100, 1)
    tech = round(df.groupby("TechSupport")["Churn"].mean() * 100, 1)
    pay = round(df.groupby("PaymentMethod")["Churn"].mean() * 100, 1)

    by_churn = df.groupby("Churn")[["MonthlyCharges", "TotalCharges"]].mean()
    monthly = {int(k): round(v["MonthlyCharges"], 2) for k, v in by_churn.iterrows()}
    total = {int(k): round(v["TotalCharges"], 2) for k, v in by_churn.iterrows()}

    churners = df[df["Churn"] == 1]
    early_share = round(100 * (churners["tenure"] <= 12).mean(), 1)

    return {
        "churn_rate": round(df["Churn"].mean() * 100, 1),
        "contract_churn_pct": contract.to_dict(),
        "tech_support_churn_pct": tech.to_dict(),
        "payment_method_churn_pct": pay.to_dict(),
        "avg_monthly_charges_by_churn": monthly,
        "avg_total_charges_by_churn": total,
        "churn_share_first_12_months_pct": early_share,
    }


def print_insights(insights):
    lines = []
    c = insights["contract_churn_pct"]
    lines.append(f"- Contract: month-to-month {c.get('Month-to-month'):.1f}%, one-year {c.get('One year'):.1f}%, two-year {c.get('Two year'):.1f}%")
    t = insights["tech_support_churn_pct"]
    lines.append(f"- Tech support: no {t.get('No'):.1f}% vs yes {t.get('Yes'):.1f}%")
    p = insights["payment_method_churn_pct"]
    lines.append(f"- Payment: electronic check {p.get('Electronic check'):.1f}%, autopay "
                 f"{p.get('Bank transfer (automatic)'):.1f}-{p.get('Credit card (automatic)'):.1f}%")
    m = insights["avg_monthly_charges_by_churn"]
    tot = insights["avg_total_charges_by_churn"]
    lines.append(f"- Spend: churned mean monthly ${m[1]:.2f} vs ${m[0]:.2f}; mean total ${tot[1]:,.0f} vs ${tot[0]:,.0f}")
    lines.append(f"- {insights['churn_share_first_12_months_pct']:.1f}% of churn happens within the first 12 months of tenure")
    return "\n".join(lines)