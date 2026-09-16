"""Print and save the key insights the README cites, computed from raw data.

Usage:
    .venv/bin/python scripts/insights_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from telco_churn.config import OUTPUTS_DIR
from telco_churn.insights import compute_insights, print_insights


def main() -> int:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    insights = compute_insights()
    (OUTPUTS_DIR / "key_insights.json").write_text(
        json.dumps(insights, indent=2, default=str)
    )
    sys.stdout.write("Key insights (from the data):\n")
    sys.stdout.write(print_insights(insights) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())