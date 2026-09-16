"""End-to-end pipeline: insights -> train -> compare -> roi.

Usage:
    .venv/bin/python scripts/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    "insights_report.py",
    "train_final.py",
    "compare_models.py",
    "roi_report.py",
]


def main() -> int:
    for name in STEPS:
        sys.stdout.write(f"\n=== {name} ===\n")
        sys.stdout.flush()
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name)])
        if result.returncode != 0:
            sys.stdout.write(f"FAILED: {name}\n")
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())