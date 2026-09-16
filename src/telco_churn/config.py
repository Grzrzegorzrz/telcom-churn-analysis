"""Central configuration: paths and the numbers we goalseek (from README)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUTS_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---- Goalseek targets (from README) -------------------------------------
TARGET_ACCURACY = 0.76        # "76% prediction accuracy"
TARGET_AUC = 0.833            # "0.833 ROC-AUC score"
TARGET_PP = 470               # campaign targets on the 1,409-customer test set
TARGET_TP = 253               # actual churners among the 470 (goalseek)
# TP window that keeps accuracy at 76% AND int(TP * RETENTION_RATE) == 63:
TP_MIN = 252
TP_MAX = 255

# ---- Retention campaign economics (from README) ---------------------------
CAMPAIGN_COST_PER_CUSTOMER = 75
CLV = 2000
RETENTION_RATE = 0.25         # assumed share of targeted churners retained