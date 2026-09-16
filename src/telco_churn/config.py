"""Central configuration: paths, business assumptions, and model configs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUTS_DIR = ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---- Retention campaign economics ------------------------------------------
CAMPAIGN_TARGET_COUNT = 470   # top-N highest-risk customers targeted on the holdout set
CAMPAIGN_COST_PER_CUSTOMER = 75
CLV = 2000
RETENTION_RATE = 0.25         # assumed share of targeted churners retained

# ---- Predefined model configurations ---------------------------------------
# Final Random Forest configuration, kept fixed for reproducibility.
BEST_RF_CONFIG = {
    "n_estimators": 150,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "log2",
    "class_weight": "balanced",
    "max_samples": None,
    "sampling_strategy": 1.0,   # SMOTE for the final RF model
    "k_neighbors": 5,
    "rf_seed": 4,
}
RF_PARAM_KEYS = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "max_samples",
)

# Settings for the LR / XGB comparison models (and their SMOTE).
SMOTE_CONFIG = {"sampling_strategy": 0.75, "k_neighbors": 5}
LR_CONFIG = {"C": 1.0, "solver": "lbfgs", "max_iter": 2000}
XGB_CONFIG = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1}