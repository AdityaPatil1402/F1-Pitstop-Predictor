# ============================================================
#  config.py  —  single source of truth for all paths & constants
#  Import this in every notebook and src/ module
# ============================================================

from pathlib import Path

# ── Root of the project (where config.py lives) ──────────────
ROOT = Path(__file__).parent

# ── Directories ───────────────────────────────────────────────
DATA_DIR    = ROOT / "data"
SRC_DIR     = ROOT / "src"
NB_DIR      = ROOT / "notebooks"
OUT_DIR     = ROOT / "outputs"
FIG_DIR     = OUT_DIR / "figures"
MODEL_DIR   = OUT_DIR / "models"

# ── Files ─────────────────────────────────────────────────────
DATA_PATH   = DATA_DIR / "f1_strategy_dataset_v4.csv"
RESULTS_CSV = OUT_DIR  / "results_summary.csv"

# ── Create output folders if they don't exist ─────────────────
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Columns ───────────────────────────────────────────────────
TARGET      = "PitNextLap"

# Rows to drop — not real race conditions
PRESEASON   = ["Pre-Season Testing", "Pre-Season Test", "Pre-Season Track Session"]

# Year-based train / val / test split
TRAIN_YEARS = [2022, 2023]
VAL_YEAR    = 2024
TEST_YEAR   = 2025

# Tyre compound degradation order (used for ordinal encoding)
COMPOUND_ORDER = {
    "SOFT": 0, "MEDIUM": 1, "HARD": 2,
    "INTERMEDIATE": 1.5, "WET": 2.5, "UNKNOWN": 1
}

# Base numeric features (before engineering)
BASE_FEATURES = [
    "LapNumber", "TyreLife", "Position", "LapTime (s)",
    "LapTime_Delta", "Cumulative_Degradation", "RaceProgress",
    "Normalized_TyreLife", "Position_Change", "Stint",
    "Driver_Freq", "Compound_Ordinal",
    "C_HARD", "C_INTERMEDIATE", "C_MEDIUM",
    "C_SOFT", "C_UNKNOWN", "C_WET"
]

# Engineered features added on top of BASE_FEATURES
ENG_FEATURES = [
    "LapTime_Roll3", "LapTime_Accel", "LapsSinceLastPit",
    "RacePhase", "TyreLife_x_Compound", "Position_Roll3", "DegRate_Roll3"
]

ALL_FEATURES = BASE_FEATURES + ENG_FEATURES

# ── Model params (defaults — overridden by tuning) ────────────
XGB_DEFAULTS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

RF_DEFAULTS = dict(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ── Plot style ────────────────────────────────────────────────
COLORS = {
    "primary"   : "#D85A30",
    "secondary" : "#5DCAA5",
    "blue"      : "#378ADD",
    "gray"      : "#B4B2A9",
    "amber"     : "#FAC775",
}
