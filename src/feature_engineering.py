# ============================================================
#  src/feature_engineering.py
#  Adds domain-knowledge features on top of the cleaned df.
#  Usage: from src.feature_engineering import engineer_features
# ============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import config as cfg


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 7 new features derived from domain knowledge of F1 strategy.
    Must be called AFTER preprocess.encode().
    Returns a new DataFrame with extra columns.
    """
    df = df.copy()

    # Sort so rolling windows are computed in correct lap order
    df = df.sort_values(["Year", "Race", "Driver", "LapNumber"])

    grp = df.groupby(["Year", "Race", "Driver"])

    # ── 1. Rolling 3-lap average lap time ─────────────────────
    # Smooths out traffic/safety-car noise
    df["LapTime_Roll3"] = (
        grp["LapTime (s)"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # ── 2. Lap time acceleration (is it getting slower faster?) ─
    df["LapTime_Accel"] = (
        grp["LapTime_Delta"]
        .transform(lambda x: x.diff().fillna(0))
    )

    # ── 3. Laps since last pit stop ───────────────────────────
    def laps_since_pit(group):
        counter, result = 0, []
        for pit in group["PitStop"]:
            result.append(counter)
            counter = 0 if pit == 1 else counter + 1
        return pd.Series(result, index=group.index)

    df["LapsSinceLastPit"] = (
        df.groupby(["Year", "Race", "Driver"], group_keys=False)
          .apply(laps_since_pit)
    )

    # ── 4. Race phase  (0=early, 1=mid, 2=late) ───────────────
    df["RacePhase"] = pd.cut(
        df["RaceProgress"],
        bins=[0, 0.33, 0.66, 1.01],
        labels=[0, 1, 2]
    ).astype(float)

    # ── 5. TyreLife × Compound_Ordinal ────────────────────────
    # Soft tyres degrade harder as they age
    df["TyreLife_x_Compound"] = df["TyreLife"] * df["Compound_Ordinal"]

    # ── 6. Rolling 3-lap position average ─────────────────────
    df["Position_Roll3"] = (
        grp["Position"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    # ── 7. Degradation rate (rolling slope of cumulative deg) ──
    df["DegRate_Roll3"] = (
        grp["Cumulative_Degradation"]
        .transform(lambda x: x.diff().rolling(3, min_periods=1).mean())
        .fillna(0)
    )

    new_cols = cfg.ENG_FEATURES
    print(f"[engineer_features] Added {len(new_cols)} features: {new_cols}")
    return df
