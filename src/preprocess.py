# ============================================================
#  src/preprocess.py
#  Handles: loading, cleaning, encoding, train/val/test split
#  Usage:  from src.preprocess import load_and_prepare
# ============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config as cfg


# ────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    """Load the raw CSV and return as DataFrame."""
    df = pd.read_csv(cfg.DATA_PATH)
    print(f"[load_raw] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


# ────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop pre-season rows.
    2. Fill missing Compound.
    3. Report null counts.
    """
    before = len(df)
    df = df[~df["Race"].isin(cfg.PRESEASON)].copy()
    print(f"[clean] Dropped {before - len(df):,} pre-season rows → {len(df):,} remain")

    null_before = df.isnull().sum().sum()
    df["Compound"] = df["Compound"].fillna("UNKNOWN")
    print(f"[clean] Filled {null_before} nulls in Compound with 'UNKNOWN'")

    return df


# ────────────────────────────────────────────────────────────
def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. One-hot encode Compound   → C_SOFT, C_MEDIUM, …
    2. Frequency encode Driver   → Driver_Freq
    3. Ordinal encode Compound   → Compound_Ordinal (degradation order)
    """
    df = df.copy()

    # One-hot — prefix C_ so columns are predictable
    ohe = pd.get_dummies(df["Compound"], prefix="C", dtype=int)
    # Ensure all expected dummies exist even if a compound is absent in a split
    for col in ["C_HARD", "C_INTERMEDIATE", "C_MEDIUM", "C_SOFT", "C_UNKNOWN", "C_WET"]:
        if col not in ohe.columns:
            ohe[col] = 0
    df = pd.concat([df, ohe], axis=1)

    # Frequency encoding for Driver
    freq = df["Driver"].value_counts(normalize=True)
    df["Driver_Freq"] = df["Driver"].map(freq)

    # Ordinal encoding for Compound
    df["Compound_Ordinal"] = df["Compound"].map(cfg.COMPOUND_ORDER)

    print(f"[encode] Added OHE columns: {[c for c in ohe.columns]}")
    print(f"[encode] Added Driver_Freq, Compound_Ordinal")
    return df


# ────────────────────────────────────────────────────────────
def split(df: pd.DataFrame):
    """
    Time-aware split — NEVER random shuffle (laps are sequential).
      Train : years 2022–2023
      Val   : year  2024
      Test  : year  2025  ← touch only at very end
    Returns (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    train = df[df["Year"].isin(cfg.TRAIN_YEARS)]
    val   = df[df["Year"] == cfg.VAL_YEAR]
    test  = df[df["Year"] == cfg.TEST_YEAR]

    def _xy(subset, feature_list):
        X = subset[feature_list].fillna(0)
        y = subset[cfg.TARGET]
        return X, y

    X_tr, y_tr = _xy(train, cfg.BASE_FEATURES)
    X_va, y_va = _xy(val,   cfg.BASE_FEATURES)
    X_te, y_te = _xy(test,  cfg.BASE_FEATURES)

    print(f"[split] Train  : {len(X_tr):>7,} rows  | pit rate = {y_tr.mean():.3f}")
    print(f"[split] Val    : {len(X_va):>7,} rows  | pit rate = {y_va.mean():.3f}")
    print(f"[split] Test   : {len(X_te):>7,} rows  | pit rate = {y_te.mean():.3f}")

    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ────────────────────────────────────────────────────────────
def split_engineered(df_eng: pd.DataFrame):
    """Same as split() but uses ALL_FEATURES (base + engineered)."""
    train = df_eng[df_eng["Year"].isin(cfg.TRAIN_YEARS)]
    val   = df_eng[df_eng["Year"] == cfg.VAL_YEAR]
    test  = df_eng[df_eng["Year"] == cfg.TEST_YEAR]

    def _xy(subset):
        X = subset[cfg.ALL_FEATURES].fillna(0)
        y = subset[cfg.TARGET]
        return X, y

    X_tr, y_tr = _xy(train)
    X_va, y_va = _xy(val)
    X_te, y_te = _xy(test)

    print(f"[split_eng] Train  : {len(X_tr):>7,} | Val: {len(X_va):>6,} | Test: {len(X_te):>6,}")
    return X_tr, X_va, X_te, y_tr, y_va, y_te


# ────────────────────────────────────────────────────────────
def scale(X_train, X_val, X_test):
    """
    Fit StandardScaler on training set only.
    Returns (X_train_s, X_val_s, X_test_s, scaler)
    — used only for Logistic Regression; trees don't need scaling.
    """
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_train)
    X_va_s = sc.transform(X_val)
    X_te_s = sc.transform(X_test)
    print("[scale] StandardScaler fitted on train, applied to val + test")
    return X_tr_s, X_va_s, X_te_s, sc


# ────────────────────────────────────────────────────────────
def class_weight_ratio(y_train) -> float:
    """Return neg/pos ratio for scale_pos_weight in XGBoost."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    ratio = neg / pos
    print(f"[class_weight] neg={neg:,} pos={pos:,} → scale_pos_weight={ratio:.3f}")
    return ratio


# ────────────────────────────────────────────────────────────
def load_and_prepare():
    """
    Full pipeline: load → clean → encode → split.
    Returns dict with all splits + the cleaned df for EDA.
    """
    df_raw  = load_raw()
    df_clean = clean(df_raw)
    df_enc   = encode(df_clean)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(df_enc)
    spw = class_weight_ratio(y_tr)

    return {
        "df"      : df_enc,       # full cleaned+encoded df (use for EDA)
        "X_train" : X_tr,
        "X_val"   : X_va,
        "X_test"  : X_te,
        "y_train" : y_tr,
        "y_val"   : y_va,
        "y_test"  : y_te,
        "spw"     : spw,
    }
