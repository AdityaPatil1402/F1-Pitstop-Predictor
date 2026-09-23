# ============================================================
#  src/train.py
#  Train Logistic Regression, Random Forest, XGBoost.
#  Also: threshold tuning + Optuna hyperparameter search.
#  Usage: from src.train import train_lr, train_rf, train_xgb, tune_threshold
# ============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier
from sklearn.metrics         import f1_score, roc_auc_score

import config as cfg


# ────────────────────────────────────────────────────────────
def train_lr(X_train_scaled, y_train):
    """
    Logistic Regression baseline.
    Requires StandardScaler-transformed inputs.
    class_weight='balanced' compensates for 75/25 imbalance.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print("[train_lr] Logistic Regression fitted.")
    return model


# ────────────────────────────────────────────────────────────
def train_rf(X_train, y_train, **kwargs):
    """
    Random Forest — comparison model (no scaling needed).
    """
    params = {**cfg.RF_DEFAULTS, **kwargs}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print(f"[train_rf] Random Forest fitted. params={params}")
    return model


# ────────────────────────────────────────────────────────────
def train_xgb(X_train, y_train, X_val, y_val, spw: float, **kwargs):
    """
    XGBoost — main model.
    spw : scale_pos_weight = (neg count / pos count)
    """
    params = {**cfg.XGB_DEFAULTS, "scale_pos_weight": spw, **kwargs}
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    print(f"[train_xgb] XGBoost fitted. n_estimators={params['n_estimators']}")
    return model


# ────────────────────────────────────────────────────────────
def tune_threshold(model, X_val, y_val, step=0.01) -> tuple[float, float]:
    """
    Sweep decision thresholds on validation set and return
    (best_threshold, best_f1).
    Works with any model that has predict_proba().
    """
    probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.20, 0.80, step)
    f1s = [f1_score(y_val, probs >= t) for t in thresholds]
    best_t = float(thresholds[np.argmax(f1s)])
    best_f = float(max(f1s))
    print(f"[tune_threshold] Best threshold={best_t:.2f}  Best F1={best_f:.4f}")
    return best_t, best_f


# ────────────────────────────────────────────────────────────
def tune_xgb_optuna(X_train, y_train, X_val, y_val,
                    spw: float, n_trials: int = 50):
    """
    Optuna hyperparameter search for XGBoost.
    Optimises F1 on validation set (with threshold sweep per trial).
    Returns (best_params, best_value).
    """
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 600),
            max_depth        = trial.suggest_int("max_depth", 3, 9),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            gamma            = trial.suggest_float("gamma", 0, 5),
            scale_pos_weight = spw,
            eval_metric      = "aucpr",
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, verbose=False)
        probs = m.predict_proba(X_val)[:, 1]
        thr   = np.arange(0.20, 0.80, 0.05)
        return max(f1_score(y_val, probs >= t) for t in thr)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update(dict(
        scale_pos_weight=spw, eval_metric="aucpr",
        random_state=42, n_jobs=-1, verbosity=0
    ))
    print(f"[tune_xgb_optuna] Best val F1={study.best_value:.4f}")
    print(f"[tune_xgb_optuna] Best params: {best}")
    return best, study.best_value


# ────────────────────────────────────────────────────────────
def save_model(model, name: str):
    """Save model to outputs/models/."""
    path = cfg.MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"[save_model] Saved → {path}")


def load_model(name: str):
    path = cfg.MODEL_DIR / f"{name}.joblib"
    model = joblib.load(path)
    print(f"[load_model] Loaded ← {path}")
    return model
