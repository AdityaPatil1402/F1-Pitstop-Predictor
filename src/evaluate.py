# ============================================================
#  src/evaluate.py
#  Metrics, confusion matrix, ROC curve, feature importance,
#  SHAP plots — all reusable across notebooks.
#  Usage: from src.evaluate import get_metrics, plot_cm, plot_roc, plot_shap
# ============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)


# ── Shared plot style ────────────────────────────────────────
plt.rcParams["figure.dpi"]        = 120
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["font.size"]         = 12


# ────────────────────────────────────────────────────────────
def get_metrics(model_name: str, y_true, y_pred, y_prob) -> dict:
    """
    Return a dict of all evaluation metrics for one model.
    y_pred : binary predictions (after threshold applied)
    y_prob : raw probability of class=1
    """
    return {
        "Model"     : model_name,
        "Accuracy"  : round(accuracy_score(y_true, y_pred), 4),
        "Precision" : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"    : round(recall_score(y_true, y_pred), 4),
        "F1"        : round(f1_score(y_true, y_pred), 4),
        "ROC-AUC"   : round(roc_auc_score(y_true, y_prob), 4),
    }


# ────────────────────────────────────────────────────────────
def print_report(model_name: str, y_true, y_pred):
    print(f"\n{'='*55}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                target_names=["No Pit", "Pit"]))


# ────────────────────────────────────────────────────────────
def plot_class_balance(df: pd.DataFrame, save=True):
    """Bar chart of PitNextLap class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[cfg.TARGET].value_counts()
    pct    = df[cfg.TARGET].value_counts(normalize=True) * 100
    bars   = ax.bar(
        ["No Pit (0)", "Pit (1)"],
        counts.values,
        color=[cfg.COLORS["secondary"], cfg.COLORS["primary"]],
        edgecolor="none", width=0.5
    )
    for bar, p in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 200,
                f"{p:.1f}%", ha="center", fontsize=11)
    ax.set_title("Target class distribution — PitNextLap", fontsize=13)
    ax.set_ylabel("Count")
    plt.tight_layout()
    if save:
        fig.savefig(cfg.FIG_DIR / "01_class_balance.png",
                    bbox_inches="tight")
    plt.show()


# ────────────────────────────────────────────────────────────
def plot_cm(y_true, y_pred, model_name: str,
            fname="confusion_matrix.png", save=True):
    """Annotated confusion matrix heatmap."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Pit", "Pit"],
                yticklabels=["No Pit", "Pit"],
                linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(cfg.FIG_DIR / fname, bbox_inches="tight")
    plt.show()


# ────────────────────────────────────────────────────────────
def plot_roc(models_probs: list, y_true, save=True):
    """
    Overlay ROC curves for multiple models.
    models_probs : list of (label_str, prob_array)
    """
    palette = [cfg.COLORS["gray"], cfg.COLORS["secondary"],
               cfg.COLORS["blue"], cfg.COLORS["primary"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    for (label, probs), col in zip(models_probs, palette):
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, label=f"{label}  (AUC={auc:.3f})",
                color=col, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    if save:
        fig.savefig(cfg.FIG_DIR / "roc_curves.png", bbox_inches="tight")
    plt.show()


# ────────────────────────────────────────────────────────────
def plot_threshold_curve(model, X_val, y_val, save=True):
    """F1 vs threshold sweep plot. Returns best threshold."""
    probs      = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.20, 0.80, 0.01)
    f1s        = [f1_score(y_val, probs >= t) for t in thresholds]
    best_t     = thresholds[np.argmax(f1s)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, f1s, color=cfg.COLORS["primary"], linewidth=2)
    ax.axvline(best_t, color="gray", linestyle="--", linewidth=1.2,
               label=f"Best threshold = {best_t:.2f}")
    ax.set_xlabel("Decision threshold", fontsize=12)
    ax.set_ylabel("F1 score", fontsize=12)
    ax.set_title("F1 score vs classification threshold", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        fig.savefig(cfg.FIG_DIR / "threshold_curve.png", bbox_inches="tight")
    plt.show()
    return float(best_t)


# ────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names: list,
                            top_n=15, save=True):
    """XGBoost / RF built-in feature importance bar chart."""
    imp = model.feature_importances_
    fi  = (pd.DataFrame({"Feature": feature_names, "Importance": imp})
             .sort_values("Importance", ascending=False)
             .head(top_n))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fi["Feature"][::-1], fi["Importance"][::-1],
            color=cfg.COLORS["blue"], edgecolor="none")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13)
    plt.tight_layout()
    if save:
        fig.savefig(cfg.FIG_DIR / "feature_importance.png", bbox_inches="tight")
    plt.show()


# ────────────────────────────────────────────────────────────
def plot_shap_summary(model, X_sample, feature_names, save=True):
    """SHAP beeswarm + bar summary plots."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # Beeswarm
    shap.summary_plot(shap_vals, X_sample,
                      feature_names=feature_names,
                      max_display=15, show=False)
    plt.title("SHAP feature importance — beeswarm", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(cfg.FIG_DIR / "shap_beeswarm.png", bbox_inches="tight")
    plt.show()

    # Bar
    shap.summary_plot(shap_vals, X_sample,
                      feature_names=feature_names,
                      plot_type="bar", max_display=15, show=False)
    plt.title("SHAP — mean |SHAP| feature importance", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(cfg.FIG_DIR / "shap_bar.png", bbox_inches="tight")
    plt.show()

    return shap_vals


# ────────────────────────────────────────────────────────────
def plot_shap_waterfall(model, X_sample, feature_names,
                        idx=0, save=True):
    """Single-prediction SHAP waterfall — great for presentation."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    exp = shap.Explanation(
        values      = shap_vals[idx],
        base_values = explainer.expected_value,
        data        = X_sample.iloc[idx] if hasattr(X_sample, "iloc") else X_sample[idx],
        feature_names = feature_names
    )
    shap.plots.waterfall(exp, show=False)
    plt.title("SHAP waterfall — single lap explanation", fontsize=13)
    plt.tight_layout()
    if save:
        plt.savefig(cfg.FIG_DIR / "shap_waterfall.png", bbox_inches="tight")
    plt.show()


# ────────────────────────────────────────────────────────────
def comparison_table(results_list: list) -> pd.DataFrame:
    """
    results_list : list of dicts from get_metrics().
    Returns a styled DataFrame and saves to CSV.
    """
    df = pd.DataFrame(results_list).set_index("Model")
    df.to_csv(cfg.RESULTS_CSV)
    print(f"\nResults saved → {cfg.RESULTS_CSV}")
    print(df.to_string())
    return df
