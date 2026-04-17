"""
src/train.py
------------
Training script — run this ONCE to produce saved model artefacts.

Usage:
    python src/train.py

Outputs (written to outputs/models/):
    lr_pipeline.joblib   — TF-IDF + Logistic Regression (calibrated probabilities)
    svc_pipeline.joblib  — TF-IDF + LinearSVC (calibrated, fastest at inference)
    tfidf_vocab.joblib   — standalone vectorizer (for inspection / LLM routing)

Metrics are written to outputs/metrics/track1_metrics.csv
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "outputs", "models")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

for d in [MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Add project root to path so src imports work
sys.path.insert(0, BASE_DIR)
from src.preprocessing import load_data, clean_data, split_data

SEED = 42
np.random.seed(SEED)

TFIDF_PARAMS = dict(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)


def build_and_train(X_train, y_train):
    """Train LR and LinearSVC pipelines. Returns (lr_pipe, svc_pipe)."""

    print("\n[train] Fitting Logistic Regression pipeline ...")
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
            C=1.0,
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    print("[train] LR done.")

    print("[train] Fitting LinearSVC pipeline ...")
    svc_base = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=2000)
    svc_cal  = CalibratedClassifierCV(svc_base, cv=3)
    svc_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   svc_cal),
    ])
    svc_pipe.fit(X_train, y_train)
    print("[train] SVC done.")

    return lr_pipe, svc_pipe


def evaluate(name, pipeline, X_test, y_test, figures_dir, metrics_dir):
    """Compute and save evaluation metrics + plots."""
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    row = {
        "Model"       : name,
        "Accuracy"    : round(accuracy_score(y_test, y_pred), 4),
        "F1-Macro"    : round(f1_score(y_test, y_pred, average="macro"), 4),
        "F1-Weighted" : round(f1_score(y_test, y_pred, average="weighted"), 4),
        "ROC-AUC"     : round(roc_auc_score(y_test, y_proba), 4),
        "PR-AUC"      : round(average_precision_score(y_test, y_proba), 4),
    }

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    for k, v in row.items():
        if k != "Model":
            print(f"  {k:<14}: {v}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    return row, y_pred, y_proba


def save_plots(results, X_test, y_test, figures_dir):
    """Save confusion matrix and ROC/PR curves."""

    # Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, y_pred, _) in zip(axes, results):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Fake", "Real"],
                    yticklabels=["Fake", "Real"])
        ax.set_title(f"Confusion Matrix\n{name}", fontweight="bold")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "05_confusion_matrices.png"), dpi=150)
    plt.close()

    # ROC + PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    linestyles = ["-", "--"]

    for (name, _, y_proba), ls in zip(results, linestyles):
        fpr, tpr, _   = roc_curve(y_test, y_proba)
        prec, rec, _  = precision_recall_curve(y_test, y_proba)
        auc_val       = roc_auc_score(y_test, y_proba)
        pr_auc        = average_precision_score(y_test, y_proba)

        axes[0].plot(fpr, tpr, ls=ls, label=f"{name} (AUC={auc_val:.3f})")
        axes[1].plot(rec, prec, ls=ls, label=f"{name} (PR-AUC={pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], "k:", label="Random")
    axes[0].set_title("ROC Curve", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    axes[1].set_title("Precision-Recall Curve", fontweight="bold")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "06_roc_pr_curves.png"), dpi=150)
    plt.close()

    print(f"[plots] Saved confusion matrices and ROC/PR curves to {figures_dir}")


def cross_validate(pipeline, X, y):
    """5-fold stratified CV on the full dataset."""
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"[cv] 5-Fold F1-Macro: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"[cv] Per-fold: {[round(s, 4) for s in scores]}")
    return scores


def main():
    print("=" * 60)
    print("  Fake vs Real News — Training Pipeline")
    print("=" * 60)

    # 1. Load & clean
    df      = load_data(BASE_DIR)
    clean   = clean_data(df)

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(clean)
    X = clean["content"]
    y = clean["label"]

    # 3. Train
    lr_pipe, svc_pipe = build_and_train(X_train, y_train)

    # 4. Evaluate
    eval_results = []
    metric_rows  = []
    pred_results = []

    for name, pipe in [("Logistic Regression", lr_pipe), ("LinearSVC (calibrated)", svc_pipe)]:
        row, y_pred, y_proba = evaluate(name, pipe, X_test, y_test, FIGURES_DIR, METRICS_DIR)
        metric_rows.append(row)
        eval_results.append((name, y_pred, y_proba))
        pred_results.append((name, y_pred))

    # 5. Plots
    save_plots(eval_results, X_test, y_test, FIGURES_DIR)

    # 6. Cross-validation (LR only for speed)
    print("\n[cv] Running 5-fold cross-validation on Logistic Regression ...")
    cross_validate(lr_pipe, X, y)

    # 7. Save metrics
    metrics_df = pd.DataFrame(metric_rows).set_index("Model")
    metrics_path = os.path.join(METRICS_DIR, "track1_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\n[metrics] Saved to {metrics_path}")

    # 8. Save models
    lr_path  = os.path.join(MODELS_DIR, "lr_pipeline.joblib")
    svc_path = os.path.join(MODELS_DIR, "svc_pipeline.joblib")
    joblib.dump(lr_pipe,  lr_path)
    joblib.dump(svc_pipe, svc_path)
    print(f"[models] Saved LR  → {lr_path}")
    print(f"[models] Saved SVC → {svc_path}")

    # 9. Smoke test — reload and predict one article
    loaded = joblib.load(svc_path)
    sample = ["WASHINGTON (Reuters) - The Senate passed the $1.9 trillion bill on Friday."]
    pred   = loaded.predict(sample)[0]
    conf   = loaded.predict_proba(sample)[0].max()
    print(f"\n[smoke test] '{sample[0][:60]}...'")
    print(f"             → {'Real' if pred == 1 else 'Fake'} (confidence={conf:.4f})")

    print("\n[done] Training complete. Models saved.")
    print(f"       Next: uvicorn app.api:app --reload")
    print(f"       Or  : streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
