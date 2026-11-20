"""
Optimized FAERS fatal-outcome model comparison
------------------------------------------------
Uses the existing output/FAERS_ML_Dataset.parquet.
Does NOT rebuild or modify the source dataset.

Sampling plan:
- Train: up to 150,000 rows from DEV reports dated through 2024-02-08
- Validation: up to 40,000 rows from DEV reports dated 2024-02-09 through 2024-11-06
- Held-out test: up to 75,000 rows from TEST reports dated 2024-11-07 onward

The test set remains chronologically held out. Metrics are sample-based and must
be described as such in reports/presentations. This script is for a fast first
model comparison, not a final full-data deployment model.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# ---------------- CONFIGURATION ----------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "output" / "FAERS_ML_Dataset.parquet"
OUT_DIR = ROOT / "research_results" / "model_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MAX = 150_000
VAL_MAX = 40_000
TEST_MAX = 75_000
RANDOM_STATE = 42

TRAIN_END = pd.Timestamp("2024-02-08")
VAL_START = pd.Timestamp("2024-02-09")
VAL_END = pd.Timestamp("2024-11-06")
TEST_START = pd.Timestamp("2024-11-07")

NUMERIC = [
    "AGE_YEARS", "WT_KG", "DRUG_COUNT",
    "N_PRIMARY_SUSPECT_DRUGS", "N_SECONDARY_SUSPECT_DRUGS"
]
CATEGORICAL = ["SEX", "COUNTRY", "REPORTER_COUNTRY", "OCCP_COD"]
TARGET = "IS_FATAL"
# ------------------------------------------------

def stratified_sample(frame, n, target=TARGET):
    if len(frame) <= n:
        return frame.copy()
    return resample(
        frame, replace=False, n_samples=n,
        random_state=RANDOM_STATE, stratify=frame[target]
    )

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place FAERS_ML_Dataset.parquet in the output folder beside this script."
        )

    print("Loading only required columns from existing Parquet dataset...")
    available = pd.read_parquet(DATA_PATH)
    required = ["SPLIT", "IS_FATAL", "REPORT_DATE"] + NUMERIC + CATEGORICAL
    missing = [c for c in required if c not in available.columns]
    if missing:
        raise ValueError(f"Required columns missing from Parquet: {missing}")

    # Keep known labels only; unknown outcomes are excluded from binary classification.
    available["REPORT_DATE"] = pd.to_datetime(available["REPORT_DATE"], errors="coerce")
    available[TARGET] = pd.to_numeric(available[TARGET], errors="coerce")
    available = available[available[TARGET].isin([0, 1])].copy()

    dev = available[available["SPLIT"].astype(str).str.upper() == "DEV"]
    test = available[available["SPLIT"].astype(str).str.upper() == "TEST"]

    train_pool = dev[dev["REPORT_DATE"] <= TRAIN_END]
    val_pool = dev[
        (dev["REPORT_DATE"] >= VAL_START) &
        (dev["REPORT_DATE"] <= VAL_END)
    ]
    test_pool = test[test["REPORT_DATE"] >= TEST_START]

    if min(len(train_pool), len(val_pool), len(test_pool)) == 0:
        raise ValueError(
            "One or more temporal pools are empty. Check SPLIT and REPORT_DATE values."
        )

    train_df = stratified_sample(train_pool, TRAIN_MAX)
    val_df = stratified_sample(val_pool, VAL_MAX)
    test_df = stratified_sample(test_pool, TEST_MAX)

    features = [c for c in NUMERIC + CATEGORICAL if c in available.columns]
    numeric_features = [c for c in NUMERIC if c in features]
    categorical_features = [c for c in CATEGORICAL if c in features]

    X_train, y_train = train_df[features], train_df[TARGET].astype(int)
    X_val, y_val = val_df[features], val_df[TARGET].astype(int)
    X_test, y_test = test_df[features], test_df[TARGET].astype(int)

    print("\nSample sizes:")
    print(f"Train      : {len(train_df):,} ({train_df.REPORT_DATE.min().date()} to {train_df.REPORT_DATE.max().date()})")
    print(f"Validation : {len(val_df):,} ({val_df.REPORT_DATE.min().date()} to {val_df.REPORT_DATE.max().date()})")
    print(f"Test       : {len(test_df):,} ({test_df.REPORT_DATE.min().date()} to {test_df.REPORT_DATE.max().date()})")
    print("\nClass counts in test:")
    print(y_test.value_counts().sort_index().rename(index={0: "Nonfatal (0)", 1: "Fatal (1)"}))

    # Compatibility with both newer and older scikit-learn releases.
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=20, max_categories=50)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore")

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", encoder)
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, solver="liblinear", class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=16, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=RANDOM_STATE
        )
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=140, max_depth=5, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="logloss", n_jobs=-1,
            random_state=RANDOM_STATE
        )
    except Exception as e:
        print(f"XGBoost unavailable; skipping: {e}")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=140, max_depth=-1, num_leaves=31,
            learning_rate=0.08, class_weight="balanced",
            n_jobs=-1, verbosity=-1, random_state=RANDOM_STATE
        )
    except Exception as e:
        print(f"LightGBM unavailable; skipping: {e}")

    # Use validation data to select a threshold; test is used only once for final evaluation.
    rows = []
    test_curves = {}
    fitted_models = {}

    for name, classifier in models.items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline([
            ("preprocess", preprocess),
            ("model", classifier)
        ])
        pipeline.fit(X_train, y_train)

        val_prob = pipeline.predict_proba(X_val)[:, 1]
        # Choose threshold maximizing validation F1; this threshold is then frozen for test.
        thresholds = np.linspace(0.10, 0.90, 81)
        val_f1 = [
            f1_score(y_val, (val_prob >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        best_threshold = float(thresholds[int(np.argmax(val_f1))])

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= best_threshold).astype(int)

        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Nonfatal", "Fatal"]
        ).plot(ax=ax, cmap="Blues", values_format=",")
        ax.set_title(f"{name} — Held-out Test Confusion Matrix")
        fig.tight_layout()
        safe = name.lower().replace(" ", "_")
        fig.savefig(OUT_DIR / f"{safe}_confusion_matrix.png", dpi=160)
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, test_prob)
        auc = roc_auc_score(y_test, test_prob)
        test_curves[name] = (fpr, tpr, auc)
        fitted_models[name] = pipeline

        rows.append({
            "model": name,
            "test_rows": len(y_test),
            "threshold_selected_on_validation": best_threshold,
            "accuracy": accuracy_score(y_test, test_pred),
            "precision_fatal": precision_score(y_test, test_pred, zero_division=0),
            "recall_fatal": recall_score(y_test, test_pred, zero_division=0),
            "f1_fatal": f1_score(y_test, test_pred, zero_division=0),
            "roc_auc": auc,
            "average_precision": average_precision_score(y_test, test_prob),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1])
        })
        print(f"{name}: ROC-AUC={auc:.4f}, F1={rows[-1]['f1_fatal']:.4f}")

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # Combined ROC plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, auc) in test_curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Chronologically Held-out Test Sample")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curves_all_models.png", dpi=170)
    plt.close(fig)

    # Save split/sample details for transparent reporting.
    split_info = pd.DataFrame([
        {"split": "train", "rows": len(train_df), "start": train_df.REPORT_DATE.min(), "end": train_df.REPORT_DATE.max()},
        {"split": "validation", "rows": len(val_df), "start": val_df.REPORT_DATE.min(), "end": val_df.REPORT_DATE.max()},
        {"split": "test", "rows": len(test_df), "start": test_df.REPORT_DATE.min(), "end": test_df.REPORT_DATE.max()},
    ])
    split_info.to_csv(OUT_DIR / "sample_split_summary.csv", index=False)

    print("\nDONE. Results saved in:")
    print(OUT_DIR)
    print("\nFiles:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f" - {p.name}")
    print("\nIMPORTANT: These are sample-based model results. Report the sample sizes and temporal dates.")

if __name__ == "__main__":
    main()
