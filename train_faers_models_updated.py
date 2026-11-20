#!/usr/bin/env python3
"""
Complete FAERS model analysis aligned with the user's 8-stage methodology.

Input:
    output/FAERS_ML_Dataset.parquet
    (created by build_faers_dataset(1).py / the FAERS dataset builder)

Implements from the available ML parquet:
    Stage 3: EDA tables and plots
    Stage 4: leakage-safe feature selection and audit
    Stage 5: Logistic Regression, Random Forest, XGBoost, LightGBM;
             class-imbalance handling and small validation-based tuning
    Stage 6: chronological train/validation/test; AUROC, AUPRC, F1,
             Brier score, confusion matrices, ROC/PR and calibration plots
    Stage 7: global and local SHAP explanations for the best validation model

Important source-data limitations:
    The current builder's PRIMARY feature set does not contain a clearly
    defined dose variable or a drug-class variable. This script will use
    those only if explicitly named columns are present in the parquet.
    It does not invent either variable. See run_notes.txt and feature_audit.csv.

FAERS interpretation:
    IS_FATAL is a reported death outcome code (DE), not proof that a medicine
    caused death and not an individual's mortality risk.
"""

from pathlib import Path
import itertools
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import ParameterGrid

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "output" / "FAERS_ML_Dataset.parquet"
RESULT_DIR = BASE_DIR / "research_results"
EDA_DIR = RESULT_DIR / "eda"
SHAP_DIR = RESULT_DIR / "shap"
for folder in (RESULT_DIR, EDA_DIR, SHAP_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20
MAX_CATEGORIES = 200
MIN_CATEGORY_FREQUENCY = 20
SHAP_SAMPLE_SIZE = 250
SHAP_BACKGROUND_SIZE = 80
THRESHOLD = 0.50

# Features created by the supplied builder's PRIMARY leakage-reduced set.
BASE_NUMERIC = [
    "AGE_YEARS", "WT_KG", "DRUG_COUNT",
    "N_PRIMARY_SUSPECT_DRUGS", "N_SECONDARY_SUSPECT_DRUGS",
    "REACTION_COUNT_NONLEAK",
]
BASE_CATEGORICAL = [
    "SEX", "COUNTRY", "REPORTER_COUNTRY", "OCCP_COD",
    "PRIMARY_SUSPECT_DRUGS", "SECONDARY_SUSPECT_DRUGS",
    "INDICATIONS", "REPORT_SOURCES", "REACTIONS_NONLEAK",
]
# Optional only if genuinely present in the parquet. The supplied builder
# currently does not show explicit dose or drug-class columns.
OPTIONAL_NUMERIC = ["DOSE", "DOSE_AMT", "DOSE_UNIT", "DAILY_DOSE"]
OPTIONAL_CATEGORICAL = ["DRUG_CLASS", "ATC_CLASS", "THERAPEUTIC_CLASS"]

TARGET = "IS_FATAL"
FORBIDDEN_PREDICTORS = {
    "IS_FATAL", "OUTCOME_CODES", "PRIMARYID", "CASEID", "REPORT_DATE",
    "CASE_FIRST_REPORT_DATE", "CASE_FIRST_REPORT_PERIOD", "SPLIT",
    "IS_LATEST_VERSION", "REACTIONS", "REACTION_COUNT",
    "REACTIONS_NO_EXPLICIT_DEATH",
}


def save_table(df, name):
    df.to_csv(RESULT_DIR / name, index=False)


def make_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=MIN_CATEGORY_FREQUENCY,
            max_categories=MAX_CATEGORIES,
        )),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ], remainder="drop")


def make_estimator(name, params, y_train):
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced",
            solver="saga", random_state=RANDOM_STATE,
            C=params.get("C", 1.0), n_jobs=-1
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 250),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 250),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=neg / pos,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 250),
            num_leaves=params.get("num_leaves", 31),
            learning_rate=params.get("learning_rate", 0.05),
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
        )
    raise ValueError(name)


def available_models():
    names = ["Logistic Regression", "Random Forest"]
    try:
        import xgboost  # noqa
        names.append("XGBoost")
    except ImportError:
        print("NOTE: xgboost is not installed; XGBoost will be skipped.")
    try:
        import lightgbm  # noqa
        names.append("LightGBM")
    except ImportError:
        print("NOTE: lightgbm is not installed; LightGBM will be skipped.")
    return names


def parameter_grid(name):
    # Small, reproducible search. Validation data selects settings;
    # the held-out TEST period is not used for tuning.
    grids = {
        "Logistic Regression": {"C": [0.1, 1.0, 5.0]},
        "Random Forest": {
            "n_estimators": [200, 350],
            "max_depth": [None, 20],
            "min_samples_leaf": [2, 5],
        },
        "XGBoost": {
            "n_estimators": [200, 350],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
        "LightGBM": {
            "n_estimators": [200, 350],
            "num_leaves": [15, 31],
            "learning_rate": [0.05, 0.1],
        },
    }
    return list(ParameterGrid(grids[name]))


def metric_dict(y, p, threshold=THRESHOLD):
    pred = (p >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1_Score": f1_score(y, pred, zero_division=0),
        "AUROC": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "AUPRC": average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "Brier_Score": brier_score_loss(y, p),
    }


def plot_eda(df, numeric_cols, categorical_cols):
    """EDA on the available report-level ML dataset."""
    # Outcome counts
    outcome = df[TARGET].value_counts(dropna=False).rename_axis("IS_FATAL").reset_index(name="REPORTS")
    save_table(outcome, "eda_outcome_counts.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Non-fatal (0)" if x == 0 else "Fatal (1)" if x == 1 else "Unknown" for x in outcome["IS_FATAL"]]
    ax.bar(labels, outcome["REPORTS"])
    ax.set_title("FAERS Reported Outcome Distribution")
    ax.set_ylabel("Number of reports")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "outcome_distribution.png", dpi=300)
    plt.close(fig)

    # Age distribution
    if "AGE_YEARS" in df.columns:
        age = pd.to_numeric(df["AGE_YEARS"], errors="coerce")
        age_summary = age.describe().to_frame("AGE_YEARS").reset_index(names="STATISTIC")
        age_summary.to_csv(EDA_DIR / "age_summary.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(age.dropna().clip(lower=0, upper=120), bins=30)
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age in years")
        ax.set_ylabel("Reports")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "age_distribution.png", dpi=300)
        plt.close(fig)

    # Sex distribution
    if "SEX" in df.columns:
        sex = df["SEX"].fillna("UNKNOWN").astype(str).value_counts().head(15)
        sex.to_csv(EDA_DIR / "sex_distribution.csv", header=["REPORTS"])
        fig, ax = plt.subplots(figsize=(7, 4))
        sex.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Sex Distribution")
        ax.set_xlabel("Reports")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "sex_distribution.png", dpi=300)
        plt.close(fig)

    # Reporter type
    if "OCCP_COD" in df.columns:
        reporter = df["OCCP_COD"].fillna("UNKNOWN").astype(str).value_counts().head(15)
        reporter.to_csv(EDA_DIR / "reporter_type_distribution.csv", header=["REPORTS"])
        fig, ax = plt.subplots(figsize=(8, 4))
        reporter.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Reporter Type Distribution (OCCP_COD)")
        ax.set_xlabel("Reports")
        fig.tight_layout()
        fig.savefig(EDA_DIR / "reporter_type_distribution.png", dpi=300)
        plt.close(fig)

    # Fatal/nonfatal comparison for categorical variables
    comparison_rows = []
    for col in ["SEX", "OCCP_COD", "COUNTRY", "REPORTER_COUNTRY"]:
        if col in df.columns:
            tab = pd.crosstab(df[col].fillna("UNKNOWN"), df[TARGET], dropna=False)
            tab.to_csv(EDA_DIR / f"{col.lower()}_by_fatality.csv")
            for category, row in tab.iterrows():
                comparison_rows.append({
                    "Feature": col, "Category": str(category),
                    "Nonfatal_reports": int(row.get(0, 0)),
                    "Fatal_reports": int(row.get(1, 0)),
                })
    if comparison_rows:
        save_table(pd.DataFrame(comparison_rows), "eda_fatal_nonfatal_categorical_comparison.csv")

    # Common drug/reaction text tokens (report-level strings; not inferred classes)
    for col, outname in [
        ("PRIMARY_SUSPECT_DRUGS", "top_primary_suspect_drugs.csv"),
        ("REACTIONS_NONLEAK", "top_nonleak_reactions.csv"),
    ]:
        if col in df.columns:
            counts = (df[col].dropna().astype(str).str.split(r";\s*")
                      .explode().str.strip().value_counts().head(50))
            counts.rename_axis("TERM").reset_index(name="REPORTS").to_csv(
                EDA_DIR / outname, index=False
            )

    # Numeric feature comparison by outcome
    numeric_present = [c for c in numeric_cols if c in df.columns]
    if numeric_present:
        summaries = df[df[TARGET].isin([0, 1])].groupby(TARGET)[numeric_present].agg(
            ["count", "mean", "median", "std"]
        )
        summaries.to_csv(EDA_DIR / "numeric_features_by_outcome.csv")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_FILE}\n"
            "Run the FAERS dataset builder first. It must create "
            "output/FAERS_ML_Dataset.parquet."
        )

    print("Loading FAERS ML dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Rows loaded: {len(df):,}")

    required = ["SPLIT", TARGET, "IS_LATEST_VERSION"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Use one latest report version per case, as advised by the builder.
    df = df[df["IS_LATEST_VERSION"] == 1].copy()
    df = df[df["SPLIT"].isin(["DEV", "TEST"])].copy()

    # Use the builder's PRIMARY features, plus dose/class only if actual columns exist.
    numeric_cols = [c for c in BASE_NUMERIC + OPTIONAL_NUMERIC if c in df.columns]
    categorical_cols = [c for c in BASE_CATEGORICAL + OPTIONAL_CATEGORICAL if c in df.columns]
    feature_cols = numeric_cols + categorical_cols
    feature_cols = [c for c in feature_cols if c not in FORBIDDEN_PREDICTORS]
    if not feature_cols:
        raise ValueError("No approved predictor columns found in the parquet file.")

    # Dataset overview and feature audit
    overview = pd.DataFrame([{
        "Rows_after_latest_DEV_TEST_filter": len(df),
        "DEV_rows_known_outcome": int(((df.SPLIT == "DEV") & df[TARGET].isin([0, 1])).sum()),
        "TEST_rows_known_outcome": int(((df.SPLIT == "TEST") & df[TARGET].isin([0, 1])).sum()),
        "Fatal_rows": int((df[TARGET] == 1).sum()),
        "Nonfatal_rows": int((df[TARGET] == 0).sum()),
        "Unknown_outcome_rows": int(df[TARGET].isna().sum()),
        "First_report_date_min": str(df["CASE_FIRST_REPORT_DATE"].min()) if "CASE_FIRST_REPORT_DATE" in df else "",
        "First_report_date_max": str(df["CASE_FIRST_REPORT_DATE"].max()) if "CASE_FIRST_REPORT_DATE" in df else "",
    }])
    save_table(overview, "dataset_overview.csv")

    audit_rows = []
    for col in feature_cols:
        audit_rows.append({
            "Feature": col,
            "Type": "numeric" if col in numeric_cols else "categorical",
            "Missing_count": int(df[col].isna().sum()),
            "Missing_percent": round(float(df[col].isna().mean() * 100), 3),
            "Unique_values": int(df[col].nunique(dropna=True)),
            "Included_as_predictor": True,
        })
    for col in sorted(FORBIDDEN_PREDICTORS.intersection(df.columns)):
        audit_rows.append({
            "Feature": col, "Type": "excluded",
            "Missing_count": int(df[col].isna().sum()),
            "Missing_percent": round(float(df[col].isna().mean() * 100), 3),
            "Unique_values": int(df[col].nunique(dropna=True)),
            "Included_as_predictor": False,
        })
    save_table(pd.DataFrame(audit_rows), "feature_audit.csv")

    notes = []
    if not any(c in df.columns for c in OPTIONAL_NUMERIC):
        notes.append("DOSE FEATURE NOT AVAILABLE: the supplied builder's ML feature list does not expose a dose column.")
    if not any(c in df.columns for c in OPTIONAL_CATEGORICAL):
        notes.append("DRUG CLASS FEATURE NOT AVAILABLE: the supplied builder's ML feature list does not expose an explicit drug-class/ATC column.")
    notes.append("Drug names and reaction terms are text features, not validated pharmacological drug classes.")
    notes.append("The builder's FAERS_Data_Quality_Report.xlsx and FAERS_ML_Design_and_Leakage_Audit.xlsx should be retained as preprocessing/de-duplication evidence.")
    (RESULT_DIR / "data_availability_notes.txt").write_text("\n".join(notes), encoding="utf-8")

    # EDA uses known outcomes and available fields. Does not claim unavailable variables.
    eda_df = df[df[TARGET].isin([0, 1])].copy()
    plot_eda(eda_df, numeric_cols, categorical_cols)

    # Prepare chronological DEV -> train/validation split using case first-report date.
    dev = df[(df["SPLIT"] == "DEV") & df[TARGET].isin([0, 1])].copy()
    test = df[(df["SPLIT"] == "TEST") & df[TARGET].isin([0, 1])].copy()
    if dev.empty or test.empty:
        raise ValueError("DEV or TEST has no known-outcome rows. Check the builder split summary.")
    if "CASE_FIRST_REPORT_DATE" not in dev.columns:
        raise ValueError("CASE_FIRST_REPORT_DATE is required for chronological validation splitting.")

    dev["CASE_FIRST_REPORT_DATE"] = pd.to_datetime(dev["CASE_FIRST_REPORT_DATE"], errors="coerce")
    test["CASE_FIRST_REPORT_DATE"] = pd.to_datetime(test["CASE_FIRST_REPORT_DATE"], errors="coerce")
    dev = dev.dropna(subset=["CASE_FIRST_REPORT_DATE"]).sort_values("CASE_FIRST_REPORT_DATE")
    unique_dates = np.sort(dev["CASE_FIRST_REPORT_DATE"].dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough distinct DEV case dates to create a chronological validation split.")
    cut_index = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * (1 - VALIDATION_FRACTION))))
    validation_start = pd.Timestamp(unique_dates[cut_index])
    train = dev[dev["CASE_FIRST_REPORT_DATE"] < validation_start].copy()
    validation = dev[dev["CASE_FIRST_REPORT_DATE"] >= validation_start].copy()
    if train.empty or validation.empty:
        raise ValueError("Chronological train/validation split is empty; inspect date coverage.")

    # Numeric conversion before imputation.
    for col in numeric_cols:
        for frame in (train, validation, dev, test):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    X_train, y_train = train[feature_cols], train[TARGET].astype(int)
    X_val, y_val = validation[feature_cols], validation[TARGET].astype(int)
    X_dev, y_dev = dev[feature_cols], dev[TARGET].astype(int)
    X_test, y_test = test[feature_cols], test[TARGET].astype(int)

    split_summary = pd.DataFrame([
        {"Partition": "Train", "Rows": len(train), "Fatal": int((y_train == 1).sum()), "Nonfatal": int((y_train == 0).sum()), "Start": str(train.CASE_FIRST_REPORT_DATE.min()), "End": str(train.CASE_FIRST_REPORT_DATE.max())},
        {"Partition": "Validation", "Rows": len(validation), "Fatal": int((y_val == 1).sum()), "Nonfatal": int((y_val == 0).sum()), "Start": str(validation.CASE_FIRST_REPORT_DATE.min()), "End": str(validation.CASE_FIRST_REPORT_DATE.max())},
        {"Partition": "Final DEV refit", "Rows": len(dev), "Fatal": int((y_dev == 1).sum()), "Nonfatal": int((y_dev == 0).sum()), "Start": str(dev.CASE_FIRST_REPORT_DATE.min()), "End": str(dev.CASE_FIRST_REPORT_DATE.max())},
        {"Partition": "Held-out TEST", "Rows": len(test), "Fatal": int((y_test == 1).sum()), "Nonfatal": int((y_test == 0).sum()), "Start": str(test.CASE_FIRST_REPORT_DATE.min()), "End": str(test.CASE_FIRST_REPORT_DATE.max())},
    ])
    save_table(split_summary, "temporal_split_summary.csv")
    print("\nTemporal split:")
    print(split_summary.to_string(index=False))
    print("\nPredictors:", ", ".join(feature_cols))

    # Hyperparameter tuning uses validation only; test is not touched here.
    tuned = {}
    tuning_rows = []
    model_names = available_models()
    for name in model_names:
        print(f"\nTuning {name}...")
        best_score = -np.inf
        best_params = None
        for params in parameter_grid(name):
            try:
                pipe = Pipeline([
                    ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
                    ("model", make_estimator(name, params, y_train)),
                ])
                pipe.fit(X_train, y_train)
                val_prob = pipe.predict_proba(X_val)[:, 1]
                score = average_precision_score(y_val, val_prob) if len(np.unique(y_val)) == 2 else -np.inf
                tuning_rows.append({
                    "Model": name, "Parameters": json.dumps(params),
                    "Validation_AUPRC": score,
                })
                if score > best_score:
                    best_score, best_params = score, params
            except Exception as exc:
                tuning_rows.append({
                    "Model": name, "Parameters": json.dumps(params),
                    "Validation_AUPRC": np.nan, "Error": str(exc)[:500],
                })
                print(f"  Candidate skipped: {exc}")
        if best_params is not None:
            tuned[name] = best_params
            print(f"  Best validation AUPRC: {best_score:.5f}; params={best_params}")
        else:
            print(f"  No successful configuration for {name}.")

    if not tuned:
        raise RuntimeError("No model trained successfully. Check dependencies and input columns.")
    save_table(pd.DataFrame(tuning_rows), "hyperparameter_tuning_results.csv")

    # Refit each selected configuration on all DEV data, then evaluate TEST once.
    metric_rows = []
    prediction_frames = []
    fitted_pipelines = {}
    for name, params in tuned.items():
        print(f"\nFinal DEV fit and TEST evaluation: {name}")
        pipe = Pipeline([
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
            ("model", make_estimator(name, params, y_dev)),
        ])
        pipe.fit(X_dev, y_dev)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= THRESHOLD).astype(int)
        metrics = metric_dict(y_test, prob)
        metrics.update({"Model": name, "Best_Parameters": json.dumps(params), "Threshold": THRESHOLD})
        metric_rows.append(metrics)
        fitted_pipelines[name] = pipe

        safe = name.lower().replace(" ", "_")
        import joblib
        joblib.dump(pipe, RESULT_DIR / f"{safe}_pipeline.joblib")

        pred_frame = pd.DataFrame({
            "CASEID": test["CASEID"].values if "CASEID" in test.columns else np.arange(len(test)),
            "ACTUAL_IS_FATAL": y_test.values,
            "PREDICTED_IS_FATAL": pred,
            "PREDICTED_FATAL_PROBABILITY": prob,
            "MODEL": name,
        })
        prediction_frames.append(pred_frame)

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=["Non-fatal", "Fatal"]).plot(
            ax=ax, values_format="d", colorbar=False
        )
        ax.set_title(f"{name} - Held-out Test Confusion Matrix")
        fig.tight_layout()
        fig.savefig(RESULT_DIR / f"{safe}_confusion_matrix.png", dpi=300)
        plt.close(fig)

    metrics_df = pd.DataFrame(metric_rows).sort_values("AUPRC", ascending=False)
    metrics_df.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_df.to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    # Comparison chart
    plot_cols = ["AUROC", "AUPRC", "F1_Score"]
    ax = metrics_df.set_index("Model")[plot_cols].plot(kind="bar", figsize=(10, 6), ylim=(0, 1), rot=15)
    ax.set_title("Model Comparison on Chronological Held-out Test Set")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "model_metric_comparison.png", dpi=300)
    plt.close()

    # ROC and PR curves
    if len(np.unique(y_test)) == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        for name in tuned:
            sub = predictions_df[predictions_df.MODEL == name]
            RocCurveDisplay.from_predictions(sub.ACTUAL_IS_FATAL, sub.PREDICTED_FATAL_PROBABILITY, name=name, ax=ax)
        ax.set_title("ROC Curves - Held-out Test Set")
        fig.tight_layout()
        fig.savefig(RESULT_DIR / "roc_curves.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        for name in tuned:
            sub = predictions_df[predictions_df.MODEL == name]
            PrecisionRecallDisplay.from_predictions(sub.ACTUAL_IS_FATAL, sub.PREDICTED_FATAL_PROBABILITY, name=name, ax=ax)
        ax.set_title("Precision-Recall Curves - Held-out Test Set")
        fig.tight_layout()
        fig.savefig(RESULT_DIR / "precision_recall_curves.png", dpi=300)
        plt.close(fig)

    # Calibration curve for each model
    if len(np.unique(y_test)) == 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        for name in tuned:
            sub = predictions_df[predictions_df.MODEL == name]
            CalibrationDisplay.from_predictions(
                sub.ACTUAL_IS_FATAL, sub.PREDICTED_FATAL_PROBABILITY,
                n_bins=10, strategy="quantile", name=name, ax=ax
            )
        ax.set_title("Probability Calibration - Held-out Test Set")
        fig.tight_layout()
        fig.savefig(RESULT_DIR / "calibration_curves.png", dpi=300)
        plt.close(fig)

    # SHAP global and local explanations for the best validation-selected model.
    best_name = max(tuned, key=lambda n: next(
        (r["Validation_AUPRC"] for r in tuning_rows
         if r["Model"] == n and r.get("Parameters") == json.dumps(tuned[n])), -np.inf
    ))
    best_pipe = fitted_pipelines[best_name]
    shap_status = "SHAP was not run."
    try:
        import shap
        pre = best_pipe.named_steps["preprocess"]
        model = best_pipe.named_steps["model"]
        sample_n = min(SHAP_SAMPLE_SIZE, len(X_test))
        X_raw_sample = X_test.sample(sample_n, random_state=RANDOM_STATE)
        X_transformed = pre.transform(X_raw_sample)
        feature_names = pre.get_feature_names_out()
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        X_transformed = np.asarray(X_transformed)

        # TreeExplainer supports RF and common boosting trees.
        if best_name in ("Random Forest", "XGBoost", "LightGBM"):
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_transformed)
            if isinstance(values, list):
                values = values[-1]
            elif getattr(values, "ndim", 0) == 3:
                values = values[:, :, -1]
            exp = shap.Explanation(
                values=values,
                data=X_transformed,
                feature_names=feature_names
            )
        else:
            bg_n = min(SHAP_BACKGROUND_SIZE, len(X_train))
            background_raw = X_train.sample(bg_n, random_state=RANDOM_STATE)
            background = pre.transform(background_raw)
            if hasattr(background, "toarray"):
                background = background.toarray()
            background = np.asarray(background)
            explainer = shap.LinearExplainer(model, background)
            values = explainer.shap_values(X_transformed)
            exp = shap.Explanation(
                values=values,
                data=X_transformed,
                feature_names=feature_names
            )

        plt.figure()
        shap.plots.bar(exp, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "global_shap_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        shap.plots.beeswarm(exp, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "global_shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Local explanation for the first test example in the SHAP sample.
        plt.figure()
        shap.plots.waterfall(exp[0], max_display=15, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "local_shap_waterfall.png", dpi=300, bbox_inches="tight")
        plt.close()
        shap_status = f"SHAP completed for {best_name}; sample size={sample_n}."
    except ImportError:
        shap_status = "SHAP package is not installed. Install with: pip install shap"
    except Exception as exc:
        shap_status = f"SHAP attempted for {best_name} but failed: {type(exc).__name__}: {exc}"

    (SHAP_DIR / "shap_status.txt").write_text(shap_status, encoding="utf-8")

    # Detailed reproducibility notes and transparent limitations.
    with open(RESULT_DIR / "run_notes.txt", "w", encoding="utf-8") as f:
        f.write("FAERS Fatal Adverse Drug Event Analysis - Model Experiment\n")
        f.write("=========================================================\n")
        f.write(f"Input: {INPUT_FILE}\n")
        f.write(f"Rows after latest-version and DEV/TEST filtering: {len(df):,}\n")
        f.write(f"Train rows: {len(train):,}; validation rows: {len(validation):,}; DEV refit rows: {len(dev):,}; TEST rows: {len(test):,}\n")
        f.write(f"Validation period starts: {validation_start}\n")
        f.write(f"Features: {', '.join(feature_cols)}\n")
        f.write(f"Numeric features: {', '.join(numeric_cols)}\n")
        f.write(f"Categorical features: {', '.join(categorical_cols)}\n")
        f.write(f"Models successfully tuned: {', '.join(tuned.keys())}\n")
        f.write(f"Best model selected by validation AUPRC: {best_name}\n")
        f.write(f"SHAP: {shap_status}\n")
        f.write(f"Classification threshold: {THRESHOLD}\n")
        f.write("Tuning used chronological validation data only. Final model refit used all DEV rows. TEST was held out for final evaluation.\n")
        f.write("Class imbalance: balanced class weights for Logistic Regression/Random Forest/LightGBM; scale_pos_weight for XGBoost.\n")
        f.write("Preprocessing: median imputation and standard scaling for numeric variables; UNKNOWN imputation and one-hot encoding for categorical variables.\n")
        f.write("No outcome codes, identifiers, report dates, split columns, or explicit-death reaction fields were used as predictors.\n")
        f.write("Dose/drug-class features are included only if their explicit columns exist; inspect data_availability_notes.txt.\n")
        f.write("IS_FATAL=1 means FAERS outcome code DE was reported; this is not proof that a drug caused death or a measure of individual mortality risk.\n")
        f.write("FAERS is a spontaneous-reporting database; findings are associations and are subject to under-reporting, missingness, and reporting bias.\n")

    print("\n" + "=" * 68)
    print("ANALYSIS FINISHED")
    print(f"Results folder: {RESULT_DIR}")
    print("Key outputs:")
    print(" - dataset_overview.csv, feature_audit.csv, temporal_split_summary.csv")
    print(" - eda/ (EDA tables and figures)")
    print(" - hyperparameter_tuning_results.csv")
    print(" - model_comparison.csv and test_predictions.csv")
    print(" - ROC, PR, confusion-matrix and calibration plots")
    print(" - shap/ (global and local SHAP, if SHAP runs successfully)")
    print(" - run_notes.txt and data_availability_notes.txt")
    print("\nRead data_availability_notes.txt and run_notes.txt before writing paper results.")


if __name__ == "__main__":
    main()
