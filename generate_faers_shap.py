
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "output" / "FAERS_ML_Dataset.parquet"
OUT_DIR = ROOT / "research_results" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_MAX = 150_000
TEST_MAX = 75_000
EXPLAIN_MAX = 500

NUMERIC = [
    "AGE_YEARS", "WT_KG", "DRUG_COUNT",
    "N_PRIMARY_SUSPECT_DRUGS",
    "N_SECONDARY_SUSPECT_DRUGS"
]
CATEGORICAL = [
    "SEX", "COUNTRY", "REPORTER_COUNTRY", "OCCP_COD"
]
TARGET = "IS_FATAL"

def stratified_sample(frame, n):
    if len(frame) <= n:
        return frame.copy()
    return resample(
        frame,
        replace=False,
        n_samples=n,
        random_state=RANDOM_STATE,
        stratify=frame[TARGET]
    )

def main():
    print("Loading existing FAERS dataset...")

    required = [
        "SPLIT", "IS_FATAL", "REPORT_DATE"
    ] + NUMERIC + CATEGORICAL

    df = pd.read_parquet(DATA_PATH, columns=required)

    df["REPORT_DATE"] = pd.to_datetime(
        df["REPORT_DATE"], errors="coerce"
    )
    df[TARGET] = pd.to_numeric(
        df[TARGET], errors="coerce"
    )

    df = df[df[TARGET].isin([0, 1])].copy()

    dev = df[df["SPLIT"].astype(str).str.upper() == "DEV"]
    test = df[df["SPLIT"].astype(str).str.upper() == "TEST"]

    train_pool = dev[
        dev["REPORT_DATE"] <= pd.Timestamp("2024-02-08")
    ]
    test_pool = test[
        test["REPORT_DATE"] >= pd.Timestamp("2024-11-07")
    ]

    train_df = stratified_sample(train_pool, TRAIN_MAX)
    test_df = stratified_sample(test_pool, TEST_MAX)

    features = NUMERIC + CATEGORICAL
    X_train = train_df[features]
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[features]
    y_test = test_df[TARGET].astype(int)

    print("Training rows:", len(train_df))
    print("Test rows:", len(test_df))

    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=20,
            max_categories=50
        )
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
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])

    print("Preprocessing training data...")
    X_train_transformed = preprocess.fit_transform(X_train)

    print("Preprocessing held-out test data...")
    X_test_transformed = preprocess.transform(X_test)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    feature_names = preprocess.get_feature_names_out()
    feature_names = [
        name.replace("num__", "").replace("cat__", "")
        for name in feature_names
    ]

    print("Fitting original LightGBM configuration...")

    model = LGBMClassifier(
        n_estimators=140,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.08,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
        random_state=RANDOM_STATE
    )

    model.fit(X_train_transformed, y_train)

    print("Selecting held-out reports for SHAP...")
    rng = np.random.RandomState(RANDOM_STATE)
    count = min(EXPLAIN_MAX, len(X_test_transformed))
    indices = rng.choice(
        len(X_test_transformed),
        size=count,
        replace=False
    )

    X_explain = X_test_transformed[indices]

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_explain)

    values = shap_values.values

    if values.ndim == 3:
        values = values[:, :, 1]

    explanation = shap.Explanation(
        values=values,
        base_values=shap_values.base_values,
        data=X_explain,
        feature_names=feature_names
    )

    print("Generating global feature importance...")
    plt.figure()
    shap.plots.bar(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "global_shap_feature_importance.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()

    print("Generating SHAP summary plot...")
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "global_shap_summary.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()

    print("Generating individual waterfall plot...")
    plt.figure()
    shap.plots.waterfall(explanation[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(
        OUT_DIR / "local_shap_waterfall.png",
        dpi=180,
        bbox_inches="tight"
    )
    plt.close()

    with open(OUT_DIR / "shap_status.txt", "w", encoding="utf-8") as f:
        f.write("SHAP analysis completed successfully.\n")
        f.write("Model: LightGBM\n")
        f.write("Training sample: " + str(len(train_df)) + "\n")
        f.write("Explained held-out reports: " + str(count) + "\n")
        f.write("Random state: 42\n")
        f.write("Features: " + str(len(feature_names)) + "\n")

    print("\nSHAP ANALYSIS COMPLETED")
    print("Results saved in:", OUT_DIR)
    for file in sorted(OUT_DIR.iterdir()):
        print("-", file.name)

if __name__ == "__main__":
    main()
