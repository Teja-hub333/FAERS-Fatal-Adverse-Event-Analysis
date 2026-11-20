from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "output" / "FAERS_ML_Dataset.parquet"
RESULTS = ROOT / "research_results"
EDA = RESULTS / "eda"
SHAP = RESULTS / "shap"

st.set_page_config(
    page_title="FAERS | Fatal Adverse Event Analytics",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional clinical-research styling
st.markdown("""

<style>
    .stApp {
        background: #f4f7fb;
        color: #172b4d;
    }

    [data-testid="stSidebar"] {
        background: #102a43;
    }

    [data-testid="stSidebar"] * {
        color: #f4f7fb !important;
    }

    .hero {
        padding: 1.35rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(110deg,#123b5d,#176b87);
        color: white;
        margin-bottom: 1rem;
    }
    /* Make metric labels clearly visible */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] label {
    color: #52657a !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Make metric numbers visible */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #123b5d !important;
    opacity: 1 !important;
}

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        color: #d8eaf2;
        margin: .35rem 0 0 0;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2eaf2;
        padding: 15px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(16,42,67,.04);
    }

    /* Metric labels */
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p {
        color: #52657a !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }

    /* Metric values */
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] div {
        color: #17324d !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Metric percentage and delta */
    div[data-testid="stMetricDelta"] {
        color: #168653 !important;
    }

    .note {
        background: #eaf2f8;
        border-left: 4px solid #247ba0;
        padding: .8rem 1rem;
        border-radius: 6px;
        color: #243b53;
    }

    .warning {
        background: #fff6e5;
        border-left: 4px solid #d69e2e;
        padding: .8rem 1rem;
        border-radius: 6px;
        color: #744210;
    }

    h2, h3 {
        color: #123b5d;
    }

    footer {
        visibility: hidden;
    }
</style>

""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Reading FAERS dataset summary…")
def load_summary(path_str, modified):
    # Read only required columns; the source file remains unchanged.
    cols = ["IS_FATAL"]
    parquet_cols = pd.read_parquet(path_str, engine="pyarrow").columns.tolist()
    wanted = [c for c in [
        "IS_FATAL", "REPORT_YEAR", "REPORT_QUARTER", "SEX",
        "COUNTRY", "REPORTER_COUNTRY", "OCCP_COD", "SPLIT",
        "AGE_YEARS", "DRUG_COUNT", "N_PRIMARY_SUSPECT_DRUGS",
        "N_SECONDARY_SUSPECT_DRUGS"
    ] if c in parquet_cols]
    df = pd.read_parquet(path_str, columns=wanted)
    df["IS_FATAL"] = pd.to_numeric(df["IS_FATAL"], errors="coerce")
    return df

def load_csv(filename):
    path = RESULTS / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def fmt(n):
    return f"{int(n):,}"

st.sidebar.markdown("## ✚ FAERS Analytics")
st.sidebar.caption("Fatal Adverse Drug Event Research")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Dataset Explorer", "Model Performance", "Explainability", "Methodology"],
)
st.sidebar.divider()
st.sidebar.caption("Data source")
st.sidebar.code("output/FAERS_ML_Dataset.parquet", language=None)
st.sidebar.caption("Research dashboard · Local deployment")

st.markdown("""
<div class="hero">
  <h1>FAERS Fatal Adverse Event Analytics</h1>
  <p>Data science and machine learning analysis of reported fatal and nonfatal adverse event outcomes</p>
</div>
""", unsafe_allow_html=True)

if not DATA_FILE.exists():
    st.error("FAERS dataset not found.")
    st.markdown(f"""
    **Expected file:** `{DATA_FILE}`

    Place your existing `FAERS_ML_Dataset.parquet` inside the `output` folder next to `app.py`.
    Do not rebuild the dataset just to launch this dashboard.
    """)
    st.stop()

file_mtime = DATA_FILE.stat().st_mtime
try:
    data = load_summary(str(DATA_FILE), file_mtime)
except Exception as exc:
    st.error(f"Could not read the Parquet dataset: {exc}")
    st.info("Install the packages in requirements.txt, including pyarrow.")
    st.stop()

known = data[data["IS_FATAL"].isin([0, 1])]
fatal = int((known["IS_FATAL"] == 1).sum())
nonfatal = int((known["IS_FATAL"] == 0).sum())
unknown = int(data["IS_FATAL"].isna().sum())
total = len(data)
fatal_pct = (fatal / len(known) * 100) if len(known) else 0

if page == "Overview":
    st.subheader("Research overview")
    st.caption("Report-level statistics from the prepared FAERS machine-learning dataset.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total reports", fmt(total))
    c2.metric("Fatal reports", fmt(fatal), f"{fatal_pct:.1f}% of known outcomes")
    c3.metric("Nonfatal reports", fmt(nonfatal))
    c4.metric("Unknown outcomes", fmt(unknown))

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### Reported outcome distribution")
        outcome_df = pd.DataFrame({
            "Outcome": ["Fatal", "Nonfatal", "Unknown"],
            "Reports": [fatal, nonfatal, unknown]
        })
        fig = px.pie(outcome_df, names="Outcome", values="Reports", hole=.58,
                     color="Outcome",
                     color_discrete_map={"Fatal":"#c65b5b","Nonfatal":"#2b7a78","Unknown":"#9aa9b8"})
        fig.update_layout(margin=dict(t=15,b=10,l=10,r=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### Report year distribution")
        if "REPORT_YEAR" in data:
            yearly = data.dropna(subset=["REPORT_YEAR"]).groupby("REPORT_YEAR").size().reset_index(name="Reports")
            yearly["REPORT_YEAR"] = yearly["REPORT_YEAR"].astype(str)
            fig = px.bar(yearly, x="REPORT_YEAR", y="Reports", color_discrete_sequence=["#247ba0"])
            fig.update_layout(xaxis_title="Report year", yaxis_title="Reports",
                              margin=dict(t=15,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("REPORT_YEAR is not available in this dataset.")

    st.markdown("### Dataset coverage")
    split_col, age_col = st.columns(2)
    with split_col:
        if "SPLIT" in data:
            split = data["SPLIT"].fillna("Unknown").value_counts().rename_axis("Partition").reset_index(name="Reports")
            fig = px.bar(split, x="Partition", y="Reports", color_discrete_sequence=["#3b82a0"])
            fig.update_layout(margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
    with age_col:
        if "AGE_YEARS" in data:
            ages = pd.to_numeric(data["AGE_YEARS"], errors="coerce").dropna()
            ages = ages[(ages >= 0) & (ages <= 120)]
            if len(ages):
                fig = px.histogram(ages, nbins=30, labels={"value":"Age (years)"})
                fig.update_traces(marker_color="#4d8ca8")
                fig.update_layout(title="Age distribution", yaxis_title="Reports", margin=dict(t=40,b=10))
                st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="warning"><b>Interpretation:</b> “Fatal” means a death outcome code (DE) was reported in FAERS. It does not establish that a drug caused the death or estimate an individual patient’s mortality risk.</div>', unsafe_allow_html=True)

elif page == "Dataset Explorer":
    st.subheader("Dataset explorer")
    st.caption("Explore the prepared report-level data without changing the source Parquet file.")
    options = [c for c in ["REPORT_YEAR","SEX","COUNTRY","REPORTER_COUNTRY","OCCP_COD","SPLIT"] if c in data]
    if options:
        chosen = st.selectbox("Group reports by", options)
        view = data.copy()
        view[chosen] = view[chosen].fillna("Unknown").astype(str)
        grouped = view.groupby([chosen, "IS_FATAL"], dropna=False).size().reset_index(name="Reports")
        grouped["Outcome"] = grouped["IS_FATAL"].map({0:"Nonfatal",1:"Fatal"}).fillna("Unknown")
        grouped = grouped.drop(columns=["IS_FATAL"])
        totals = grouped.groupby(chosen)["Reports"].sum().sort_values(ascending=False)
        top_n = st.slider("Number of categories to show", 5, 30, 15)
        keep = totals.head(top_n).index
        grouped = grouped[grouped[chosen].isin(keep)]
        fig = px.bar(grouped, x=chosen, y="Reports", color="Outcome", barmode="group",
                     color_discrete_map={"Fatal":"#c65b5b","Nonfatal":"#2b7a78","Unknown":"#9aa9b8"})
        fig.update_layout(xaxis_title=chosen.replace("_"," ").title(), margin=dict(t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(grouped.sort_values("Reports", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No supported grouping columns were found.")

    st.markdown("### Numeric feature summary")
    numeric = [c for c in ["AGE_YEARS","DRUG_COUNT","N_PRIMARY_SUSPECT_DRUGS","N_SECONDARY_SUSPECT_DRUGS"] if c in data]
    if numeric:
        st.dataframe(data[numeric].describe().T, use_container_width=True)
    st.caption("This explorer uses aggregated views; it does not expose personal or patient-level records.")

elif page == "Model Performance":
    st.subheader("Model performance")
    st.caption("Evaluation on the chronologically held-out test sample.")

    MODEL_RESULTS = RESULTS / "model_evaluation"
    metrics_path = MODEL_RESULTS / "model_comparison.csv"

    if not metrics_path.exists():
        st.warning(
            "Model evaluation results were not found. Run "
            "`train_faers_models_fast.py` first."
        )
    else:
        try:
            metrics = pd.read_csv(metrics_path)
        except Exception as exc:
            st.error(f"Could not read model metrics: {exc}")
            metrics = pd.DataFrame()

        if not metrics.empty:
            # The fast training script saves lowercase column names.
            required_metric_cols = [
                "model", "roc_auc", "f1_fatal",
                "precision_fatal", "recall_fatal"
            ]
            missing_metrics = [c for c in required_metric_cols if c not in metrics.columns]
            if missing_metrics:
                st.error(
                    "The model_comparison.csv format is missing columns: "
                    + ", ".join(missing_metrics)
                )
            else:
                st.markdown("#### Model comparison")
                best_auc = metrics.loc[metrics["roc_auc"].idxmax()]
                best_f1 = metrics.loc[metrics["f1_fatal"].idxmax()]

                k1, k2, k3 = st.columns(3)
                k1.metric("Models evaluated", str(len(metrics)))
                k2.metric("Best ROC-AUC", f"{best_auc['roc_auc']:.4f}",
                          str(best_auc["model"]))
                k3.metric("Best F1-score", f"{best_f1['f1_fatal']:.4f}",
                          str(best_f1["model"]))

                display_cols = [
                    c for c in [
                        "model", "test_rows", "threshold_selected_on_validation",
                        "accuracy", "precision_fatal", "recall_fatal",
                        "f1_fatal", "roc_auc", "average_precision",
                        "tn", "fp", "fn", "tp"
                    ] if c in metrics.columns
                ]
                display_metrics = metrics[display_cols].copy()
                display_metrics = display_metrics.rename(columns={
                    "model": "Model",
                    "test_rows": "Test reports",
                    "threshold_selected_on_validation": "Validation threshold",
                    "accuracy": "Accuracy",
                    "precision_fatal": "Fatal precision",
                    "recall_fatal": "Fatal recall",
                    "f1_fatal": "Fatal F1",
                    "roc_auc": "ROC-AUC",
                    "average_precision": "Average precision",
                    "tn": "True negatives",
                    "fp": "False positives",
                    "fn": "False negatives",
                    "tp": "True positives"
                })
                st.dataframe(
                    display_metrics.style.format({
                        c: "{:.4f}" for c in [
                            "Validation threshold", "Accuracy", "Fatal precision",
                            "Fatal recall", "Fatal F1", "ROC-AUC", "Average precision"
                        ] if c in display_metrics.columns
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("#### Compare model scores")
                score_cols = [
                    c for c in ["roc_auc", "f1_fatal", "precision_fatal", "recall_fatal"]
                    if c in metrics.columns
                ]
                score_long = metrics.melt(
                    id_vars="model", value_vars=score_cols,
                    var_name="Metric", value_name="Score"
                )
                score_long["Metric"] = score_long["Metric"].map({
                    "roc_auc": "ROC-AUC",
                    "f1_fatal": "Fatal F1-score",
                    "precision_fatal": "Fatal precision",
                    "recall_fatal": "Fatal recall"
                })
                fig = px.bar(
                    score_long, x="model", y="Score", color="Metric",
                    barmode="group", range_y=[0, 1],
                    color_discrete_sequence=["#1f6f8b", "#4d8ca8", "#82b4c7", "#d69e2e"]
                )
                fig.update_layout(
                    xaxis_title="Model", yaxis_title="Score",
                    legend_title_text="", margin=dict(t=20, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### ROC curve")
                roc_path = MODEL_RESULTS / "roc_curves_all_models.png"
                if roc_path.exists():
                    st.image(str(roc_path), use_container_width=True)
                else:
                    st.info("Combined ROC curve image has not been generated.")

                st.markdown("#### Confusion matrices")
                st.caption(
                    "Rows show actual classes; columns show predicted classes. "
                    "Class 0 = nonfatal, class 1 = fatal."
                )
                model_files = {
                    "Logistic Regression": "logistic_regression_confusion_matrix.png",
                    "Random Forest": "random_forest_confusion_matrix.png",
                    "XGBoost": "xgboost_confusion_matrix.png",
                    "LightGBM": "lightgbm_confusion_matrix.png"
                }
                available_matrices = [
                    (name, MODEL_RESULTS / filename)
                    for name, filename in model_files.items()
                    if (MODEL_RESULTS / filename).exists()
                ]
                if available_matrices:
                    tabs = st.tabs([name for name, _ in available_matrices])
                    for tab, (name, image_path) in zip(tabs, available_matrices):
                        with tab:
                            st.image(str(image_path), caption=f"{name} confusion matrix",
                                     use_container_width=True)
                else:
                    st.info("No confusion matrix images were found in model_evaluation.")

                split_path = MODEL_RESULTS / "sample_split_summary.csv"
                if split_path.exists():
                    with st.expander("Evaluation sample and date ranges"):
                        split_info = pd.read_csv(split_path)
                        st.dataframe(split_info, use_container_width=True, hide_index=True)

                st.warning(
                    "Research note: These are sample-based results (up to 150,000 "
                    "training, 40,000 validation and 75,000 test reports), not "
                    "full-dataset training results. The classification threshold "
                    "was selected using validation data. Interpret the scores as "
                    "model discrimination on this held-out sample, not as proof "
                    "of drug causation or individual mortality risk."
                )
                st.caption(
                    "Metrics are read directly from the training script's saved "
                    "CSV and plots. No evaluation values are fabricated."
                )

elif page == "Explainability":
    st.subheader("Model explainability")
    st.write(
        "SHAP (SHapley Additive exPlanations) shows how input features contribute "
        "to the LightGBM model output. These explanations describe model behaviour; "
        "they do not establish that a drug or feature caused a fatal outcome."
    )

    status = SHAP / "shap_status.txt"
    if status.exists():
        st.success("SHAP analysis completed. The following plots were generated from the held-out reports.")
        with st.expander("SHAP analysis details", expanded=False):
            st.code(status.read_text(encoding="utf-8"), language=None)
    else:
        st.warning(
            "SHAP status file was not found. Check that shap_status.txt and the PNG "
            "files are inside research_results/shap next to app.py."
        )

    importance = SHAP / "global_shap_feature_importance.png"
    summary = SHAP / "global_shap_summary.png"
    waterfall = SHAP / "local_shap_waterfall.png"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Global feature importance")
        if importance.exists():
            st.image(str(importance), use_container_width=True)
            st.caption("Ranks features by their average absolute SHAP contribution.")
        else:
            st.error(f"Missing file: {importance.name}")
    with col2:
        st.markdown("#### 2. SHAP summary plot")
        if summary.exists():
            st.image(str(summary), use_container_width=True)
            st.caption("Shows the direction and distribution of feature contributions.")
        else:
            st.error(f"Missing file: {summary.name}")

    st.markdown("#### 3. Local SHAP waterfall explanation")
    if waterfall.exists():
        st.image(str(waterfall), use_container_width=True)
        st.caption("Explains how individual feature contributions move one report's model output away from the baseline.")
    else:
        st.error(f"Missing file: {waterfall.name}")

    st.info(
        "Interpretation note: SHAP values are contributions to the model output, "
        "not percentages or proof of clinical risk. The global plots summarize the "
        "500 held-out reports explained in this SHAP run."
    )

elif page == "Methodology":
    st.subheader("Study methodology")
    st.markdown("""
    **Outcome definition**
    - `IS_FATAL = 1`: a report includes FAERS outcome code `DE`.
    - `IS_FATAL = 0`: a known non-death outcome is reported and no `DE` is present.
    - Missing or unknown outcomes remain unknown and are excluded from supervised model fitting.

    **Predictors in the supplied training script**
    - Numeric: age, weight, drug count, primary suspect drug count and secondary suspect drug count, where present.
    - Categorical: sex, country, reporter country and occupation code, where present.
    - The script excludes the target, outcome codes, report identifiers, report dates and split metadata from predictors.

    **Evaluation design**
    - Chronological training and validation within the development period.
    - The optimized workflow uses stratified samples from the chronological train, validation and held-out test periods.
    - Classification thresholds are selected on validation data and then evaluated on the later held-out TEST sample.
    - Report the sample sizes and date ranges alongside all metrics.

    **Models**
    - Logistic Regression
    - Random Forest
    - XGBoost (if installed)
    - LightGBM (if installed)

    **Limitations**
    FAERS is a spontaneous-reporting database. Reports may be incomplete, duplicated, selectively submitted or affected by reporting bias. The database has no exposure denominator, so these analyses cannot calculate incidence or prove causation.
    """)
    notes = RESULTS / "data_availability_notes.txt"
    if notes.exists():
        st.markdown("#### Data availability notes from training workflow")
        st.code(notes.read_text(encoding="utf-8"))

st.divider()
st.caption("FAERS research dashboard · For research and educational use · Not a clinical decision-support system")
