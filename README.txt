# FAERS Professional Research Dashboard

This Streamlit website displays statistics from the existing FAERS ML Parquet dataset and, when available, model evaluation and SHAP outputs created by the supplied training script.

## Project layout

Keep this layout:

```text
faers_professional_dashboard/
  app.py
  requirements.txt
  train_faers_models_updated.py
  output/
    FAERS_ML_Dataset.parquet
  research_results/
    model_comparison.csv          # created by completed training run
    model_metric_comparison.png   # created by completed training run
    shap/
      shap_status.txt
      global_shap_feature_importance.png
      global_shap_summary.png
      local_shap_waterfall.png
    eda/
```

## Setup on Windows

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy your existing `output\FAERS_ML_Dataset.parquet` into the `output` folder. Do not rebuild the FAERS database just to launch the dashboard.

Copy the supplied `train_faers_models_updated.py` into this project folder if you want the script and website together. The dashboard does not automatically start model training; it reads the result files only after training has completed.

## Start the website

```powershell
streamlit run app.py
```

Streamlit will print a local address, normally `http://localhost:8501`. Open it in your browser.

## Important

- This is a local dashboard; it is not publicly deployed.
- Model results pages remain empty until the training script has successfully generated the corresponding files.
- The dashboard never invents model scores.
- `IS_FATAL=1` means a death outcome code was reported in FAERS. It is not proof that a drug caused death or an estimate of an individual's mortality risk.
- FAERS is a spontaneous-reporting database and is subject to missingness, under-reporting and reporting bias.
