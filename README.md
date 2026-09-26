# Day-Ahead Hourly Load Forecasting for Two 11 kV Feeders

This repository is the project scaffold for comparing Random Forest and XGBoost models for 24-hour-ahead feeder load forecasting.

## Status

The project structure is initialized. Data preparation, analysis, modelling, evaluation, and the dashboard have not yet been implemented.

## Local inputs

- `feeder_2023_2025.xlsx` — synthetic hourly feeder dataset for 2023–2025
- `readme.txt.txt` — course project guidelines

The Excel dataset is excluded from Git by default. Keep source data private/local unless it is intentionally approved for publication.

## Structure

```text
config/                 Project configuration
data/processed/         Generated modelling datasets
dashboard/              Streamlit dashboard
models/                 Serialized trained models
notebooks/              Numbered analysis notebooks
reports/figures/         Generated report figures
src/                     Reusable Python package
tests/                   Automated tests
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Planned workflow

1. Data audit and exploratory analysis
2. Leakage-safe feature engineering
3. Temporal validation and weekly persistence baseline
4. Random Forest and XGBoost tuning
5. Rolling day-ahead evaluation on 2025
6. Model comparison, explainability, and dashboard
