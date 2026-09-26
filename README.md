# Day-Ahead Hourly Load Forecasting for Two 11 kV Feeders

This beginner academic project compares Random Forest and XGBoost for forecasting the next 24 hourly feeder-load values.

## Project files

```text
feeder_2023_2025.xlsx    Original synthetic dataset
project.ipynb            Complete analysis and model-building notebook
dashboard/app.py         Dashboard, to be completed near the end
models/                  Saved final model
README.md                Project description and instructions
requirements.txt         Required Python packages
```

The work will be developed gradually in `project.ipynb`. The original Excel file will remain unchanged.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Workflow

1. Understand and check the dataset
2. Perform basic exploratory analysis
3. Create calendar and historical-load features
4. Split the data chronologically
5. Build a weekly baseline
6. Train Random Forest and XGBoost
7. Compare MAE and RMSE
8. Save the best model and create a simple dashboard
