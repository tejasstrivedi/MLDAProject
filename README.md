# Day-Ahead Hourly Load Forecasting for Two 11 kV Feeders

This beginner academic project compares Random Forest and XGBoost for forecasting the next 24 hourly feeder-load values.

## Project files

```text
feeder_2023_2025.xlsx    Original synthetic dataset
project.ipynb            Complete analysis and model-building notebook
external_data/           Rajkot weather and Gujarat holiday inputs
dashboard/app.py         Dashboard, to be completed near the end
models/                  Saved final model
README.md                Project description and instructions
requirements.txt         Required Python packages
```

The work will be developed gradually in `project.ipynb`. The original Excel file will remain unchanged.

## External data

- Hourly Rajkot temperature and relative humidity: Open-Meteo Historical Forecast API, 2023–2025, Asia/Kolkata timezone.
- Gujarat public-holiday calendar: Python Holidays project, India subdivision `GJ`, 2023–2025.
- `rajkot_synthetic_weather_2026.csv`: synthetic future weather inputs based on Rajkot's 2025 hourly pattern with reproducible smooth random variation.
- `gujarat_public_holidays_2026.csv`: Gujarat holiday dates used for the future-forecast demonstration.

The weather series is historical forecast-model output and is not restricted to forecasts issued at one fixed 24-hour lead time. This limitation should be considered when interpreting day-ahead test performance.

The 2026 weather file is synthetic and must not be described as observed or official forecast weather. It is reserved for the final future-forecast demonstration and is not included in model training or 2025 testing.

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
