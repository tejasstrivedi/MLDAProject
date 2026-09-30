"""Streamlit dashboard for the feeder day-ahead forecasting project."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "final_load_forecasting_bundle.joblib"
FEEDER_DATA_PATH = PROJECT_ROOT / "feeder_2023_2025.xlsx"
WEATHER_2026_PATH = (
    PROJECT_ROOT / "external_data" / "rajkot_synthetic_weather_2026.csv"
)
HOLIDAYS_2026_PATH = (
    PROJECT_ROOT / "external_data" / "gujarat_public_holidays_2026.csv"
)


st.set_page_config(
    page_title="Rajkot Feeder Load Forecast",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource
def load_model_bundle(model_path: Path) -> dict:
    """Load the trained models and their supporting metadata once."""
    return joblib.load(model_path)


@st.cache_data
def load_forecast_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical feeder readings and merge the 2026 external inputs."""
    historical_data = pd.read_excel(
        FEEDER_DATA_PATH,
        sheet_name="Hourly_Data",
        usecols=["Date", "Hour Interval", "Feeder Name", "MW(I)"],
    )
    historical_data["timestamp"] = pd.to_datetime(
        historical_data["Date"].astype(str)
        + " "
        + historical_data["Hour Interval"].astype(str)
    )
    historical_data = historical_data.sort_values(
        ["Feeder Name", "timestamp"]
    ).reset_index(drop=True)

    weather_data = pd.read_csv(WEATHER_2026_PATH, parse_dates=["timestamp"])
    holiday_data = pd.read_csv(HOLIDAYS_2026_PATH, parse_dates=["date"])
    weather_data["date"] = weather_data["timestamp"].dt.normalize()
    future_external_data = weather_data.merge(
        holiday_data,
        on="date",
        how="left",
        validate="many_to_one",
    )
    future_external_data["is_holiday"] = (
        future_external_data["holiday_name"].notna().astype(int)
    )
    return historical_data, future_external_data


def build_forecast_features(
    feeder_name: str,
    forecast_date: pd.Timestamp,
    historical_data: pd.DataFrame,
    future_external_data: pd.DataFrame,
    feeder_mapping: dict,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leakage-safe features for the selected forecast date."""
    forecast_start = pd.Timestamp(forecast_date).normalize()
    forecast_end = forecast_start + pd.Timedelta(days=1)
    forecast_rows = future_external_data.loc[
        (future_external_data["timestamp"] >= forecast_start)
        & (future_external_data["timestamp"] < forecast_end)
    ].copy()
    forecast_rows["Feeder Name"] = feeder_name

    forecast_rows["hour"] = forecast_rows["timestamp"].dt.hour
    forecast_rows["day_of_week_number"] = (
        forecast_rows["timestamp"].dt.dayofweek
    )
    forecast_rows["month"] = forecast_rows["timestamp"].dt.month
    forecast_rows["is_weekend"] = (
        forecast_rows["day_of_week_number"] >= 5
    ).astype(int)
    forecast_rows["feeder_id"] = feeder_mapping[feeder_name]

    forecast_rows["hour_sin"] = np.sin(
        2 * np.pi * forecast_rows["hour"] / 24
    )
    forecast_rows["hour_cos"] = np.cos(
        2 * np.pi * forecast_rows["hour"] / 24
    )
    forecast_rows["day_of_week_sin"] = np.sin(
        2 * np.pi * forecast_rows["day_of_week_number"] / 7
    )
    forecast_rows["day_of_week_cos"] = np.cos(
        2 * np.pi * forecast_rows["day_of_week_number"] / 7
    )
    forecast_rows["month_sin"] = np.sin(
        2 * np.pi * (forecast_rows["month"] - 1) / 12
    )
    forecast_rows["month_cos"] = np.cos(
        2 * np.pi * (forecast_rows["month"] - 1) / 12
    )

    feeder_history = (
        historical_data.loc[historical_data["Feeder Name"] == feeder_name]
        .set_index("timestamp")["MW(I)"]
        .sort_index()
    )

    for lag_hours in (24, 48, 168):
        forecast_rows[f"load_lag_{lag_hours}"] = forecast_rows[
            "timestamp"
        ].map(lambda timestamp: feeder_history.get(
            timestamp - pd.Timedelta(hours=lag_hours), np.nan
        ))

    def rolling_value(timestamp: pd.Timestamp, window: int, method: str) -> float:
        window_end = timestamp - pd.Timedelta(hours=24)
        window_start = window_end - pd.Timedelta(hours=window - 1)
        required_hours = pd.date_range(window_start, window_end, freq="h")
        values = feeder_history.reindex(required_hours)
        if values.isna().any():
            return np.nan
        return values.mean() if method == "mean" else values.std()

    forecast_rows["rolling_mean_24"] = forecast_rows["timestamp"].map(
        lambda timestamp: rolling_value(timestamp, 24, "mean")
    )
    forecast_rows["rolling_std_24"] = forecast_rows["timestamp"].map(
        lambda timestamp: rolling_value(timestamp, 24, "std")
    )
    forecast_rows["rolling_mean_168"] = forecast_rows["timestamp"].map(
        lambda timestamp: rolling_value(timestamp, 168, "mean")
    )

    model_input = forecast_rows[feature_columns].copy()
    if model_input.isna().any().any():
        missing_columns = model_input.columns[model_input.isna().any()].tolist()
        raise ValueError(f"Missing forecast values in: {missing_columns}")
    return forecast_rows, model_input


def create_load_template(
    feeder_names: list[str], required_date: pd.Timestamp
) -> pd.DataFrame:
    """Create 24 blank hourly readings for every feeder on one fixed date."""
    timestamps = pd.date_range(
        pd.Timestamp(required_date).normalize(), periods=24, freq="h"
    )
    template = pd.MultiIndex.from_product(
        [timestamps, feeder_names],
        names=["timestamp", "Feeder Name"],
    ).to_frame(index=False)
    template["MW(I)"] = np.nan
    return template


def validate_uploaded_load(
    uploaded_data: pd.DataFrame,
    required_date: pd.Timestamp,
    feeder_names: list[str],
) -> pd.DataFrame:
    """Validate one complete day of actual hourly load for both feeders."""
    required_columns = ["timestamp", "Feeder Name", "MW(I)"]
    missing_columns = set(required_columns).difference(uploaded_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    clean_data = uploaded_data[required_columns].copy()
    clean_data["timestamp"] = pd.to_datetime(
        clean_data["timestamp"], errors="coerce"
    )
    clean_data["MW(I)"] = pd.to_numeric(clean_data["MW(I)"], errors="coerce")

    if clean_data[required_columns].isna().any().any():
        raise ValueError("Timestamp, feeder name and MW(I) must be filled in every row.")
    if clean_data.duplicated(["timestamp", "Feeder Name"]).any():
        raise ValueError("Duplicate feeder-timestamp rows were found.")
    if set(clean_data["Feeder Name"]) != set(feeder_names):
        raise ValueError("The file must contain exactly the two recognized feeder names.")
    if (clean_data["MW(I)"] < 0).any():
        raise ValueError("MW(I) cannot contain negative values.")

    expected_template = create_load_template(feeder_names, required_date)
    expected_keys = set(map(tuple, expected_template[["timestamp", "Feeder Name"]].to_numpy()))
    uploaded_keys = set(map(tuple, clean_data[["timestamp", "Feeder Name"]].to_numpy()))
    if uploaded_keys != expected_keys:
        raise ValueError(
            f"The file must contain all 24 hours for both feeders on "
            f"{pd.Timestamp(required_date).date()} only."
        )

    return clean_data.sort_values(["Feeder Name", "timestamp"]).reset_index(drop=True)


st.title("⚡ Day-Ahead Hourly Load Forecasting")
st.caption("Two 11 kV feeders in Rajkot city | PGVCL demonstration project")

if not MODEL_BUNDLE_PATH.exists():
    st.error(
        "The saved model bundle was not found. Run the model-saving cells "
        "near the end of project.ipynb before starting the dashboard."
    )
    st.stop()

try:
    model_bundle = load_model_bundle(MODEL_BUNDLE_PATH)
except Exception as error:
    st.error(f"The saved model bundle could not be loaded: {error}")
    st.stop()

required_bundle_keys = {
    "models",
    "recommended_model_name",
    "model_ranking",
    "dashboard_model_options",
    "feeder_mapping",
}
missing_bundle_keys = required_bundle_keys.difference(model_bundle)

if missing_bundle_keys:
    st.error(
        "The model bundle is incomplete. Re-run the revised model-saving "
        f"cell. Missing items: {sorted(missing_bundle_keys)}"
    )
    st.stop()

model_ranking = pd.DataFrame(model_bundle["model_ranking"])
recommended_model_name = model_bundle["recommended_model_name"]
feeder_names = sorted(model_bundle["feeder_mapping"])
model_options = model_bundle["dashboard_model_options"]

recommended_label = model_ranking.loc[
    model_ranking["model"] == recommended_model_name,
    "dashboard_label",
].iloc[0]

st.sidebar.header("Forecast selections")
selected_feeder = st.sidebar.selectbox(
    "11 kV feeder",
    options=feeder_names,
)
selected_model_label = st.sidebar.selectbox(
    "Forecasting method",
    options=model_options,
    index=model_options.index(recommended_label),
    help="Ranks are based on MAE for the fixed 2024 validation period.",
)

if selected_model_label == "Same-hour previous week":
    selected_model_name = "Same-hour previous week"
    selected_model_rank = "Benchmark"
else:
    selected_row = model_ranking.loc[
        model_ranking["dashboard_label"] == selected_model_label
    ].iloc[0]
    selected_model_name = selected_row["model"]
    selected_model_rank = f"Rank {int(selected_row['performance_rank'])}"

st.subheader("Current selection")
selection_column_1, selection_column_2, selection_column_3 = st.columns(3)
selection_column_1.metric("Feeder", selected_feeder)
selection_column_2.metric("Method", selected_model_name)
selection_column_3.metric("Validation position", selected_model_rank)

if selected_model_name == recommended_model_name:
    st.success(
        f"{recommended_model_name} is the recommended model because it achieved "
        "the lowest MAE on the fixed 2024 validation period."
    )
elif selected_model_name == "Same-hour previous week":
    st.info(
        "The weekly baseline repeats the load recorded at the same hour one week "
        "earlier. It provides a simple reference for the trained models."
    )
else:
    st.info(
        "This model is available for comparison. Rank 1 remains the recommended "
        "default based on validation performance."
    )

with st.expander("View validation-based model ranking"):
    st.dataframe(
        model_ranking[
            ["performance_rank", "model", "MAE", "RMSE", "R2"]
        ].rename(
            columns={
                "performance_rank": "Rank",
                "model": "Model",
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.subheader("Historical performance on the 2025 test period")
st.caption(
    "These values come from the untouched 2025 data. Lower MAE and RMSE are "
    "better; higher R² is better. They are separate from the future 2026 forecast."
)

test_results = pd.DataFrame(model_bundle["test_results"])
test_result_name = (
    "Weekly baseline"
    if selected_model_name == "Same-hour previous week"
    else selected_model_name
)
selected_test_result = test_results.loc[
    test_results["model"] == test_result_name
].iloc[0]

test_column_1, test_column_2, test_column_3, test_column_4 = st.columns(4)
test_column_1.metric("Test MAE", f"{selected_test_result['MAE']:.4f} MW")
test_column_2.metric("Test RMSE", f"{selected_test_result['RMSE']:.4f} MW")
test_column_3.metric("Test R²", f"{selected_test_result['R2']:.4f}")
test_column_4.metric(
    "MAE improvement over baseline",
    f"{selected_test_result['MAE improvement over baseline (%)']:.2f}%",
)

test_chart_data = (
    test_results.set_index("model")[["MAE", "RMSE"]]
    .sort_values("MAE", ascending=False)
)
st.bar_chart(
    test_chart_data,
    x_label="Forecasting method",
    y_label="Error in MW (lower is better)",
    height=400,
)

with st.expander("View complete 2025 test-results table"):
    st.dataframe(
        test_results.rename(
            columns={
                "model": "Model",
                "MAE improvement over baseline (%)": "MAE improvement (%)",
                "RMSE improvement over baseline (%)": "RMSE improvement (%)",
            }
        ).round(4),
        hide_index=True,
        width="stretch",
    )

st.divider()
required_data_paths = [
    FEEDER_DATA_PATH,
    WEATHER_2026_PATH,
    HOLIDAYS_2026_PATH,
]
missing_data_paths = [path for path in required_data_paths if not path.exists()]

if missing_data_paths:
    st.error(
        "Required project data could not be found: "
        + ", ".join(str(path) for path in missing_data_paths)
    )
    st.stop()

try:
    historical_data, future_external_data = load_forecast_source_data()
except Exception as error:
    st.error(f"The project data could not be loaded: {error}")
    st.stop()

if "uploaded_load_history" not in st.session_state:
    st.session_state.uploaded_load_history = pd.DataFrame(
        columns=["timestamp", "Feeder Name", "MW(I)"]
    )

uploaded_history = st.session_state.uploaded_load_history.copy()
combined_history = pd.concat(
    [
        historical_data[["timestamp", "Feeder Name", "MW(I)"]],
        uploaded_history,
    ],
    ignore_index=True,
).sort_values(["Feeder Name", "timestamp"])

next_upload_date = combined_history["timestamp"].max().normalize() + pd.Timedelta(days=1)
last_supported_date = pd.Timestamp("2026-12-31")

st.subheader("Update actual load history")
st.write(
    "Download the fixed-date template, enter the actual MW(I) readings for both "
    "feeders, and upload the completed CSV. A successful upload unlocks the next "
    "forecast date."
)

if next_upload_date <= last_supported_date:
    load_template = create_load_template(feeder_names, next_upload_date)
    template_csv = load_template.to_csv(index=False).encode("utf-8")

    upload_column, instruction_column = st.columns([1, 2])
    with upload_column:
        st.download_button(
            f"Download template for {next_upload_date.date()}",
            data=template_csv,
            file_name=f"actual_feeder_load_{next_upload_date.date()}.csv",
            mime="text/csv",
        )
    with instruction_column:
        st.caption(
            "Keep timestamp and feeder-name cells unchanged. Fill only the 48 "
            "MW(I) values: 24 hours for each feeder."
        )

    completed_upload = st.file_uploader(
        f"Upload completed readings for {next_upload_date.date()}",
        type=["csv"],
        key=f"load_upload_{next_upload_date.date()}",
    )

    if completed_upload is not None:
        if st.button("Validate and add actual readings", type="primary"):
            try:
                uploaded_data = pd.read_csv(completed_upload)
                validated_data = validate_uploaded_load(
                    uploaded_data,
                    required_date=next_upload_date,
                    feeder_names=feeder_names,
                )
                st.session_state.uploaded_load_history = pd.concat(
                    [st.session_state.uploaded_load_history, validated_data],
                    ignore_index=True,
                )
                st.success(
                    f"Actual readings for {next_upload_date.date()} were added. "
                    "The following forecast date is now available."
                )
                st.rerun()
            except Exception as error:
                st.error(f"Upload rejected: {error}")
else:
    st.success("Actual load history is complete through 31 December 2026.")

if not uploaded_history.empty:
    uploaded_dates = uploaded_history["timestamp"].dt.normalize().nunique()
    history_column_1, history_column_2 = st.columns([2, 1])
    history_column_1.success(
        f"Session history contains {uploaded_dates} uploaded day(s) and "
        f"{len(uploaded_history)} validated readings."
    )
    if history_column_2.button("Clear uploaded session data"):
        st.session_state.uploaded_load_history = pd.DataFrame(
            columns=["timestamp", "Feeder Name", "MW(I)"]
        )
        st.rerun()

first_forecast_date = pd.Timestamp("2026-01-01")
latest_available_forecast_date = min(next_upload_date, last_supported_date)
available_forecast_dates = pd.date_range(
    first_forecast_date,
    latest_available_forecast_date,
    freq="D",
)
selected_forecast_date = pd.Timestamp(
    st.sidebar.selectbox(
        "Forecast date",
        options=available_forecast_dates,
        index=len(available_forecast_dates) - 1,
        format_func=lambda value: pd.Timestamp(value).strftime("%d %B %Y"),
        help="A new date becomes available after the previous day's actual load is uploaded.",
    )
)

st.divider()
st.subheader(
    f"24-hour forecast for {selected_forecast_date.strftime('%d %B %Y')}"
)
st.caption(
    "Synthetic Rajkot weather and Gujarat holiday information are combined with "
    "all actual feeder readings available before the selected forecast day."
)

try:
    with st.spinner("Preparing the 24-hour forecast input..."):
        forecast_rows, model_input = build_forecast_features(
            feeder_name=selected_feeder,
            forecast_date=selected_forecast_date,
            historical_data=combined_history,
            future_external_data=future_external_data,
            feeder_mapping=model_bundle["feeder_mapping"],
            feature_columns=model_bundle["feature_columns"],
        )

        if selected_model_name == "Same-hour previous week":
            predicted_load = forecast_rows["load_lag_168"].to_numpy()
        else:
            predicted_load = model_bundle["models"][selected_model_name].predict(
                model_input
            )
except Exception as error:
    st.error(f"The forecast could not be generated: {error}")
    st.stop()

forecast_display = forecast_rows[
    ["timestamp", "temperature_c", "humidity_percent", "holiday_name"]
].copy()
forecast_display["Predicted Load (MW)"] = predicted_load
forecast_display["Hour"] = forecast_display["timestamp"].dt.strftime("%H:%M")

peak_position = int(np.argmax(predicted_load))
summary_column_1, summary_column_2, summary_column_3 = st.columns(3)
summary_column_1.metric("Average forecast", f"{np.mean(predicted_load):.3f} MW")
summary_column_2.metric("Peak forecast", f"{np.max(predicted_load):.3f} MW")
summary_column_3.metric(
    "Expected peak hour",
    forecast_display.iloc[peak_position]["Hour"],
)

chart_data = forecast_display.set_index("timestamp")[["Predicted Load (MW)"]]
st.line_chart(chart_data, height=430)

with st.expander("View hourly forecast table"):
    st.dataframe(
        forecast_display[
            [
                "Hour",
                "Predicted Load (MW)",
                "temperature_c",
                "humidity_percent",
                "holiday_name",
            ]
        ].rename(
            columns={
                "temperature_c": "Temperature (°C)",
                "humidity_percent": "Humidity (%)",
                "holiday_name": "Holiday",
            }
        ),
        hide_index=True,
        width="stretch",
    )

download_data = forecast_display.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download selected forecast as CSV",
    data=download_data,
    file_name=(
        f"{selected_feeder.replace(' ', '_')}_"
        f"{selected_forecast_date.date()}_forecast.csv"
    ),
    mime="text/csv",
)

st.info(
    "This is a future forecast demonstration, so actual 2026 load and future-error "
    "metrics are not shown. Historical model accuracy is reported using the "
    "untouched 2025 test period."
)
