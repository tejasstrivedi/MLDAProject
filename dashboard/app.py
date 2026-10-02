"""Interactive dashboard for day-ahead 11 kV feeder load forecasting."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "final_load_forecasting_bundle.joblib"
FEEDER_DATA_PATH = PROJECT_ROOT / "feeder_2023_2025.xlsx"
HISTORICAL_WEATHER_PATH = PROJECT_ROOT / "external_data" / "rajkot_hourly_weather_2023_2025.csv"
HISTORICAL_HOLIDAYS_PATH = PROJECT_ROOT / "external_data" / "gujarat_public_holidays_2023_2025.csv"
WEATHER_2026_PATH = PROJECT_ROOT / "external_data" / "rajkot_synthetic_weather_2026.csv"
HOLIDAYS_2026_PATH = PROJECT_ROOT / "external_data" / "gujarat_public_holidays_2026.csv"

COLORS = {
    "navy": "#12304A", "blue": "#2474E5", "cyan": "#20A4B8",
    "orange": "#F59E0B", "green": "#1C9A66", "red": "#D94A4A",
    "purple": "#7C5CE5", "grey": "#7A8793",
}

st.set_page_config(
    page_title="Rajkot Feeder Load Forecast", page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #F5F9FC 0%, #FFFFFF 34%);}
    [data-testid="stSidebar"] {background: #EEF5F9;}
    .hero {padding:1.35rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#12304A 0%,#176B87 58%,#20A4B8 100%);color:white;margin-bottom:1rem;box-shadow:0 8px 24px rgba(18,48,74,.16);}
    .hero h1 {margin:0;font-size:2rem;color:white;}.hero p {margin:.35rem 0 0;opacity:.92;}
    .note-card {background:white;border:1px solid #DCE8EF;border-left:5px solid #20A4B8;border-radius:12px;padding:.8rem 1rem;margin:.4rem 0 1rem;}
    div[data-testid="stMetric"] {background:rgba(255,255,255,.88);border:1px solid #DCE8EF;border-radius:14px;padding:.75rem 1rem;box-shadow:0 4px 14px rgba(18,48,74,.06);}
    div[data-testid="stDataFrame"] {
        border: 1px solid #D7E5ED;
        border-radius: 13px;
        overflow: hidden;
        box-shadow: 0 5px 16px rgba(18,48,74,.07);
        background: #FFFFFF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: .45rem;
        padding: .42rem;
        border: 1px solid #D7E5ED;
        border-radius: 14px;
        background: #EAF2F7;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        flex: 1 0 auto;
        min-width: 150px;
        height: 46px;
        padding: 0 1rem;
        border: 1px solid #D7E5ED;
        border-radius: 10px;
        background: #FFFFFF;
        color: #29465B;
        font-weight: 650;
        transition: background .18s ease, color .18s ease, box-shadow .18s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #DDECF4;
        color: #12304A;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(120deg, #176B87, #20A4B8) !important;
        border-color: #176B87 !important;
        color: #FFFFFF !important;
        box-shadow: 0 5px 14px rgba(23,107,135,.24);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p {
        color: #FFFFFF !important;
        font-weight: 750;
    }
    .stTabs [data-baseweb="tab-highlight"] {display: none;}
    @media (max-width: 900px) {
        .block-container {padding-left: 1rem; padding-right: 1rem;}
        .hero {padding: 1.05rem 1.1rem; border-radius: 14px;}
        .hero h1 {font-size: 1.55rem; line-height: 1.2;}
        .hero p {font-size: .9rem;}
        .stTabs [data-baseweb="tab-list"] {justify-content: flex-start;}
        .stTabs [data-baseweb="tab"] {min-width: 165px;}
    }
    @media (max-width: 600px) {
        .block-container {padding-top: 1rem; padding-left: .65rem; padding-right: .65rem;}
        .hero h1 {font-size: 1.3rem;}
        .hero p {font-size: .82rem;}
        div[data-testid="stMetric"] {padding: .55rem .7rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle(model_path: Path) -> dict:
    return joblib.load(model_path)


def merge_external_data(weather_path: Path, holiday_path: Path) -> pd.DataFrame:
    weather = pd.read_csv(weather_path, parse_dates=["timestamp"])
    holidays = pd.read_csv(holiday_path, parse_dates=["date"])
    weather["date"] = weather["timestamp"].dt.normalize()
    result = weather.merge(holidays, on="date", how="left", validate="many_to_one")
    result["is_holiday"] = result["holiday_name"].notna().astype(int)
    return result


@st.cache_data
def load_project_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    historical_load = pd.read_excel(
        FEEDER_DATA_PATH, sheet_name="Hourly_Data",
        usecols=["Date", "Hour Interval", "Feeder Name", "MW(I)"],
    )
    historical_load["timestamp"] = pd.to_datetime(
        historical_load["Date"].astype(str) + " " + historical_load["Hour Interval"].astype(str)
    )
    historical_load = historical_load[["timestamp", "Feeder Name", "MW(I)"]].sort_values(
        ["Feeder Name", "timestamp"]
    ).reset_index(drop=True)
    historical_external = merge_external_data(HISTORICAL_WEATHER_PATH, HISTORICAL_HOLIDAYS_PATH)
    future_external = merge_external_data(WEATHER_2026_PATH, HOLIDAYS_2026_PATH)
    return historical_load, historical_external, future_external


def build_forecast_features(
    feeder_name: str, forecast_date: pd.Timestamp, load_history: pd.DataFrame,
    external_data: pd.DataFrame, feeder_mapping: dict, feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recreate the exact leakage-safe features used during model training."""
    forecast_start = pd.Timestamp(forecast_date).normalize()
    forecast_end = forecast_start + pd.Timedelta(days=1)
    rows = external_data.loc[
        (external_data["timestamp"] >= forecast_start)
        & (external_data["timestamp"] < forecast_end)
    ].copy()
    if len(rows) != 24:
        raise ValueError("Exactly 24 hourly weather records are required.")

    rows["Feeder Name"] = feeder_name
    rows["hour"] = rows["timestamp"].dt.hour
    rows["day_of_week_number"] = rows["timestamp"].dt.dayofweek
    rows["month"] = rows["timestamp"].dt.month
    rows["is_weekend"] = (rows["day_of_week_number"] >= 5).astype(int)
    rows["feeder_id"] = feeder_mapping[feeder_name]
    rows["hour_sin"] = np.sin(2 * np.pi * rows["hour"] / 24)
    rows["hour_cos"] = np.cos(2 * np.pi * rows["hour"] / 24)
    rows["day_of_week_sin"] = np.sin(2 * np.pi * rows["day_of_week_number"] / 7)
    rows["day_of_week_cos"] = np.cos(2 * np.pi * rows["day_of_week_number"] / 7)
    rows["month_sin"] = np.sin(2 * np.pi * (rows["month"] - 1) / 12)
    rows["month_cos"] = np.cos(2 * np.pi * (rows["month"] - 1) / 12)

    feeder_history = (
        load_history.loc[load_history["Feeder Name"] == feeder_name]
        .drop_duplicates("timestamp", keep="last").set_index("timestamp")["MW(I)"].sort_index()
    )
    for lag_hours in (24, 48, 168):
        rows[f"load_lag_{lag_hours}"] = rows["timestamp"].map(
            lambda timestamp, lag=lag_hours: feeder_history.get(
                timestamp - pd.Timedelta(hours=lag), np.nan
            )
        )

    def rolling_value(timestamp: pd.Timestamp, window: int, method: str) -> float:
        window_end = timestamp - pd.Timedelta(hours=24)
        window_start = window_end - pd.Timedelta(hours=window - 1)
        values = feeder_history.reindex(pd.date_range(window_start, window_end, freq="h"))
        if values.isna().any():
            return np.nan
        return float(values.mean() if method == "mean" else values.std())

    rows["rolling_mean_24"] = rows["timestamp"].map(lambda t: rolling_value(t, 24, "mean"))
    rows["rolling_std_24"] = rows["timestamp"].map(lambda t: rolling_value(t, 24, "std"))
    rows["rolling_mean_168"] = rows["timestamp"].map(lambda t: rolling_value(t, 168, "mean"))
    model_input = rows[feature_columns].copy()
    if model_input.isna().any().any():
        missing = model_input.columns[model_input.isna().any()].tolist()
        raise ValueError(f"Missing feature values: {missing}")
    return rows, model_input


def create_load_template(feeder_names: list[str], required_date: pd.Timestamp) -> pd.DataFrame:
    timestamps = pd.date_range(pd.Timestamp(required_date).normalize(), periods=24, freq="h")
    template = pd.MultiIndex.from_product(
        [timestamps, feeder_names], names=["timestamp", "Feeder Name"]
    ).to_frame(index=False)
    template["MW(I)"] = ""
    return template


def upload_quality_summary(uploaded_data: pd.DataFrame) -> dict:
    expected_columns = {"timestamp", "Feeder Name", "MW(I)"}
    available_columns = set(uploaded_data.columns)
    timestamps = pd.to_datetime(uploaded_data.get("timestamp", pd.Series(dtype=object)), errors="coerce")
    loads = pd.to_numeric(uploaded_data.get("MW(I)", pd.Series(dtype=object)), errors="coerce")
    duplicates = 0
    if {"timestamp", "Feeder Name"}.issubset(available_columns):
        duplicates = int(uploaded_data.duplicated(["timestamp", "Feeder Name"]).sum())
    return {
        "Rows": len(uploaded_data), "Expected rows": 48,
        "Recognized columns": len(expected_columns.intersection(available_columns)),
        "Missing/invalid timestamps": int(timestamps.isna().sum()),
        "Missing/invalid load values": int(loads.isna().sum()),
        "Duplicate feeder-hours": duplicates,
    }


def validate_uploaded_load(
    uploaded_data: pd.DataFrame, required_date: pd.Timestamp, feeder_names: list[str]
) -> pd.DataFrame:
    required_columns = ["timestamp", "Feeder Name", "MW(I)"]
    missing_columns = set(required_columns).difference(uploaded_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    clean = uploaded_data[required_columns].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean["MW(I)"] = pd.to_numeric(clean["MW(I)"], errors="coerce")
    if clean.isna().any().any():
        raise ValueError("Every timestamp, feeder name and MW(I) value is required.")
    if clean.duplicated(["timestamp", "Feeder Name"]).any():
        raise ValueError("Duplicate feeder-timestamp rows were found.")
    if set(clean["Feeder Name"]) != set(feeder_names):
        raise ValueError("The file must contain exactly the two recognized feeders.")
    if (clean["MW(I)"] < 0).any():
        raise ValueError("MW(I) cannot contain negative values.")
    expected = create_load_template(feeder_names, required_date)
    expected_keys = set(map(tuple, expected[["timestamp", "Feeder Name"]].to_numpy()))
    uploaded_keys = set(map(tuple, clean[["timestamp", "Feeder Name"]].to_numpy()))
    if uploaded_keys != expected_keys:
        raise ValueError(
            f"Provide all 24 hours for both feeders on {pd.Timestamp(required_date).date()} only."
        )
    return clean.sort_values(["Feeder Name", "timestamp"]).reset_index(drop=True)


def predict_method(method_name: str, rows: pd.DataFrame, model_input: pd.DataFrame, bundle: dict) -> np.ndarray:
    if method_name == "Same-hour previous week":
        return rows["load_lag_168"].to_numpy()
    return bundle["models"][method_name].predict(model_input)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def style_figure(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left", "y": 0.98}, height=500,
        margin=dict(l=20, r=20, t=125, b=20), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.16, x=0),
        xaxis=dict(title="Hour", gridcolor="#E8EEF2"),
        yaxis=dict(title=y_title, gridcolor="#E8EEF2"),
    )
    return figure


def base_table_style(data: pd.DataFrame):
    """Apply a consistent reader-facing style to dashboard tables."""
    return (
        data.style
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", COLORS["navy"]),
                        ("color", "white"),
                        ("font-weight", "700"),
                        ("text-align", "center"),
                        ("border", "1px solid #31506A"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-bottom", "1px solid #E4EDF2"),
                        ("padding", "8px"),
                    ],
                },
            ]
        )
        .set_properties(**{"color": "#19384D"})
    )


required_paths = [
    MODEL_BUNDLE_PATH, FEEDER_DATA_PATH, HISTORICAL_WEATHER_PATH,
    HISTORICAL_HOLIDAYS_PATH, WEATHER_2026_PATH, HOLIDAYS_2026_PATH,
]
missing_paths = [path for path in required_paths if not path.exists()]
if missing_paths:
    st.error("Missing project files: " + ", ".join(map(str, missing_paths)))
    st.stop()

try:
    model_bundle = load_model_bundle(MODEL_BUNDLE_PATH)
    historical_load, historical_external, future_external = load_project_data()
except Exception as error:
    st.error(f"The project files could not be loaded: {error}")
    st.stop()

required_bundle_keys = {
    "models", "recommended_model_name", "model_ranking", "dashboard_model_options",
    "feature_columns", "feeder_mapping", "validation_results", "test_results",
}
missing_bundle_keys = required_bundle_keys.difference(model_bundle)
if missing_bundle_keys:
    st.error(f"Re-run the notebook model-saving section. Missing bundle items: {sorted(missing_bundle_keys)}")
    st.stop()

model_ranking = pd.DataFrame(model_bundle["model_ranking"])
validation_results = pd.DataFrame(model_bundle["validation_results"])
test_results = pd.DataFrame(model_bundle["test_results"])
recommended_model_name = model_bundle["recommended_model_name"]
feeder_names = sorted(model_bundle["feeder_mapping"])
deployable_model_names = list(model_bundle["models"])
model_options = model_bundle["dashboard_model_options"]
recommended_label = model_ranking.loc[
    model_ranking["model"] == recommended_model_name, "dashboard_label"
].iloc[0]
TOTAL_BUS_LABEL = "Total bus load (2 feeders)"

st.markdown(
    """<div class="hero"><h1>Day-Ahead Hourly Load Forecasting</h1>
    <p>Two 11 kV feeders in Rajkot city · PGVCL academic demonstration · 24-hour outlook</p></div>""",
    unsafe_allow_html=True,
)

st.sidebar.header("Dashboard controls")
selected_load_point = st.sidebar.selectbox(
    "Forecast load point",
    feeder_names + [TOTAL_BUS_LABEL],
    index=len(feeder_names),
    help="Choose an individual feeder or the combined load of both project feeders.",
)
selected_feeder = (
    selected_load_point
    if selected_load_point in feeder_names
    else feeder_names[0]
)
selected_model_label = st.sidebar.selectbox(
    "Forecasting method", model_options, index=model_options.index(recommended_label),
    help="Rank is based on MAE for the fixed 2024 validation period.",
)
if selected_model_label == "Same-hour previous week":
    selected_model_name, selected_model_rank = "Same-hour previous week", "Benchmark"
else:
    selected_rank_row = model_ranking.loc[
        model_ranking["dashboard_label"] == selected_model_label
    ].iloc[0]
    selected_model_name = selected_rank_row["model"]
    selected_model_rank = f"Rank {int(selected_rank_row['performance_rank'])}"
st.sidebar.markdown("---")
st.sidebar.caption("Rank 1 is recommended from validation performance. The weekly method is a benchmark, not a trained model.")

future_tab, historical_tab, comparison_tab, information_tab = st.tabs(
    ["Future forecast", "Historical evaluation", "Model comparison", "Project information"]
)
if "uploaded_load_history" not in st.session_state:
    st.session_state.uploaded_load_history = pd.DataFrame(
        columns=["timestamp", "Feeder Name", "MW(I)"]
    )


with future_tab:
    st.subheader("Day-ahead forecast demonstration")
    selection_col_1, selection_col_2, selection_col_3 = st.columns(3)
    selection_col_1.metric("Selected load point", selected_load_point)
    selection_col_2.metric("Selected method", selected_model_name)
    selection_col_3.metric("Validation position", selected_model_rank)
    st.markdown(
        '<div class="note-card">Upload each completed day of actual feeder load to unlock the following forecast date. The model is not retrained; only the next day’s input features are recalculated.</div>',
        unsafe_allow_html=True,
    )
    uploaded_history = st.session_state.uploaded_load_history.copy()
    combined_history = pd.concat([historical_load, uploaded_history], ignore_index=True).sort_values(
        ["Feeder Name", "timestamp"]
    )
    next_upload_date = combined_history["timestamp"].max().normalize() + pd.Timedelta(days=1)
    last_supported_date = pd.Timestamp("2026-12-31")

    with st.expander("Add actual load readings and unlock the next date"):
        if next_upload_date <= last_supported_date:
            st.write(f"Next required date: **{next_upload_date.strftime('%d %B %Y')}**")
            template = create_load_template(feeder_names, next_upload_date)
            st.download_button(
                "Download fixed-date sample CSV", data=template.to_csv(index=False).encode("utf-8"),
                file_name=f"actual_feeder_load_{next_upload_date.date()}.csv", mime="text/csv",
            )
            st.caption("Do not change timestamps or feeder names. Fill all 48 MW(I) cells.")
            uploaded_file = st.file_uploader(
                "Upload the completed CSV", type=["csv"], key=f"load_upload_{next_upload_date.date()}"
            )
            if uploaded_file is not None:
                uploaded_candidate = pd.read_csv(uploaded_file)
                quality = upload_quality_summary(uploaded_candidate)
                st.markdown("#### Upload quality summary")
                quality_columns = st.columns(3)
                quality_columns[0].metric("Rows", f"{quality['Rows']} / 48")
                quality_columns[1].metric("Invalid load values", quality["Missing/invalid load values"])
                quality_columns[2].metric("Duplicate feeder-hours", quality["Duplicate feeder-hours"])
                quality_table = pd.DataFrame([quality])
                quality_styler = base_table_style(quality_table)
                quality_styler = quality_styler.map(
                    lambda value: (
                        "background-color:#DDF4E8;color:#146B46;font-weight:700"
                        if value == 0
                        else "background-color:#FDE2E2;color:#A12B2B;font-weight:700"
                    ),
                    subset=[
                        "Missing/invalid timestamps",
                        "Missing/invalid load values",
                        "Duplicate feeder-hours",
                    ],
                )
                quality_styler = quality_styler.map(
                    lambda value: (
                        "background-color:#DDF4E8;color:#146B46;font-weight:700"
                        if value == 48
                        else "background-color:#FFF0D2;color:#8A5700;font-weight:700"
                    ),
                    subset=["Rows"],
                )
                st.dataframe(quality_styler, hide_index=True, width="stretch")
                if st.button("Validate and add readings", type="primary"):
                    try:
                        validated = validate_uploaded_load(uploaded_candidate, next_upload_date, feeder_names)
                        st.session_state.uploaded_load_history = pd.concat(
                            [st.session_state.uploaded_load_history, validated], ignore_index=True
                        )
                        st.success("Upload accepted. The next forecast date is unlocked.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Upload rejected: {error}")
        else:
            st.success("Actual-load history is complete through 31 December 2026.")
        if not uploaded_history.empty:
            uploaded_day_count = uploaded_history["timestamp"].dt.normalize().nunique()
            st.success(f"Validated session data: {uploaded_day_count} day(s), {len(uploaded_history)} readings.")
            if st.button("Clear uploaded session data"):
                st.session_state.uploaded_load_history = pd.DataFrame(
                    columns=["timestamp", "Feeder Name", "MW(I)"]
                )
                st.rerun()

    first_forecast_date = pd.Timestamp("2026-01-01")
    latest_forecast_date = min(next_upload_date, last_supported_date)
    available_dates = pd.date_range(first_forecast_date, latest_forecast_date, freq="D")
    control_col_1, control_col_2, control_col_3, control_col_4 = st.columns(4)
    selected_forecast_date = pd.Timestamp(control_col_1.selectbox(
        "Forecast date", available_dates, index=len(available_dates) - 1,
        format_func=lambda value: pd.Timestamp(value).strftime("%d %B %Y"),
    ))
    feeder_view = control_col_2.radio(
        "Display mode",
        ["Selected load point", "Compare both feeders"],
        horizontal=True,
    )
    model_view = control_col_3.radio(
        "Forecast view", ["Selected method", "Compare all methods"], horizontal=True
    )
    overload_threshold_mw = control_col_4.number_input(
        "Demonstration overload threshold (MW)",
        min_value=0.01,
        value=1.00,
        step=0.05,
        format="%.2f",
        help=(
            "Enter an approved equipment limit when available. Otherwise this "
            "value is only a demonstration threshold."
        ),
    )
    feeders_to_plot = (
        [selected_feeder]
        if feeder_view == "Selected load point" and selected_load_point != TOTAL_BUS_LABEL
        else feeder_names
    )
    methods_to_plot = (
        [selected_model_name] if model_view == "Selected method"
        else ["Same-hour previous week"] + deployable_model_names
    )
    if model_view == "Selected method" and selected_model_name != "Same-hour previous week":
        methods_to_plot = ["Same-hour previous week", selected_model_name]

    forecast_frames = []
    try:
        with st.spinner("Generating the 24-hour forecast..."):
            for feeder_name in feeders_to_plot:
                rows, model_input = build_forecast_features(
                    feeder_name, selected_forecast_date, combined_history, future_external,
                    model_bundle["feeder_mapping"], model_bundle["feature_columns"],
                )
                for method_name in methods_to_plot:
                    values = predict_method(method_name, rows, model_input, model_bundle)
                    result = rows[[
                        "timestamp", "Feeder Name", "temperature_c", "humidity_percent", "holiday_name"
                    ]].copy()
                    result["Method"], result["Forecast MW"] = method_name, values
                    forecast_frames.append(result)
        future_predictions = pd.concat(forecast_frames, ignore_index=True)
    except Exception as error:
        st.error(f"The future forecast could not be generated: {error}")
        st.stop()

    if feeder_view == "Selected load point" and selected_load_point == TOTAL_BUS_LABEL:
        displayed_predictions = (
            future_predictions.groupby(["timestamp", "Method"], as_index=False)
            .agg(
                {
                    "Forecast MW": "sum",
                    "temperature_c": "first",
                    "humidity_percent": "first",
                    "holiday_name": "first",
                }
            )
        )
        displayed_predictions["Feeder Name"] = TOTAL_BUS_LABEL
        selected_entity = TOTAL_BUS_LABEL
        st.info(
            "Total bus load is calculated by adding the two feeder forecasts at "
            "each hour. It represents the complete bus load only if these are the "
            "only feeders connected to that bus."
        )
    else:
        displayed_predictions = future_predictions.copy()
        selected_entity = selected_feeder

    selected_curve = displayed_predictions.loc[
        (displayed_predictions["Feeder Name"] == selected_entity)
        & (displayed_predictions["Method"] == selected_model_name)
    ]
    if selected_curve.empty:
        selected_curve = displayed_predictions.loc[
            displayed_predictions["Method"] == selected_model_name
        ].head(24)
    selected_curve = selected_curve.copy()
    selected_curve["Utilization (%)"] = (
        selected_curve["Forecast MW"] / overload_threshold_mw * 100
    )
    selected_curve["Load status"] = np.select(
        [
            selected_curve["Utilization (%)"] > 100,
            selected_curve["Utilization (%)"] >= 80,
        ],
        ["Overload", "Warning"],
        default="Normal",
    )
    overload_rows = selected_curve.loc[selected_curve["Load status"] == "Overload"]
    warning_rows = selected_curve.loc[selected_curve["Load status"] == "Warning"]
    maximum_utilization = selected_curve["Utilization (%)"].max()
    if maximum_utilization > 100:
        daily_load_status = "Overload"
    elif maximum_utilization >= 80:
        daily_load_status = "Warning"
    else:
        daily_load_status = "Normal"

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Average forecast", f"{selected_curve['Forecast MW'].mean():.3f} MW")
    metric_col_2.metric("Minimum forecast", f"{selected_curve['Forecast MW'].min():.3f} MW")
    metric_col_3.metric("Peak forecast", f"{selected_curve['Forecast MW'].max():.3f} MW")
    peak_row = selected_curve.loc[selected_curve["Forecast MW"].idxmax()]
    metric_col_4.metric("Expected peak hour", peak_row["timestamp"].strftime("%H:%M"))

    st.markdown("#### Overload indication for the selected method")
    overload_col_1, overload_col_2, overload_col_3, overload_col_4 = st.columns(4)
    overload_col_1.metric("Daily load status", daily_load_status)
    overload_col_2.metric("Peak utilization", f"{maximum_utilization:.1f}%")
    overload_col_3.metric("Warning hours", len(warning_rows))
    overload_col_4.metric("Overload hours", len(overload_rows))
    if not overload_rows.empty:
        first_overload_time = overload_rows.iloc[0]["timestamp"].strftime("%H:%M")
        st.error(
            f"Forecast overload begins at {first_overload_time}. The selected "
            f"forecast exceeds the {overload_threshold_mw:.2f} MW demonstration threshold."
        )
    elif not warning_rows.empty:
        st.warning(
            "The forecast enters the warning zone at or above 80% of the entered threshold."
        )
    else:
        st.success("The selected forecast remains below 80% of the entered threshold.")
    st.caption(
        "This is a user-entered demonstration threshold. It represents verified "
        "equipment capacity only when an approved operational rating is entered."
    )

    forecast_figure = go.Figure()
    method_colors = {
        "Same-hour previous week": COLORS["grey"],
        "Multiple Linear Regression": COLORS["orange"],
        "Random Forest (tuned)": COLORS["blue"], "XGBoost (tuned)": COLORS["green"],
    }
    for (feeder_name, method_name), plot_rows in displayed_predictions.groupby(
        ["Feeder Name", "Method"], sort=False
    ):
        trace_name = (
            method_name
            if feeder_view == "Selected load point"
            else f"{feeder_name} · {method_name}"
        )
        line_dash = "dash" if method_name == "Same-hour previous week" else "solid"
        if feeder_view == "Compare both feeders" and feeder_name != selected_feeder:
            line_dash = "dot" if method_name != "Same-hour previous week" else "dashdot"
        forecast_figure.add_trace(go.Scatter(
            x=plot_rows["timestamp"], y=plot_rows["Forecast MW"], name=trace_name, mode="lines",
            line=dict(color=method_colors[method_name], width=3, dash=line_dash),
            customdata=np.column_stack([plot_rows["temperature_c"], plot_rows["humidity_percent"]]),
            hovertemplate=("%{x|%H:%M}<br>Forecast: %{y:.3f} MW"
                           "<br>Temperature: %{customdata[0]:.1f} °C"
                           "<br>Humidity: %{customdata[1]:.0f}%<extra></extra>"),
        ))
    forecast_figure.add_trace(go.Scatter(
        x=[peak_row["timestamp"]], y=[peak_row["Forecast MW"]], name="Selected peak",
        mode="markers+text", marker=dict(size=12, color=COLORS["red"], symbol="diamond"),
        text=[f"Peak {peak_row['Forecast MW']:.3f} MW"], textposition="top center", hoverinfo="skip",
    ))
    forecast_figure.add_hline(
        y=overload_threshold_mw,
        line_color=COLORS["red"],
        line_dash="dash",
        line_width=2,
        annotation_text=f"Overload threshold {overload_threshold_mw:.2f} MW",
        annotation_position="top right",
    )
    forecast_figure.add_hline(
        y=overload_threshold_mw * 0.80,
        line_color=COLORS["orange"],
        line_dash="dot",
        line_width=1.5,
        annotation_text="Warning level 80%",
        annotation_position="bottom right",
    )
    if not overload_rows.empty:
        forecast_figure.add_trace(go.Scatter(
            x=overload_rows["timestamp"],
            y=overload_rows["Forecast MW"],
            name="Overload hour",
            mode="markers",
            marker=dict(size=11, color=COLORS["red"], symbol="x"),
            hovertemplate=(
                "%{x|%H:%M}<br>Forecast: %{y:.3f} MW"
                "<br>Status: Overload<extra></extra>"
            ),
        ))
    holiday_names = displayed_predictions["holiday_name"].dropna().unique()
    if len(holiday_names):
        forecast_figure.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.98, xanchor="right", yanchor="top",
            text=f"Holiday: {holiday_names[0]}", showarrow=False,
            bgcolor="#FFF3CD", bordercolor=COLORS["orange"],
        )
    style_figure(
        forecast_figure, f"Forecast profile · {selected_forecast_date.strftime('%d %B %Y')}",
        "Forecast load (MW)",
    )
    st.plotly_chart(forecast_figure, width="stretch")
    future_summary = displayed_predictions.groupby(["Feeder Name", "Method"])["Forecast MW"].agg(
        Average="mean", Minimum="min", Maximum="max"
    ).reset_index()
    future_summary_styler = (
        base_table_style(future_summary)
        .format({"Average": "{:.4f}", "Minimum": "{:.4f}", "Maximum": "{:.4f}"})
        .background_gradient(
            cmap="Blues", subset=["Average", "Minimum", "Maximum"], low=0.15, high=0.75
        )
        .highlight_max(
            subset=["Maximum"],
            props="background-color:#DDF4E8;color:#146B46;font-weight:700",
        )
    )
    st.dataframe(future_summary_styler, hide_index=True, width="stretch")

    st.markdown("### Hourly load forecast")
    forecast_day_name = selected_forecast_date.day_name()
    forecast_day_type = (
        "Weekend" if selected_forecast_date.dayofweek >= 5 else "Weekday"
    )
    common_holiday_names = displayed_predictions["holiday_name"].dropna().unique()
    public_holiday_label = (
        str(common_holiday_names[0]) if len(common_holiday_names) else "No"
    )
    st.markdown("#### Common information for the selected date")
    common_col_1, common_col_2, common_col_3, common_col_4 = st.columns(4)
    common_col_1.metric("Forecast date", selected_forecast_date.strftime("%d %B %Y"))
    common_col_2.metric("Day", forecast_day_name)
    common_col_3.metric("Day classification", forecast_day_type)
    common_col_4.metric("Public holiday", public_holiday_label)
    st.caption(
        "Weather location: Rajkot city, Gujarat. Temperature and humidity remain "
        "hourly values and are therefore shown inside the table."
    )

    hourly_context = (
        displayed_predictions.groupby(["timestamp", "Feeder Name"], as_index=False)
        .agg(
            {
                "temperature_c": "first",
                "humidity_percent": "first",
            }
        )
    )
    hourly_values = (
        displayed_predictions.pivot_table(
            index=["timestamp", "Feeder Name"],
            columns="Method",
            values="Forecast MW",
            aggfunc="first",
        )
        .reset_index()
    )
    hourly_forecast_table = hourly_context.merge(
        hourly_values,
        on=["timestamp", "Feeder Name"],
        how="left",
        validate="one_to_one",
    )
    hourly_status = displayed_predictions.loc[
        displayed_predictions["Method"] == selected_model_name,
        ["timestamp", "Feeder Name", "Forecast MW"],
    ].copy()
    hourly_status["Utilization (%)"] = (
        hourly_status["Forecast MW"] / overload_threshold_mw * 100
    )
    hourly_status["Load status"] = np.select(
        [
            hourly_status["Utilization (%)"] > 100,
            hourly_status["Utilization (%)"] >= 80,
        ],
        ["Overload", "Warning"],
        default="Normal",
    )
    hourly_status = hourly_status.drop(columns="Forecast MW")
    hourly_forecast_table = hourly_forecast_table.merge(
        hourly_status,
        on=["timestamp", "Feeder Name"],
        how="left",
        validate="one_to_one",
    )
    hourly_forecast_table.insert(
        0, "Hour", hourly_forecast_table["timestamp"].dt.strftime("%H:%M")
    )
    hourly_forecast_table = hourly_forecast_table.drop(columns="timestamp").rename(
        columns={
            "Feeder Name": "Load point",
            "temperature_c": "Temperature (°C)",
            "humidity_percent": "Humidity (%)",
        }
    )
    hourly_model_columns = [
        column
        for column in methods_to_plot
        if column in hourly_forecast_table.columns
    ]
    hourly_forecast_table = hourly_forecast_table[
        ["Hour", "Load point"]
        + hourly_model_columns
        + ["Utilization (%)", "Load status", "Temperature (°C)", "Humidity (%)"]
    ]
    hourly_table_styler = (
        base_table_style(hourly_forecast_table)
        .format(
            {
                **{column: "{:.4f}" for column in hourly_model_columns},
                "Utilization (%)": "{:.1f}%",
            }
        )
        .background_gradient(
            cmap="Blues",
            subset=hourly_model_columns,
            low=0.05,
            high=0.6,
        )
        .map(
            lambda value: (
                "background-color:#FDE2E2;color:#A12B2B;font-weight:700"
                if value == "Overload"
                else (
                    "background-color:#FFF0D2;color:#8A5700;font-weight:700"
                    if value == "Warning"
                    else "background-color:#DDF4E8;color:#146B46;font-weight:700"
                )
            ),
            subset=["Load status"],
        )
    )
    st.dataframe(
        hourly_table_styler,
        hide_index=True,
        width="stretch",
        height=620,
    )
    st.download_button(
        "Download displayed forecast data", data=displayed_predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"feeder_forecast_{selected_forecast_date.date()}.csv", mime="text/csv",
    )


with historical_tab:
    st.subheader("Actual versus predicted load")
    st.write(
        "Choose any day from the held-out 2025 test period. The 2025 records were not used to train "
        "the models or tune their settings; they were reserved for final evaluation. The chart compares "
        "observed synthetic load, the selected trained model and the weekly baseline."
    )
    history_control_1, history_control_2 = st.columns(2)
    historical_date = pd.Timestamp(history_control_1.date_input(
        "Evaluation date", value=pd.Timestamp("2025-01-01").date(),
        min_value=pd.Timestamp("2025-01-01").date(), max_value=pd.Timestamp("2025-12-31").date(),
    ))
    evaluation_options = [option for option in model_options if option != "Same-hour previous week"]
    evaluation_label = history_control_2.selectbox(
        "Model for historical evaluation", evaluation_options,
        index=evaluation_options.index(recommended_label),
    )
    evaluation_model = model_ranking.loc[
        model_ranking["dashboard_label"] == evaluation_label, "model"
    ].iloc[0]
    try:
        evaluation_rows, evaluation_input = build_forecast_features(
            selected_feeder, historical_date, historical_load, historical_external,
            model_bundle["feeder_mapping"], model_bundle["feature_columns"],
        )
        evaluation_predictions = predict_method(evaluation_model, evaluation_rows, evaluation_input, model_bundle)
        baseline_predictions = evaluation_rows["load_lag_168"].to_numpy()
        actual_rows = historical_load.loc[
            (historical_load["Feeder Name"] == selected_feeder)
            & (historical_load["timestamp"] >= historical_date)
            & (historical_load["timestamp"] < historical_date + pd.Timedelta(days=1))
        ].sort_values("timestamp")
        if len(actual_rows) != 24:
            raise ValueError("Exactly 24 observed load values were expected.")
        actual_values = actual_rows["MW(I)"].to_numpy()
    except Exception as error:
        st.error(f"Historical evaluation could not be prepared: {error}")
        st.stop()

    daily_metrics = regression_metrics(actual_values, evaluation_predictions)
    baseline_metrics = regression_metrics(actual_values, baseline_predictions)
    improvement = ((baseline_metrics["MAE"] - daily_metrics["MAE"]) / baseline_metrics["MAE"] * 100
                   if baseline_metrics["MAE"] else 0)
    history_metric_1, history_metric_2, history_metric_3, history_metric_4 = st.columns(4)
    history_metric_1.metric("Daily MAE", f"{daily_metrics['MAE']:.4f} MW")
    history_metric_2.metric("Daily RMSE", f"{daily_metrics['RMSE']:.4f} MW")
    history_metric_3.metric("Daily R²", f"{daily_metrics['R2']:.4f}")
    history_metric_4.metric("MAE vs baseline", f"{improvement:.2f}%")

    history_figure = go.Figure()
    history_figure.add_trace(go.Scatter(
        x=actual_rows["timestamp"], y=actual_values, name="Actual load", mode="lines+markers",
        line=dict(color=COLORS["navy"], width=4),
    ))
    history_figure.add_trace(go.Scatter(
        x=evaluation_rows["timestamp"], y=evaluation_predictions, name=evaluation_model, mode="lines",
        line=dict(color=COLORS["blue"], width=3),
    ))
    history_figure.add_trace(go.Scatter(
        x=evaluation_rows["timestamp"], y=baseline_predictions, name="Weekly baseline", mode="lines",
        line=dict(color=COLORS["grey"], width=2, dash="dash"),
    ))
    style_figure(
        history_figure, f"{selected_feeder} · {historical_date.strftime('%d %B %Y')}", "Load (MW)"
    )
    st.plotly_chart(history_figure, width="stretch")
    historical_table = pd.DataFrame({
        "Timestamp": actual_rows["timestamp"].to_numpy(), "Actual MW": actual_values,
        f"{evaluation_model} MW": evaluation_predictions, "Weekly baseline MW": baseline_predictions,
        "Absolute error MW": np.abs(actual_values - evaluation_predictions),
    })
    with st.expander("View hourly actual and predicted values"):
        historical_table_display = historical_table.copy()
        historical_numeric_columns = historical_table_display.select_dtypes(
            include="number"
        ).columns
        historical_table_display[historical_numeric_columns] = (
            historical_table_display[historical_numeric_columns].round(4)
        )
        historical_styler = (
            base_table_style(historical_table_display)
            .format(
                {
                    column: "{:.4f}"
                    for column in historical_numeric_columns
                }
            )
            .background_gradient(
                cmap="Reds", subset=["Absolute error MW"], low=0.05, high=0.65
            )
        )
        st.dataframe(historical_styler, hide_index=True, width="stretch")


with comparison_tab:
    st.subheader("Model performance and ranking")
    st.markdown(
        '<div class="note-card">Models are ranked using 2024 validation MAE. The held-out 2025 data were excluded from training and tuning, and are used only to report final performance.</div>',
        unsafe_allow_html=True,
    )
    validation_display = validation_results.copy()
    validation_display["model"] = validation_display["model"].replace(
        {"Same-hour previous week": "Weekly baseline"}
    )
    performance_table = validation_display[["model", "MAE", "RMSE", "R2"]].merge(
        test_results[["model", "MAE", "RMSE", "R2"]], on="model", how="inner",
        suffixes=(" validation", " test"),
    )
    rank_lookup = dict(zip(model_ranking["model"], model_ranking["performance_rank"]))
    performance_table.insert(0, "Position", performance_table["model"].map(
        lambda name: f"Rank {int(rank_lookup[name])}" if name in rank_lookup else "Benchmark"
    ))
    performance_table["Status"] = performance_table["model"].map(
        lambda name: "Recommended" if name == recommended_model_name
        else ("Reference" if name == "Weekly baseline" else "Available")
    )
    performance_table = performance_table.rename(columns={"model": "Model"})
    def highlight_performance_row(row: pd.Series) -> list[str]:
        if row["Status"] == "Recommended":
            style = "background-color:#DDF4E8;color:#146B46;font-weight:700"
        elif row["Status"] == "Reference":
            style = "background-color:#EDF1F4;color:#4B5963"
        else:
            style = "background-color:#E8F2FF;color:#194F8C"
        return [style] * len(row)

    performance_numeric_columns = [
        "MAE validation", "RMSE validation", "R2 validation",
        "MAE test", "RMSE test", "R2 test",
    ]
    performance_styler = (
        base_table_style(performance_table)
        .format({column: "{:.4f}" for column in performance_numeric_columns})
        .apply(highlight_performance_row, axis=1)
    )
    st.dataframe(performance_styler, hide_index=True, width="stretch")

    comparison_figure = go.Figure()
    comparison_figure.add_trace(go.Bar(
        x=performance_table["Model"], y=performance_table["MAE validation"],
        name="Validation MAE", marker_color=COLORS["cyan"],
    ))
    comparison_figure.add_trace(go.Bar(
        x=performance_table["Model"], y=performance_table["MAE test"],
        name="Test MAE", marker_color=COLORS["blue"],
    ))
    comparison_figure.update_layout(barmode="group")
    style_figure(
        comparison_figure, "Validation and test MAE by forecasting method", "MAE (MW) · lower is better"
    )
    st.plotly_chart(comparison_figure, width="stretch")
    selected_test_name = "Weekly baseline" if selected_model_name == "Same-hour previous week" else selected_model_name
    selected_test = test_results.loc[test_results["model"] == selected_test_name].iloc[0]
    score_1, score_2, score_3, score_4 = st.columns(4)
    score_1.metric("Selected test MAE", f"{selected_test['MAE']:.4f} MW")
    score_2.metric("Selected test RMSE", f"{selected_test['RMSE']:.4f} MW")
    score_3.metric("Selected test R²", f"{selected_test['R2']:.4f}")
    score_4.metric("Test MAE improvement", f"{selected_test['MAE improvement over baseline (%)']:.2f}%")


with information_tab:
    st.subheader("Project purpose")
    st.write(
        "The project demonstrates a reproducible method for forecasting 24 hourly load values for two "
        "11 kV feeders. It supports feeder-level planning by showing expected demand shape, minimum load, "
        "peak load and peak timing."
    )
    scope_column, limitation_column = st.columns(2)
    with scope_column:
        st.markdown("### Current academic prototype")
        st.markdown("""
        - Synthetic hourly feeder data for 2023–2025
        - Two Rajkot city 11 kV feeders
        - Weekly baseline and three trained models
        - Chronological training, validation and test periods
        - Rajkot temperature, humidity and Gujarat holidays
        - 2026 day-ahead demonstration with sequential actual-load uploads
        - Downloadable forecasts and historical accuracy evaluation
        """)
    with limitation_column:
        st.markdown("### Assumptions and limitations")
        st.markdown("""
        - Data demonstrate methodology, not operational accuracy.
        - Feeder connectivity and consumer composition remain stable.
        - 2026 weather values are synthetic.
        - Reverse power export from rooftop solar is not simulated.
        - External weather was merged as a predictor and did not modify MW(I).
        - A new day requires the previous day's actual hourly load.
        - Forecast uncertainty intervals are outside the present scope.
        """)

    st.markdown("### Future enhancements for a live project")
    enhancement_col_1, enhancement_col_2 = st.columns(2)
    with enhancement_col_1:
        st.markdown("""
        - **Authentication and role-based access** for operators and administrators
        - **Live weather APIs** for temperature, humidity and rainfall forecasts
        - **SCADA, AMI or meter integration** instead of CSV-only updates
        - **Operational database** for readings, forecasts and audit history
        - **Automatic scheduling** for daily forecast generation
        - **Secure cloud deployment** for access from operating locations
        - **Data-quality monitoring** for missing and abnormal readings
        """)
    with enhancement_col_2:
        st.markdown("""
        - **Model-performance monitoring** and drift detection
        - **Periodic retraining** using newly verified operational data
        - **Peak-load and overload alerts** for system operators
        - **Outage, event and consumer-mix inputs** for planned changes
        - **Solar import/export forecasting** where reverse flow occurs
        - **Prediction intervals** to communicate uncertainty
        - **Expansion to more feeders** and feeder-specific models
        """)
    st.info(
        "These future items describe a possible production roadmap. They are not claimed as implemented "
        "features of this academic prototype."
    )
