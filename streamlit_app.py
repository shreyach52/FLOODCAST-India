from pathlib import Path
import sys
import json

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "frontend" / "dashboard"))

import logic

st.set_page_config(
    page_title="FLOODCAST India",
    page_icon="🌊",
    layout="wide",
)

st.markdown("""
<style>
    .main {background-color: #f5f9ff;}
    h1, h2, h3 {color: #164a78;}
    div[data-testid="stMetric"] {
        background: #e8f3ff;
        padding: 15px;
        border-radius: 12px;
    }
    .notice {
        background: #fff4d6;
        padding: 12px;
        border-radius: 8px;
        color: #684b00;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    discharge = pd.read_csv(
        ROOT / "data/interim/discharge_daily.csv.gz",
        parse_dates=["date"],
    )
    weather = pd.read_csv(
        ROOT / "data/interim/weather_daily.csv.gz",
        parse_dates=["date"],
    )
    forecasts = pd.read_csv(
        ROOT / "data/processed/forecasts.csv.gz",
        parse_dates=["issue_date"],
    )
    metrics = pd.read_csv(
        ROOT / "data/interim/baseline_metrics.csv",
    )

    config_path = ROOT / "frontend/configs/frontend.json"
    cfg = logic.load_cfg(config_path)

    return discharge, weather, forecasts, metrics, cfg


try:
    df, weather, forecasts, metrics, cfg = load_data()
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.stop()


thresholds = logic.thresholds(df, cfg)
stations = logic.station_table(df, thresholds, cfg)

st.title("🌊 FLOODCAST INDIA")
st.caption(
    "River discharge monitoring and historical flood forecasting | "
    "Karnataka and Kerala pilot"
)

st.markdown(
    '<div class="notice"><b>Prototype notice:</b> Forecasts shown here '
    'are historical backtest predictions, not live future predictions. '
    'This dashboard reads existing backend outputs and does not train '
    'or modify models.</div>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Dashboard controls")

station_ids = sorted(df["station_id"].dropna().unique().tolist())
selected_station = st.sidebar.selectbox(
    "Select monitoring station",
    station_ids,
)

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [1, 3, 7],
    format_func=lambda x: f"{x}-day forecast",
)

days = st.sidebar.selectbox(
    "Discharge history",
    [30, 90, 180, 365, 730],
    index=3,
    format_func=lambda x: f"Last {x} days",
)

# Overview
st.header("National monitoring overview")

fresh = stations[stations["fresh"]]
counts = {
    risk: int((stations["risk"] == risk).sum())
    for risk in logic.RISKS
}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Monitoring stations", len(stations))
c2.metric("Recently reporting", len(fresh))
c3.metric("High risk", counts["High"])
c4.metric("Elevated risk", counts["Elevated"])

st.caption(
    f"Latest discharge date: {df['date'].max().strftime('%d %B %Y')} "
    f"| Reporting freshness: {cfg['fresh_days']} days"
)

left, right = st.columns(2)

with left:
    st.subheader("Station risk distribution")
    risk_data = pd.DataFrame({
        "Risk": list(counts.keys()),
        "Stations": list(counts.values()),
    }).set_index("Risk")
    st.bar_chart(risk_data)

with right:
    st.subheader("Stations by state")
    state_data = (
        stations.reset_index()
        .groupby("state")["last_q"]
        .count()
        .rename("Stations")
    )
    st.bar_chart(state_data)

st.subheader("Monitoring station status")

display = stations.reset_index().rename(
    columns={
        "station_id": "Station",
        "last_date": "Last reporting date",
        "last_q": "Latest discharge",
        "risk": "Risk level",
    }
)

columns = [
    c for c in [
        "Station", "river", "state",
        "Last reporting date", "Latest discharge",
        "Risk level", "fresh",
    ] if c in display.columns
]

st.dataframe(
    display[columns],
    use_container_width=True,
    hide_index=True,
)

# Station explorer
st.divider()
st.header(f"Station explorer: {selected_station}")

g = df[df["station_id"] == selected_station].sort_values("date")
station_info = stations.loc[selected_station]

a, b, c = st.columns(3)
a.metric("Latest discharge", f"{station_info['last_q']:,.2f}")
b.metric("River", str(station_info.get("river", "Unknown")))
c.metric("State", str(station_info.get("state", "Unknown")))

st.subheader("Historical river discharge")

recent = g.tail(days).set_index("date")
st.line_chart(recent[["discharge"]], height=300)

st.caption(
    f"Available record: {g['date'].min().date()} to "
    f"{g['date'].max().date()}"
)

# Forecasts
st.divider()
st.header("Historical forecast evaluation")

st.info(
    "These predictions were generated during historical testing. "
    "They are not forecasts for dates after the latest observed data."
)

fc = forecasts[forecasts["station_id"] == selected_station]
obs = g.set_index("date")["discharge"]

aligned = logic.align_forecasts(
    fc, obs, horizon, days=365
)

if aligned:
    forecast_frame = pd.DataFrame({
        "Predicted discharge": aligned["q_pred"],
        "Actual discharge": aligned["actual"],
    }, index=pd.to_datetime(aligned["dates"]))

    st.subheader(f"{horizon}-day forecast vs actual")
    st.line_chart(forecast_frame, height=350)

    x, y = st.columns(2)
    x.metric("Latest backtest issue date", aligned["last_issue_date"])
    y.metric("Latest target date", aligned["last_target_date"])

    st.caption(
        "The forecast dates represent historical target dates, "
        "not future dates."
    )
else:
    st.warning("No stored forecasts are available for this station and horizon.")

# Weather
st.divider()
st.header("Weather conditions")

wx = weather[weather["station_id"] == selected_station].sort_values("date")

if not wx.empty:
    wx_recent = wx.tail(90).set_index("date")

    st.subheader("Daily rainfall (last 90 available days)")
    st.bar_chart(wx_recent[["rain_mm"]], height=250)

    st.subheader("Soil moisture — layer 1")
    st.line_chart(wx_recent[["soil1"]], height=250)

    st.caption(
        "Soil layer 2 is unavailable in the current weather dataset."
    )
else:
    st.warning("No weather records available for this station.")

# Model metrics
st.divider()
st.header("Model performance")

if not metrics.empty:
    metric_horizon = metrics[
        (metrics["horizon"] == horizon)
    ].copy()

    if not metric_horizon.empty:
        split_options = sorted(metric_horizon["split"].dropna().unique())
        selected_split = st.selectbox(
            "Evaluation group",
            split_options,
        )

        view = metric_horizon[
            metric_horizon["split"] == selected_split
        ].copy()

        st.subheader(f"{horizon}-day model comparison")

        show_cols = [
            c for c in [
                "model", "n", "rmse", "mae",
                "hi_recall", "hi_precision",
                "onset_recall", "onset_precision",
                "median_station_nse",
            ] if c in view.columns
        ]

        st.dataframe(
            view[show_cols].round(3),
            use_container_width=True,
            hide_index=True,
        )

        if "rmse" in view.columns:
            chart = view.set_index("model")[["rmse"]]
            st.subheader("RMSE comparison (lower is better)")
            st.bar_chart(chart)
    else:
        st.warning("No metrics for this forecast horizon.")

st.divider()
st.caption(
    "FLOODCAST India | Read-only prototype | "
    "Karnataka and Kerala pilot"
)