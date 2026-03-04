"""Streamlit dashboard for AI-powered stock and ETF signal generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import MODELS_DIR, load_config
from modules.alerts import (
    AlertConfig,
    build_alert_message,
    send_email_alert,
    send_slack_alert,
    signal_label,
)
from modules.backtester import BacktestSummary, Backtester
from modules.data_loader import DataLoader
from modules.feature_engineering import add_technical_indicators, build_target, feature_columns
from modules.signal_engine import SignalEngine, TrainResult

LOGGER = logging.getLogger(__name__)

MODEL_MAP = {
    "RandomForest": "random_forest",
    "GradientBoosting": "gradient_boosting",
}


@dataclass
class AnalysisResult:
    """Container for dashboard analysis output."""

    ticker: str
    model_label: str
    train_result: TrainResult
    backtest_summary: BacktestSummary
    backtest_df: pd.DataFrame
    latest_signal: int
    current_price: float


def configure_logging() -> None:
    """Configure application logging once."""

    if logging.getLogger().handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def configure_page() -> None:
    """Configure Streamlit page settings and top header."""

    st.set_page_config(
        page_title="AI Stock & ETF Signal Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
    )

    st.title("AI Stock & ETF Signal Dashboard")
    st.caption(
        "Interactive ML signal generation with backtesting, visualization, and optional alerts."
    )


@st.cache_data(ttl=1800)
def fetch_data_cached(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch historical OHLCV data and cache API calls."""

    cfg = load_config()
    loader = DataLoader(alpha_vantage_api_key=cfg.alpha_vantage_api_key)
    return loader.fetch_ticker_data(
        ticker=ticker,
        start=start,
        end=end,
        interval="1d",
        provider="yfinance",
    )


@st.cache_data(ttl=120)
def fetch_live_price_cached(ticker: str) -> float | None:
    """Fetch current market price and cache short-lived API calls."""

    return DataLoader.fetch_live_price(ticker)


def sidebar_inputs() -> tuple[str, date, date, str, bool, bool]:
    """Render sidebar controls and return selected values."""

    today = date.today()
    default_start = today - timedelta(days=5 * 365)

    with st.sidebar:
        st.header("Inputs")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        start_date = st.date_input("Start date", value=default_start, max_value=today)
        end_date = st.date_input("End date", value=today, max_value=today)
        model_label = st.selectbox("Model", options=list(MODEL_MAP.keys()), index=0)
        enable_alerts = st.checkbox("Enable Alerts", value=False)
        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

    return ticker, start_date, end_date, model_label, enable_alerts, run_clicked


def validate_inputs(ticker: str, start_date: date, end_date: date) -> None:
    """Validate user inputs before running analysis."""

    if not ticker:
        raise ValueError("Ticker is required.")
    if end_date <= start_date:
        raise ValueError("End date must be later than start date.")


def run_analysis(ticker: str, start_date: date, end_date: date, model_label: str) -> AnalysisResult:
    """Execute full pipeline: data, features, model, signals, and backtest."""

    raw_df = fetch_data_cached(ticker, start_date.isoformat(), end_date.isoformat())

    feature_df = add_technical_indicators(raw_df)
    model_df = build_target(feature_df)
    features = feature_columns()

    missing_features = [col for col in features if col not in model_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")
    if len(model_df) < 200:
        raise ValueError(f"Insufficient rows for training: found {len(model_df)} rows.")
    if model_df["Target"].nunique() < 2:
        raise ValueError("Target has fewer than two classes. Increase the date range.")

    model_name = MODEL_MAP[model_label]
    signal_engine = SignalEngine(model_dir=MODELS_DIR)
    train_result = signal_engine.train_and_evaluate(
        df=model_df,
        features=features,
        model_name=model_name,
        test_size=0.2,
        random_state=42,
    )

    signal_engine.save_model(
        train_result.model,
        filename=f"{ticker}_{model_name}_streamlit.joblib",
    )

    signal_df = signal_engine.generate_signals(model_df, train_result.model, features)
    backtester = Backtester(initial_capital=10_000.0)
    backtest_df, summary = backtester.run(signal_df)

    latest_signal = int(backtest_df["Signal"].iloc[-1])
    live_price = fetch_live_price_cached(ticker)
    current_price = float(live_price) if live_price is not None else float(backtest_df["Adj Close"].iloc[-1])

    return AnalysisResult(
        ticker=ticker,
        model_label=model_label,
        train_result=train_result,
        backtest_summary=summary,
        backtest_df=backtest_df,
        latest_signal=latest_signal,
        current_price=current_price,
    )


def send_alerts(result: AnalysisResult) -> tuple[bool, bool, str]:
    """Send email and Slack alerts for BUY/SELL signal and return status."""

    if result.latest_signal not in (1, -1):
        return False, False, "Latest signal is HOLD. No alert sent."

    cfg = load_config()
    alert_config = AlertConfig(
        smtp_host=cfg.smtp_host,
        smtp_port=cfg.smtp_port,
        smtp_user=cfg.smtp_user,
        smtp_password=cfg.smtp_password,
        email_from=cfg.email_from,
        email_to=cfg.email_to,
        slack_webhook_url=cfg.slack_webhook_url,
    )

    signal_text = signal_label(result.latest_signal)
    message = build_alert_message(
        ticker=result.ticker,
        signal=result.latest_signal,
        price=result.current_price,
    )
    subject = f"[Signal Alert] {result.ticker} {signal_text}"

    email_sent = send_email_alert(alert_config, subject=subject, body=message)
    slack_sent = send_slack_alert(alert_config, message=message)

    status = (
        f"Alerts processed for {result.ticker} {signal_text}. "
        f"Email: {'sent' if email_sent else 'skipped/failed'} | "
        f"Slack: {'sent' if slack_sent else 'skipped/failed'}"
    )
    return email_sent, slack_sent, status


def plot_price_chart(df: pd.DataFrame) -> None:
    """Render interactive candlestick chart with overlays and markers."""

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#00A86B",
            decreasing_line_color="#D72638",
        )
    )

    if "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                mode="lines",
                line={"width": 1.8, "color": "#1f77b4"},
                name="SMA 20",
            )
        )

    if "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                mode="lines",
                line={"width": 1.8, "color": "#ff7f0e"},
                name="SMA 50",
            )
        )

    buys = df[df["Trade_Action"] == 1]
    sells = df[df["Trade_Action"] == -1]

    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys.index,
                y=buys["Adj Close"],
                mode="markers",
                marker={"symbol": "triangle-up", "size": 12, "color": "#00A86B"},
                name="Buy",
            )
        )

    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells.index,
                y=sells["Adj Close"],
                mode="markers",
                marker={"symbol": "triangle-down", "size": 12, "color": "#D72638"},
                name="Sell",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 8, "r": 8, "t": 30, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_equity_curve(df: pd.DataFrame) -> None:
    """Render interactive equity curve chart."""

    fig = go.Figure(
        go.Scatter(
            x=df.index,
            y=df["Portfolio_Value"],
            mode="lines",
            line={"width": 2, "color": "#0A3D62"},
            name="Equity Curve",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 8, "r": 8, "t": 30, "b": 8},
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_signal_summary(result: AnalysisResult) -> None:
    """Render latest signal, current price, and model accuracy."""

    st.subheader("Section 2 - Signal Summary")
    c1, c2, c3 = st.columns(3)

    c1.metric("Latest Signal", signal_label(result.latest_signal))
    c2.metric("Current Price", f"${result.current_price:,.2f}")
    c3.metric("Model Accuracy", f"{result.train_result.accuracy * 100:.2f}%")


def render_backtest_metrics(summary: BacktestSummary) -> None:
    """Render backtest KPI metrics."""

    st.subheader("Section 3 - Backtest Performance")
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Return %", f"{summary.total_return_pct:.2f}%")
    k2.metric("Sharpe Ratio", f"{summary.sharpe_ratio:.3f}")
    k3.metric("Max Drawdown %", f"{summary.max_drawdown_pct:.2f}%")
    k4.metric("Win Rate %", f"{summary.win_rate_pct:.2f}%")


def render_data_table(df: pd.DataFrame) -> None:
    """Render the last 20 rows with signal columns."""

    st.subheader("Section 4 - Data Table")

    display_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "RSI_14",
        "MACD",
        "Signal",
        "Trade_Action",
        "Portfolio_Value",
    ]
    available_columns = [col for col in display_columns if col in df.columns]

    table_df = df[available_columns].tail(20).copy().reset_index()
    first_col = table_df.columns[0]
    table_df = table_df.rename(columns={first_col: "Date"})
    table_df["Date"] = pd.to_datetime(table_df["Date"]).dt.strftime("%Y-%m-%d")

    st.dataframe(table_df, use_container_width=True, hide_index=True)


def render_dashboard(result: AnalysisResult) -> None:
    """Render all dashboard sections using analysis results."""

    st.subheader("Section 1 - Price Chart")
    plot_price_chart(result.backtest_df)

    render_signal_summary(result)

    render_backtest_metrics(result.backtest_summary)
    plot_equity_curve(result.backtest_df)

    render_data_table(result.backtest_df)


def main() -> None:
    """Streamlit application entry point."""

    configure_logging()
    configure_page()

    ticker, start_date, end_date, model_label, enable_alerts, run_clicked = sidebar_inputs()

    if run_clicked:
        try:
            validate_inputs(ticker, start_date, end_date)
            with st.spinner("Running analysis pipeline..."):
                result = run_analysis(ticker, start_date, end_date, model_label)
            st.session_state["analysis_result"] = result
            st.success("Analysis completed successfully.")

            if enable_alerts:
                _, _, alert_status = send_alerts(result)
                st.session_state["alert_status"] = alert_status
            else:
                st.session_state["alert_status"] = "Alerts are disabled."

        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Dashboard analysis failed: %s", exc)
            st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("analysis_result")
    if result is None:
        st.info("Use the sidebar to configure inputs, then click Run Analysis.")
        return

    render_dashboard(result)

    alert_status = st.session_state.get("alert_status")
    if alert_status:
        st.info(f"Alerts: {alert_status}")


if __name__ == "__main__":
    main()
