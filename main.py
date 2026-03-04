"""Entry point for AI-powered stock and ETF signal generation platform."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from config import DATA_DIR, MODELS_DIR, load_config
from modules.alerts import (
    AlertConfig,
    build_alert_message,
    send_email_alert,
    send_slack_alert,
    signal_label,
)
from modules.backtester import Backtester
from modules.data_loader import DataLoader
from modules.feature_engineering import add_technical_indicators, build_target, feature_columns
from modules.signal_engine import SignalEngine

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="AI-powered stock and ETF signal generation (Milestone 1-3)."
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol(s), comma-separated (e.g. AAPL or AAPL,MSFT,SPY).",
    )
    parser.add_argument("--provider", default="yfinance", choices=["yfinance", "alpha_vantage"])
    parser.add_argument("--start", default="2018-01-01", help="Historical start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=None, help="Historical end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1d", help="Data interval, e.g. 1d, 1h, 15m.")
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "random_forest", "gradient_boosting"],
        help="Model selection: auto trains both and selects best by test accuracy.",
    )
    parser.add_argument(
        "--resample",
        default=None,
        help="Optional pandas resample rule (e.g. W, M) applied post-download.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Add normalized OHLCV columns in ingestion output.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable matplotlib plots for non-interactive environments.",
    )
    parser.add_argument(
        "--disable-email",
        action="store_true",
        help="Disable SMTP email alerts.",
    )
    parser.add_argument(
        "--disable-slack",
        action="store_true",
        help="Disable Slack webhook alerts.",
    )
    return parser.parse_args()


def configure_logging(log_level: str) -> None:
    """Set application logging configuration."""

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_tickers(raw_tickers: str) -> List[str]:
    """Parse a comma-separated ticker string into a clean list."""

    items = [t.strip().upper() for t in raw_tickers.split(",")]
    return [t for t in items if t]


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist dataframe to CSV with index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def run_for_ticker(
    ticker: str,
    args: argparse.Namespace,
    loader: DataLoader,
    signal_engine: SignalEngine,
    alert_config: AlertConfig,
) -> None:
    """Execute end-to-end data, ML, backtest, and alert pipeline for one ticker."""

    LOGGER.info("========== Processing %s ==========", ticker)

    raw_df = loader.fetch_ticker_data(
        ticker=ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        provider=args.provider,
        resample_rule=args.resample,
        normalize=args.normalize,
    )
    save_dataframe(raw_df, DATA_DIR / f"{ticker}_raw.csv")

    feat_df = add_technical_indicators(raw_df)
    model_df = build_target(feat_df)
    features = feature_columns()

    missing_features = [c for c in features if c not in model_df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns for {ticker}: {missing_features}")

    if len(model_df) < 200:
        raise ValueError(f"Insufficient rows for training {ticker}: found {len(model_df)} rows.")
    if model_df["Target"].nunique() < 2:
        raise ValueError(
            f"Target has <2 classes for {ticker}. Expand date range or use a different ticker."
        )

    if args.model == "auto":
        best_result, all_results = signal_engine.train_models(
            df=model_df,
            features=features,
            random_state=42,
        )
        for name, result in all_results.items():
            LOGGER.info(
                "%s metrics | accuracy=%.4f | confusion=%s",
                name,
                result.accuracy,
                result.confusion,
            )
    else:
        best_result = signal_engine.train_and_evaluate(
            df=model_df,
            features=features,
            model_name=args.model,
            test_size=0.2,
            random_state=42,
        )
        LOGGER.info(
            "%s metrics | accuracy=%.4f | confusion=%s",
            best_result.model_name,
            best_result.accuracy,
            best_result.confusion,
        )

    model_path = signal_engine.save_model(
        best_result.model, filename=f"{ticker}_{best_result.model_name}.joblib"
    )
    LOGGER.info("Model for %s saved at %s", ticker, model_path)

    signal_df = signal_engine.generate_signals(
        df=model_df,
        model=best_result.model,
        features=features,
    )
    save_dataframe(signal_df, DATA_DIR / f"{ticker}_signals.csv")

    backtester = Backtester(initial_capital=10_000.0)
    backtest_df, summary = backtester.run(signal_df)
    save_dataframe(backtest_df, DATA_DIR / f"{ticker}_backtest.csv")
    backtester.log_summary(summary)

    if not args.no_plots:
        backtester.plot_equity_curve(backtest_df, title=f"{ticker} Equity Curve")
        backtester.plot_signals(backtest_df, title=f"{ticker} Buy/Sell Signals")

    latest = signal_df.iloc[-1]
    latest_signal = int(latest["Signal"])
    latest_price = DataLoader.fetch_live_price(ticker) or float(latest["Adj Close"])
    signal_text = signal_label(latest_signal)
    LOGGER.info(
        "Latest signal for %s | signal=%s (%s) | price=%.2f",
        ticker,
        signal_text,
        latest_signal,
        latest_price,
    )

    if latest_signal in (1, -1):
        message = build_alert_message(ticker=ticker, signal=latest_signal, price=latest_price)
        subject = f"[Signal Alert] {ticker} {signal_text}"

        if not args.disable_email:
            send_email_alert(alert_config, subject=subject, body=message)
        if not args.disable_slack:
            send_slack_alert(alert_config, message=message)
    else:
        LOGGER.info("No BUY/SELL alert for %s because latest signal is HOLD.", ticker)


def run_pipeline(args: argparse.Namespace) -> None:
    """Run signal pipeline for one or more ticker symbols."""

    app_cfg = load_config()
    configure_logging(app_cfg.log_level)

    tickers = parse_tickers(args.ticker)
    if not tickers:
        raise ValueError("At least one valid ticker must be provided.")

    loader = DataLoader(alpha_vantage_api_key=app_cfg.alpha_vantage_api_key)
    signal_engine = SignalEngine(model_dir=MODELS_DIR)
    alert_config = AlertConfig(
        smtp_host=app_cfg.smtp_host,
        smtp_port=app_cfg.smtp_port,
        smtp_user=app_cfg.smtp_user,
        smtp_password=app_cfg.smtp_password,
        email_from=app_cfg.email_from,
        email_to=app_cfg.email_to,
        slack_webhook_url=app_cfg.slack_webhook_url,
    )

    failures: list[tuple[str, str]] = []
    for ticker in tickers:
        try:
            run_for_ticker(ticker, args, loader, signal_engine, alert_config)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Ticker %s failed: %s", ticker, exc)
            failures.append((ticker, str(exc)))

    if failures:
        joined = "; ".join(f"{t}: {err}" for t, err in failures)
        raise RuntimeError(f"Pipeline finished with failures: {joined}")
    LOGGER.info("Pipeline completed successfully for tickers: %s", ", ".join(tickers))


def main() -> None:
    """CLI program entry point."""

    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
