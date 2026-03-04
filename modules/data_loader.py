"""Data ingestion module for market data providers."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
import yfinance as yf

LOGGER = logging.getLogger(__name__)
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


class DataLoader:
    """Load historical OHLCV data from supported market data sources."""

    def __init__(self, alpha_vantage_api_key: Optional[str] = None) -> None:
        self.alpha_vantage_api_key = alpha_vantage_api_key

    def fetch_ticker_data(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        provider: str = "yfinance",
        resample_rule: Optional[str] = None,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Fetch and clean data for a single ticker."""

        provider = provider.lower()
        if provider == "alpha_vantage":
            if not self.alpha_vantage_api_key:
                raise ValueError(
                    "Alpha Vantage provider selected but ALPHA_VANTAGE_API_KEY is not set."
                )
            raw = self._fetch_alpha_vantage_daily(ticker=ticker)
        elif provider == "yfinance":
            raw = self._fetch_yfinance(
                ticker=ticker, start=start, end=end, interval=interval
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        cleaned = self._clean_ohlcv(raw)
        if resample_rule:
            cleaned = self._resample_ohlcv(cleaned, resample_rule=resample_rule)
        if normalize:
            cleaned = self._add_normalized_columns(cleaned)
        if cleaned.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")
        return cleaned

    def fetch_multiple_tickers(
        self,
        tickers: Iterable[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        provider: str = "yfinance",
        resample_rule: Optional[str] = None,
        normalize: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers and return as ticker->DataFrame."""

        results: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            symbol = ticker.strip().upper()
            if not symbol:
                continue
            try:
                results[symbol] = self.fetch_ticker_data(
                    ticker=symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    provider=provider,
                    resample_rule=resample_rule,
                    normalize=normalize,
                )
                LOGGER.info("Fetched %s rows for %s", len(results[symbol]), symbol)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to fetch data for %s: %s", symbol, exc)
        return results

    @staticmethod
    def fetch_live_price(ticker: str) -> Optional[float]:
        """Fetch latest available live/near-real-time price using yfinance."""

        try:
            tkr = yf.Ticker(ticker)
            fast_info = getattr(tkr, "fast_info", {}) or {}
            for key in ("lastPrice", "last_price", "regularMarketPrice"):
                value = fast_info.get(key)
                if value is not None:
                    return float(value)

            intraday = tkr.history(period="1d", interval="1m")
            if not intraday.empty:
                return float(intraday["Close"].dropna().iloc[-1])
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to fetch live price for %s: %s", ticker, exc)
        return None

    @staticmethod
    def _fetch_yfinance(
        ticker: str,
        start: Optional[str],
        end: Optional[str],
        interval: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance."""

        try:
            data = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"yfinance request failed for {ticker}: {exc}") from exc

        if data.empty:
            raise ValueError(f"yfinance returned empty data for {ticker}.")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    def _fetch_alpha_vantage_daily(self, ticker: str) -> pd.DataFrame:
        """Fetch daily adjusted OHLCV data from Alpha Vantage."""

        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": "full",
            "apikey": self.alpha_vantage_api_key,
        }
        try:
            response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Alpha Vantage request failed for {ticker}: {exc}") from exc

        if "Error Message" in payload:
            raise ValueError(f"Alpha Vantage error for {ticker}: {payload['Error Message']}")
        if "Note" in payload:
            raise ValueError(f"Alpha Vantage rate-limit notice for {ticker}: {payload['Note']}")

        series_key = "Time Series (Daily)"
        if series_key not in payload:
            raise ValueError(
                f"Unexpected Alpha Vantage payload for {ticker}: missing '{series_key}'."
            )

        frame = pd.DataFrame.from_dict(payload[series_key], orient="index")
        rename_map = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj Close",
            "6. volume": "Volume",
        }
        frame = frame.rename(columns=rename_map)
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

    @staticmethod
    def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV market data and enforce a standard schema."""

        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        working = df.copy()

        if not isinstance(working.index, pd.DatetimeIndex):
            working.index = pd.to_datetime(working.index, errors="coerce")
        working = working[~working.index.isna()].sort_index()

        if "Adj Close" not in working.columns and "Close" in working.columns:
            working["Adj Close"] = working["Close"]

        missing = [col for col in required_columns if col not in working.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        num_columns: List[str] = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for col in num_columns:
            if col in working.columns:
                working[col] = pd.to_numeric(working[col], errors="coerce")

        working = working[num_columns].ffill().bfill()
        working = working[working["Volume"] >= 0]
        return working

    @staticmethod
    def _resample_ohlcv(df: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
        """Resample OHLCV data to a new frequency."""

        resampled = (
            df.resample(resample_rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Adj Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
        )
        return resampled

    @staticmethod
    def _add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add min-max normalized columns without mutating raw price columns."""

        data = df.copy()
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            col_min = data[col].min()
            col_max = data[col].max()
            if col_max == col_min:
                data[f"{col}_Norm"] = 0.0
            else:
                data[f"{col}_Norm"] = (data[col] - col_min) / (col_max - col_min)
        return data


def utc_now_iso() -> str:
    """Return UTC ISO-8601 timestamp string."""

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
