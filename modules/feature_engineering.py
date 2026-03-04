"""Feature engineering for technical indicators and supervised target."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add standard technical indicators to OHLCV dataframe."""

    data = df.copy()

    data["Return"] = data["Adj Close"].pct_change()
    data["SMA_20"] = data["Adj Close"].rolling(window=20, min_periods=20).mean()
    data["SMA_50"] = data["Adj Close"].rolling(window=50, min_periods=50).mean()
    data["EMA_20"] = data["Adj Close"].ewm(span=20, adjust=False).mean()

    delta = data["Adj Close"].diff()
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gains, index=data.index).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(losses, index=data.index).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    data["RSI_14"] = 100 - (100 / (1 + rs))
    data["RSI_14"] = data["RSI_14"].fillna(50.0)

    ema_12 = data["Adj Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Adj Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]

    rolling_std = data["Adj Close"].rolling(window=20, min_periods=20).std()
    data["BB_MIDDLE"] = data["SMA_20"]
    data["BB_UPPER"] = data["BB_MIDDLE"] + 2 * rolling_std
    data["BB_LOWER"] = data["BB_MIDDLE"] - 2 * rolling_std
    data["BB_WIDTH"] = (data["BB_UPPER"] - data["BB_LOWER"]) / data["BB_MIDDLE"]

    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data


def build_target(df: pd.DataFrame, up_threshold: float = 0.01, down_threshold: float = -0.01) -> pd.DataFrame:
    """Build multiclass next-day return target: BUY=1, SELL=-1, HOLD=0."""

    data = df.copy()
    data["Next_Return"] = data["Adj Close"].pct_change().shift(-1)
    data["Target"] = 0
    data.loc[data["Next_Return"] > up_threshold, "Target"] = 1
    data.loc[data["Next_Return"] < down_threshold, "Target"] = -1
    data = data.dropna(subset=["Next_Return"])
    return data


def feature_columns() -> list[str]:
    """Return model feature columns used by the signal engine."""

    return [
        "Return",
        "SMA_20",
        "SMA_50",
        "EMA_20",
        "RSI_14",
        "MACD",
        "MACD_SIGNAL",
        "MACD_HIST",
        "BB_MIDDLE",
        "BB_UPPER",
        "BB_LOWER",
        "BB_WIDTH",
        "Volume",
    ]

