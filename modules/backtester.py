"""Backtesting engine for long-only signal strategy."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestSummary:
    """Backtesting summary metrics."""

    initial_capital: float
    final_portfolio_value: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int


class Backtester:
    """Long-only backtester with full capital allocation per entry."""

    def __init__(self, initial_capital: float = 10_000.0) -> None:
        self.initial_capital = initial_capital

    def run(self, df: pd.DataFrame) -> tuple[pd.DataFrame, BacktestSummary]:
        """Run backtest on dataframe with `Adj Close` and `Signal` columns."""

        if "Adj Close" not in df.columns or "Signal" not in df.columns:
            raise ValueError("Backtester requires 'Adj Close' and 'Signal' columns.")

        data = df.copy()
        data["Signal"] = data["Signal"].astype(int)

        cash = self.initial_capital
        shares = 0.0
        position = 0
        entry_price = 0.0
        trade_returns: list[float] = []

        portfolio_values = []
        positions = []
        trade_action = []

        for idx, row in data.iterrows():
            price = float(row["Adj Close"])
            signal = int(row["Signal"])
            action = 0

            if signal == 1 and position == 0:
                shares = cash / price if price > 0 else 0.0
                cash = 0.0
                position = 1
                entry_price = price
                action = 1
            elif signal == -1 and position == 1:
                cash = shares * price
                shares = 0.0
                position = 0
                action = -1
                trade_return = (price - entry_price) / entry_price if entry_price > 0 else 0.0
                trade_returns.append(trade_return)

            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            positions.append(position)
            trade_action.append(action)

        data["Position"] = positions
        data["Trade_Action"] = trade_action
        data["Portfolio_Value"] = portfolio_values
        data["Strategy_Return"] = data["Portfolio_Value"].pct_change().fillna(0.0)

        if position == 1 and shares > 0:
            last_price = float(data["Adj Close"].iloc[-1])
            final_trade_return = (last_price - entry_price) / entry_price if entry_price > 0 else 0.0
            trade_returns.append(final_trade_return)

        summary = self._compute_metrics(data, trade_returns)
        return data, summary

    def _compute_metrics(self, data: pd.DataFrame, trade_returns: list[float]) -> BacktestSummary:
        """Compute performance metrics from backtest results."""

        final_value = float(data["Portfolio_Value"].iloc[-1])
        total_return = (final_value / self.initial_capital) - 1.0
        daily_returns = data["Strategy_Return"]

        if daily_returns.std(ddof=0) > 0:
            sharpe = (daily_returns.mean() / daily_returns.std(ddof=0)) * np.sqrt(252)
        else:
            sharpe = 0.0

        cummax = data["Portfolio_Value"].cummax()
        drawdown = (data["Portfolio_Value"] - cummax) / cummax
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        wins = sum(1 for r in trade_returns if r > 0)
        trades = len(trade_returns)
        win_rate = (wins / trades) if trades > 0 else 0.0

        return BacktestSummary(
            initial_capital=self.initial_capital,
            final_portfolio_value=final_value,
            total_return_pct=total_return * 100,
            sharpe_ratio=float(sharpe),
            max_drawdown_pct=max_drawdown * 100,
            win_rate_pct=win_rate * 100,
            trades=trades,
        )

    @staticmethod
    def plot_equity_curve(df: pd.DataFrame, title: str = "Strategy Equity Curve") -> None:
        """Plot portfolio value over time."""

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["Portfolio_Value"], label="Portfolio Value", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_signals(df: pd.DataFrame, title: str = "Price with Buy/Sell Signals") -> None:
        """Plot adjusted close with buy/sell markers."""

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["Adj Close"], label="Adj Close", color="tab:blue", linewidth=1.5)

        buys = df[df["Trade_Action"] == 1]
        sells = df[df["Trade_Action"] == -1]

        ax.scatter(buys.index, buys["Adj Close"], marker="^", color="green", s=80, label="Buy")
        ax.scatter(sells.index, sells["Adj Close"], marker="v", color="red", s=80, label="Sell")

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def log_summary(summary: BacktestSummary) -> None:
        """Log backtesting summary in a structured format."""

        LOGGER.info("Backtest Summary")
        LOGGER.info("Initial capital: $%.2f", summary.initial_capital)
        LOGGER.info("Final portfolio value: $%.2f", summary.final_portfolio_value)
        LOGGER.info("Total return: %.2f%%", summary.total_return_pct)
        LOGGER.info("Sharpe ratio: %.3f", summary.sharpe_ratio)
        LOGGER.info("Max drawdown: %.2f%%", summary.max_drawdown_pct)
        LOGGER.info("Win rate: %.2f%%", summary.win_rate_pct)
        LOGGER.info("Completed trades: %s", summary.trades)

