from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from ..utils import apply_fee, calc_nav, freq_to_pandas


@dataclass
class StaticStrategy:
    config: Dict[str, Any]
    cash: float

    def __post_init__(self) -> None:
        self.tickers: List[str] = list(self.config["STATIC"]["ASSETS"].values())
        self.weights: List[float] = list(self.config["STATIC"]["WEIGHTS"])
        self.rebalance_freq: str = str(self.config["STATIC"]["REBALANCE_FREQ"])
        self.fee: float = float(self.config.get("FEES", 0.0))
        self.holdings: Dict[str, float] = {t: 0.0 for t in self.tickers}

    def rebalance_dates(self, index: pd.Index) -> set[pd.Timestamp]:
        freq = freq_to_pandas(self.rebalance_freq)
        # resample은 Series/DataFrame에만 존재하므로 더미 Series로 처리
        anchor = pd.Series(1, index=pd.DatetimeIndex(index))
        return set(anchor.resample(freq).last().index)

    def rebalance(self, prices: pd.Series) -> None:
        nav = calc_nav(prices, self.holdings, self.cash)
        targets = {t: nav * w for t, w in zip(self.tickers, self.weights)}

        for t in self.tickers:
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                continue
            target_shares = targets[t] / price
            delta = target_shares - self.holdings.get(t, 0.0)
            trade_value = delta * price
            fee = apply_fee(trade_value, self.fee)
            self.cash -= trade_value + fee
            self.holdings[t] = self.holdings.get(t, 0.0) + delta

    def on_day(
        self, date: pd.Timestamp, prices: pd.Series, rebalance_set: set[pd.Timestamp]
    ) -> None:
        if date in rebalance_set:
            self.rebalance(prices)

    def nav(self, prices: pd.Series) -> float:
        return calc_nav(prices, self.holdings, self.cash)

    def add_cash(self, amount: float) -> None:
        self.cash += amount

    def withdraw_cash(self, amount: float, prices: pd.Series) -> None:
        if amount <= 0:
            return
        nav = self.nav(prices)
        if nav <= 0:
            return
        ratio = min(amount / nav, 1.0)
        for t in list(self.holdings.keys()):
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                continue
            sell_shares = self.holdings[t] * ratio
            trade_value = sell_shares * price
            fee = apply_fee(trade_value, self.fee)
            self.cash += trade_value - fee
            self.holdings[t] -= sell_shares
