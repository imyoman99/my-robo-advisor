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
        static_cfg = self.config.get("STATIC", {}) or {}

        assets = static_cfg.get("ASSETS", {}) or {}
        if isinstance(assets, dict):
            self.tickers = [str(v) for v in assets.values()]
        elif isinstance(assets, list):
            self.tickers = [str(v) for v in assets]
        else:
            self.tickers = []

        # tickers가 비어있으면 안전하게 동작하도록 기본값 1개(dummy) 방지 대신 예외
        if not self.tickers:
            raise ValueError(
                "STATIC.ASSETS가 비어있습니다. config에서 자산을 지정하세요."
            )

        weights_raw = static_cfg.get("WEIGHTS")
        if isinstance(weights_raw, list) and len(weights_raw) == len(self.tickers):
            weights = [float(w) for w in weights_raw]
        else:
            # 길이가 맞지 않거나 없으면 동일가중치
            n = len(self.tickers)
            weights = [1.0 / n for _ in range(n)]
        self.weights = weights

        self.rebalance_freq = str(static_cfg.get("REBALANCE_FREQ", "Monthly"))
        fee = static_cfg.get("FEES", None)
        if fee is None:
            fee = self.config.get("FEES", 0.0)
        self.fee = float(fee)
        self.holdings: Dict[str, float] = {t: 0.0 for t in self.tickers}
        self.pending_shares: Dict[str, float] | None = None

    def rebalance_dates(self, index: pd.Index) -> set[pd.Timestamp]:
        freq = freq_to_pandas(self.rebalance_freq)
        # resample은 Series/DataFrame에만 존재하므로 더미 Series로 처리
        anchor = pd.Series(1, index=pd.DatetimeIndex(index))
        return set(anchor.resample(freq).last().index)

    def _calc_target_shares(self, prices: pd.Series) -> Dict[str, float]:
        nav = calc_nav(prices, self.holdings, self.cash)
        targets = {t: nav * w for t, w in zip(self.tickers, self.weights)}

        target_shares: Dict[str, float] = {}
        for t in self.tickers:
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                target_shares[t] = self.holdings.get(t, 0.0)
                continue
            target_shares[t] = targets[t] / price
        return target_shares

    def _execute_to_shares(
        self, prices: pd.Series, target_shares: Dict[str, float]
    ) -> Dict[str, float]:
        remaining: Dict[str, float] = {}
        for t, target in target_shares.items():
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                remaining[t] = target
                continue
            current = self.holdings.get(t, 0.0)
            delta = target - current
            trade_value = delta * price
            fee = apply_fee(trade_value, self.fee)
            self.cash -= trade_value + fee
            new_shares = current + delta
            self.holdings[t] = new_shares
        return remaining

    def on_day(
        self,
        date: pd.Timestamp,
        close_prices: pd.Series,
        open_prices: pd.Series,
        rebalance_set: set[pd.Timestamp],
    ) -> None:
        if self.pending_shares:
            remaining = self._execute_to_shares(open_prices, self.pending_shares)
            self.pending_shares = remaining or None

        if date in rebalance_set:
            self.pending_shares = self._calc_target_shares(close_prices)

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
