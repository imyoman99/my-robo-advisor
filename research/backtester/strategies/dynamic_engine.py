from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..universe import UniverseLoader
from core.engine import build_dynamic_components
from core.types import DynamicState
from ..utils import apply_fee, calc_nav, freq_to_pandas, month_day_in_season


@dataclass
class DynamicStrategy:
    config: Dict[str, Any]
    cash: float

    def __post_init__(self) -> None:
        self.seasons = self.config["DYNAMIC"]["SEASONS"]
        self.logic = self.config["DYNAMIC"]["LOGIC"]
        self.rebalance_freq = str(self.config["DYNAMIC"]["REBALANCE_FREQ"])
        dynamic_cfg = self.config.get("DYNAMIC", {}) or {}
        fee = dynamic_cfg.get("FEES", None)
        if fee is None:
            fee = self.config.get("FEES", 0.0)
        self.fee = float(fee)

        self.holdings: Dict[str, float] = {}
        self.high_water: Dict[str, float] = {}
        self.active_season: Optional[str] = None
        self.pending_shares: Dict[str, float] | None = None

        self.slot_count = self._infer_slot_count()
        (
            self.correlation_filter,
            self.momentum_ranker,
            self.selector,
            self.allocator,
            self.stop_loss,
        ) = build_dynamic_components(
            self.config, fee=self.fee, slot_count=self.slot_count
        )

    def rebalance_dates(self, index: pd.Index) -> set[pd.Timestamp]:
        freq = freq_to_pandas(self.rebalance_freq)
        anchor = pd.Series(1, index=pd.DatetimeIndex(index))
        return set(anchor.resample(freq).last().index)

    def _current_season(self, date: pd.Timestamp) -> dict:
        for season in self.seasons:
            if month_day_in_season(date, season["start_md"], season["end_md"]):
                return season
        return self.seasons[0]

    def _infer_slot_count(self) -> int:
        # 플러그인 스키마
        alloc_cfg = self.logic.get("allocator", {}) or {}
        if isinstance(alloc_cfg, dict):
            slot_weights = alloc_cfg.get("slot_weights")
            if isinstance(slot_weights, list) and slot_weights:
                return len(slot_weights)

        # 레거시 스키마
        legacy_weights = self.logic.get("SLOT_WEIGHTS")
        if isinstance(legacy_weights, list) and legacy_weights:
            return len(legacy_weights)

        return 4

    def _calc_target_shares(
        self, prices: pd.Series, targets: Dict[str, float]
    ) -> Dict[str, float]:
        nav = calc_nav(prices, self.holdings, self.cash)
        target_shares: Dict[str, float] = {}

        universe = set(self.holdings.keys()) | set(targets.keys())
        for t in universe:
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                target_shares[t] = self.holdings.get(t, 0.0)
                continue
            w = float(targets.get(t, 0.0))
            target_value = nav * w
            target_shares[t] = target_value / price
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
            if new_shares <= 0:
                self.holdings.pop(t, None)
                self.high_water.pop(t, None)
            else:
                self.holdings[t] = new_shares
                self.high_water[t] = max(self.high_water.get(t, price), price)
        return remaining

    def on_day(
        self,
        date: pd.Timestamp,
        close_prices: pd.Series,
        open_prices: pd.Series,
        prices_df: pd.DataFrame,
        rebalance_set: set[pd.Timestamp],
        loader: UniverseLoader,
    ) -> None:
        if self.pending_shares:
            remaining = self._execute_to_shares(open_prices, self.pending_shares)
            self.pending_shares = remaining or None

        season = self._current_season(date)
        if self.active_season != season["name"]:
            self.active_season = season["name"]
            rebalance_set = rebalance_set | {date}

        tickers = season.get("tickers", []) or []
        if not tickers:
            return

        filtered = self.correlation_filter.filter(prices_df, tickers, date)
        ranked = self.momentum_ranker.rank(prices_df, filtered, date)
        if ranked.empty:
            return

        state = DynamicState(
            cash=self.cash, holdings=self.holdings, high_water=self.high_water
        )
        state = self.stop_loss.apply(state=state, prices=close_prices, ranked=ranked)
        self.cash, self.holdings, self.high_water = (
            state.cash,
            state.holdings,
            state.high_water,
        )

        if date in rebalance_set:
            selected = self.selector.select(ranked, self.slot_count)
            targets = self.allocator.targets(
                prices=prices_df, tickers=selected, as_of=date
            )
            self.pending_shares = self._calc_target_shares(close_prices, targets)

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
