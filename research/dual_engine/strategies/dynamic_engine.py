from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from ..universe import UniverseLoader
from ..logic.factory import build_dynamic_components
from ..logic.types import DynamicState
from ..utils import apply_fee, calc_nav, freq_to_pandas, month_day_in_season


@dataclass
class DynamicStrategy:
    config: Dict[str, Any]
    cash: float

    def __post_init__(self) -> None:
        self.seasons = self.config["DYNAMIC"]["SEASONS"]
        self.logic = self.config["DYNAMIC"]["LOGIC"]
        self.rebalance_freq = str(self.config["DYNAMIC"]["REBALANCE_FREQ"])
        self.fee = float(self.config.get("FEES", 0.0))

        self.holdings: Dict[str, float] = {}
        self.high_water: Dict[str, float] = {}
        self.active_season: Optional[str] = None

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

    def _rebalance_to_targets(
        self, prices: pd.Series, targets: Dict[str, float]
    ) -> None:
        nav = calc_nav(prices, self.holdings, self.cash)

        for t, w in targets.items():
            price = float(prices.get(t, 0.0))
            if price <= 0 or pd.isna(price):
                continue
            target_value = nav * float(w)
            target_shares = target_value / price
            current_shares = self.holdings.get(t, 0.0)
            delta = target_shares - current_shares
            trade_value = delta * price
            fee = apply_fee(trade_value, self.fee)
            self.cash -= trade_value + fee
            self.holdings[t] = current_shares + delta
            self.high_water[t] = max(self.high_water.get(t, price), price)

        for t in list(self.holdings.keys()):
            if t not in targets:
                price = float(prices.get(t, 0.0))
                if price <= 0 or pd.isna(price):
                    # 가격이 비정상이면 청산 계산을 스킵 (다음 유효 시점에 처리)
                    continue
                shares = self.holdings.pop(t, 0.0)
                trade_value = shares * price
                fee = apply_fee(trade_value, self.fee)
                self.cash += trade_value - fee
                self.high_water.pop(t, None)

    def on_day(
        self,
        date: pd.Timestamp,
        prices: pd.Series,
        prices_df: pd.DataFrame,
        rebalance_set: set[pd.Timestamp],
        loader: UniverseLoader,
    ) -> None:
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
        state = self.stop_loss.apply(state=state, prices=prices, ranked=ranked)
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
            self._rebalance_to_targets(prices, targets)

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
