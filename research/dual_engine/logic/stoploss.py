from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ..utils import apply_fee
from .types import DynamicState, Selector


@dataclass(frozen=True)
class NoStopLoss:
    def apply(
        self, *, state: DynamicState, prices: pd.Series, ranked: pd.Series
    ) -> DynamicState:
        return state


@dataclass(frozen=True)
class TrailingStopLoss:
    pct: float
    fee: float
    selector: Selector
    slot_count: int

    def apply(
        self, *, state: DynamicState, prices: pd.Series, ranked: pd.Series
    ) -> DynamicState:
        stop_pct = float(self.pct)
        if stop_pct <= 0:
            return state

        cash = float(state.cash)
        holdings = dict(state.holdings)
        high_water = dict(state.high_water)

        for symbol in list(holdings.keys()):
            px = float(prices.get(symbol, 0.0))
            if px <= 0 or pd.isna(px):
                continue

            hwm = high_water.get(symbol, px)
            hwm = max(float(hwm), px)
            high_water[symbol] = hwm

            if px <= hwm * (1 - stop_pct):
                shares = float(holdings.pop(symbol, 0.0))
                trade_value = shares * px
                cash += trade_value - apply_fee(trade_value, float(self.fee))
                high_water.pop(symbol, None)

                # replacement: pick first candidate not held
                candidates: List[str] = self.selector.select(
                    ranked, int(self.slot_count)
                )
                for cand in candidates:
                    if cand not in holdings:
                        holdings[cand] = 0.0
                        cand_px = float(prices.get(cand, 0.0))
                        high_water[cand] = (
                            cand_px if (cand_px > 0 and not pd.isna(cand_px)) else 0.0
                        )
                        break

        return DynamicState(cash=cash, holdings=holdings, high_water=high_water)
