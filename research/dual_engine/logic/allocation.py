from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class EqualWeightAllocator:
    def targets(
        self, *, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> Dict[str, float]:
        if not tickers:
            return {}
        w = 1.0 / float(len(tickers))
        return {t: w for t in tickers}


@dataclass(frozen=True)
class SlotWeightAllocator:
    slot_weights: Sequence[float]

    def targets(
        self, *, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> Dict[str, float]:
        if not tickers:
            return {}
        total = float(sum(self.slot_weights)) if self.slot_weights else 1.0
        if total == 0:
            weights = [0.0 for _ in self.slot_weights]
        else:
            weights = [float(w) / total for w in self.slot_weights]
        return {t: w for t, w in zip(tickers, weights)}


@dataclass(frozen=True)
class InverseVolatilityAllocator:
    window: int = 60
    min_vol: float = 1e-8
    max_weight: float | None = None

    def targets(
        self, *, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> Dict[str, float]:
        if not tickers:
            return {}

        window = int(self.window)
        windowed = prices[tickers].loc[:as_of].tail(window + 1)
        returns = windowed.pct_change(fill_method=None).dropna()
        if returns.empty:
            w = 1.0 / float(len(tickers))
            return {t: w for t in tickers}

        vol = (
            returns.std().replace(0.0, float(self.min_vol)).fillna(float(self.min_vol))
        )
        vol = vol.clip(lower=float(self.min_vol))
        inv = 1.0 / vol
        inv = inv.fillna(float(self.min_vol))

        weights = (
            inv / inv.sum()
            if float(inv.sum()) > 0
            else pd.Series(1.0 / len(tickers), index=tickers)
        )

        if self.max_weight is not None:
            cap = float(self.max_weight)
            weights = weights.clip(upper=cap)
            s = float(weights.sum())
            if s > 0:
                weights = weights / s

        return {t: float(weights.get(t, 0.0)) for t in tickers}
