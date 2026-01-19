from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class SimpleMomentumRanker:
    window: int = 60

    def rank(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> pd.Series:
        window = int(self.window)
        windowed = prices[tickers].loc[:as_of].tail(window + 1)
        if windowed.empty:
            return pd.Series(dtype=float)
        mom = windowed.pct_change(window, fill_method=None).iloc[-1]
        return mom.sort_values(ascending=False)


@dataclass(frozen=True)
class VolAdjustedMomentumRanker:
    window: int = 60
    vol_window: int | None = None
    min_vol: float = 1e-8

    def rank(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> pd.Series:
        w = int(self.window)
        vw = int(self.vol_window) if self.vol_window is not None else w
        windowed = prices[tickers].loc[:as_of].tail(max(w, vw) + 1)
        if windowed.empty:
            return pd.Series(dtype=float)

        rets = windowed.pct_change(fill_method=None)
        mom = windowed.pct_change(w, fill_method=None).iloc[-1]
        vol = rets.tail(vw).std().clip(lower=float(self.min_vol))

        score = (mom / vol).fillna(0.0)
        return score.sort_values(ascending=False)
