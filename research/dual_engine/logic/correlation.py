from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class NoCorrelationFilter:
    def filter(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> List[str]:
        return tickers


@dataclass(frozen=True)
class AverageCorrelationFilter:
    threshold: float = 0.0
    window: int = 60

    def filter(
        self, prices: pd.DataFrame, tickers: List[str], as_of: pd.Timestamp
    ) -> List[str]:
        subset = prices[tickers].loc[:as_of].tail(int(self.window))
        returns = subset.pct_change(fill_method=None).dropna()
        if returns.empty or len(tickers) <= 1:
            return tickers

        corr = returns.corr()
        avg_corr = corr.mean(axis=1)
        filtered = avg_corr[avg_corr >= float(self.threshold)].index.tolist()
        return filtered if filtered else tickers
