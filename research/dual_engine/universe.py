from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from infra.data_manager.downloader import load_ohlcv_bulk


class UniverseLoader:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def _dynamic_tickers(self) -> List[str]:
        tickers: List[str] = []
        for season in self.config["DYNAMIC"]["SEASONS"]:
            tickers.extend(season.get("tickers", []) or [])
        return list(dict.fromkeys(tickers))

    def _static_tickers(self) -> List[str]:
        return list(self.config["STATIC"]["ASSETS"].values())

    def all_tickers(self) -> List[str]:
        tickers = self._static_tickers() + self._dynamic_tickers()
        if self.config.get("BENCHMARK_TICKER"):
            tickers.append(self.config["BENCHMARK_TICKER"])
        return list(dict.fromkeys(tickers))

    def load_prices(self) -> pd.DataFrame:
        data = load_ohlcv_bulk(
            self.all_tickers(),
            start=self.config.get("START_DATE"),
            end=self.config.get("END_DATE"),
            source=self.config["DATA"].get("source", "fdr"),
            use_cache=self.config["DATA"].get("use_cache", True),
            cache_dir=self.config["DATA"].get("cache_dir", "data/raw"),
            preprocess=self.config["DATA"].get("preprocess", True),
        )
        prices = pd.concat({k: v["close"] for k, v in data.items()}, axis=1)
        if self.config.get("FFILL", True):
            prices = prices.ffill()
        return prices.dropna(how="all")

    def correlation_filter(
        self,
        prices: pd.DataFrame,
        tickers: List[str],
        as_of: pd.Timestamp,
    ) -> List[str]:
        logic = self.config["DYNAMIC"]["LOGIC"]
        threshold = float(logic.get("CORRELATION_THRESHOLD", 0.0))
        window = int(self.config["DYNAMIC"].get("CORRELATION_WINDOW", 60))

        subset = prices[tickers].loc[:as_of].tail(window)
        returns = subset.pct_change().dropna()
        if returns.empty or len(tickers) <= 1:
            return tickers

        corr = returns.corr()
        avg_corr = corr.mean(axis=1)
        filtered = avg_corr[avg_corr >= threshold].index.tolist()
        return filtered if filtered else tickers
