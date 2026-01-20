from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .data_loader import load_local_parquet_opens, load_local_parquet_prices


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
        data_cfg = self.config.get("DATA", {}) or {}
        data_root = "data"
        if isinstance(data_cfg, dict):
            data_root = data_cfg.get("root") or data_cfg.get("base_dir") or data_root

        data_root_path = Path(data_root)
        if not data_root_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            data_root_path = (project_root / data_root_path).resolve()
        if data_root_path.name not in {"processed", "raw"}:
            processed_dir = data_root_path / "processed"
            if processed_dir.exists():
                data_root_path = processed_dir

        prices = load_local_parquet_prices(
            self.all_tickers(),
            data_root=str(data_root_path),
            start=self.config.get("START_DATE"),
            end=self.config.get("END_DATE"),
            ffill=bool(self.config.get("FFILL", True)),
        )
        return prices.dropna(how="all")

    def load_open_prices(self) -> pd.DataFrame:
        data_cfg = self.config.get("DATA", {}) or {}
        data_root = "data"
        if isinstance(data_cfg, dict):
            data_root = data_cfg.get("root") or data_cfg.get("base_dir") or data_root

        data_root_path = Path(data_root)
        if not data_root_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            data_root_path = (project_root / data_root_path).resolve()
        if data_root_path.name not in {"processed", "raw"}:
            processed_dir = data_root_path / "processed"
            if processed_dir.exists():
                data_root_path = processed_dir

        prices = load_local_parquet_opens(
            self.all_tickers(),
            data_root=str(data_root_path),
            start=self.config.get("START_DATE"),
            end=self.config.get("END_DATE"),
            ffill=False,
        )
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
