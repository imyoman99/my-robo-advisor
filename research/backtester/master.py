from __future__ import annotations

from typing import Any, Dict, Optional, cast

import pandas as pd

from .strategies import DynamicStrategy, StaticStrategy
from .universe import UniverseLoader
from .utils import to_date_index


class MasterPortfolio:
    def __init__(
        self, config: Dict[str, Any], *, prices: Optional[pd.DataFrame] = None
    ) -> None:
        self.config = config
        self.loader = UniverseLoader(config)
        self.full_prices = prices if prices is not None else self.loader.load_prices()
        self.full_prices.index = pd.DatetimeIndex(to_date_index(self.full_prices))
        self.full_open_prices = self.loader.load_open_prices()
        self.full_open_prices.index = pd.DatetimeIndex(
            to_date_index(self.full_open_prices)
        )
        self.full_open_prices = self.full_open_prices.reindex(
            index=self.full_prices.index, columns=self.full_prices.columns
        )

        start = self.config.get("START_DATE")
        end = self.config.get("END_DATE")
        if start or end:
            self.prices = self.full_prices.loc[start:end]
            self.open_prices = self.full_open_prices.loc[start:end]
        else:
            self.prices = self.full_prices
            self.open_prices = self.full_open_prices

        self.static_engine = StaticStrategy(config, cash=self._initial_static())
        self.dynamic_engine = DynamicStrategy(config, cash=self._initial_dynamic())

        self.equity: pd.Series = pd.Series(dtype=float)
        self.static_equity: pd.Series = pd.Series(dtype=float)
        self.dynamic_equity: pd.Series = pd.Series(dtype=float)

    def _initial_static(self) -> float:
        static_cfg = self.config.get("STATIC", {}) or {}
        dynamic_cfg = self.config.get("DYNAMIC", {}) or {}
        static_ratio = static_cfg.get("RATIO")
        dynamic_ratio = dynamic_cfg.get("RATIO")
        if static_ratio is None and dynamic_ratio is None:
            static_ratio = 0.5
            dynamic_ratio = 0.5
        elif static_ratio is None:
            dynamic_ratio = float(dynamic_ratio)
            static_ratio = max(0.0, 1.0 - dynamic_ratio)
        elif dynamic_ratio is None:
            static_ratio = float(static_ratio)
            dynamic_ratio = max(0.0, 1.0 - static_ratio)
        else:
            static_ratio = float(static_ratio)
            dynamic_ratio = float(dynamic_ratio)
            total = static_ratio + dynamic_ratio
            if total > 0:
                static_ratio /= total

        return float(self.config["INITIAL_CAPITAL"] * float(static_ratio))

    def _initial_dynamic(self) -> float:
        static_cfg = self.config.get("STATIC", {}) or {}
        dynamic_cfg = self.config.get("DYNAMIC", {}) or {}
        static_ratio = static_cfg.get("RATIO")
        dynamic_ratio = dynamic_cfg.get("RATIO")
        if static_ratio is None and dynamic_ratio is None:
            static_ratio = 0.5
            dynamic_ratio = 0.5
        elif static_ratio is None:
            dynamic_ratio = float(dynamic_ratio)
            static_ratio = max(0.0, 1.0 - dynamic_ratio)
        elif dynamic_ratio is None:
            static_ratio = float(static_ratio)
            dynamic_ratio = max(0.0, 1.0 - static_ratio)
        else:
            static_ratio = float(static_ratio)
            dynamic_ratio = float(dynamic_ratio)
            total = static_ratio + dynamic_ratio
            if total > 0:
                dynamic_ratio /= total

        return float(self.config["INITIAL_CAPITAL"] * float(dynamic_ratio))

    def run(self) -> pd.Series:
        equity = []
        static_equity = []
        dynamic_equity = []
        last_nav: float | None = None
        for date, row in self.prices.iterrows():
            date_ts = pd.Timestamp(cast(Any, date))
            open_row = (
                self.open_prices.loc[date_ts]
                if date_ts in self.open_prices.index
                else row
            )

            self.static_engine.on_day(date_ts, row, open_row, set())
            self.dynamic_engine.on_day(
                date_ts,
                row,
                open_row,
                self.full_prices,
                self.full_open_prices,
                set(),
                self.loader,
            )

            static_nav = self.static_engine.nav(row)
            dynamic_nav = self.dynamic_engine.nav(row)
            total_nav = static_nav + dynamic_nav
            if (
                not pd.notna(total_nav)
                or total_nav == float("inf")
                or total_nav == float("-inf")
            ):
                if last_nav is None:
                    continue
                total_nav = last_nav
            last_nav = float(total_nav)
            equity.append((date_ts, total_nav))
            static_equity.append((date_ts, static_nav))
            dynamic_equity.append((date_ts, dynamic_nav))

        self.equity = pd.Series({d: v for d, v in equity}).sort_index()
        self.static_equity = pd.Series({d: v for d, v in static_equity}).sort_index()
        self.dynamic_equity = pd.Series({d: v for d, v in dynamic_equity}).sort_index()
        return self.equity
