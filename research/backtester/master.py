from __future__ import annotations

from typing import Any, Dict, Optional, cast

import pandas as pd

from .strategies import DynamicStrategy, StaticStrategy
from .universe import UniverseLoader
from .utils import freq_to_pandas, to_date_index


class MasterPortfolio:
    def __init__(
        self, config: Dict[str, Any], *, prices: Optional[pd.DataFrame] = None
    ) -> None:
        self.config = config
        self.loader = UniverseLoader(config)
        self.prices = prices if prices is not None else self.loader.load_prices()
        self.prices.index = pd.DatetimeIndex(to_date_index(self.prices))

        self.static_engine = StaticStrategy(config, cash=self._initial_static())
        self.dynamic_engine = DynamicStrategy(config, cash=self._initial_dynamic())

        self.master_rebalance_dates = set(
            self.prices.resample(freq_to_pandas(config["MASTER"]["REBALANCE_FREQ"]))
            .last()
            .index
        )
        self.static_rebalance_dates = self.static_engine.rebalance_dates(
            self.prices.index
        )
        self.dynamic_rebalance_dates = self.dynamic_engine.rebalance_dates(
            self.prices.index
        )

        self.equity: pd.Series = pd.Series(dtype=float)
        self.static_equity: pd.Series = pd.Series(dtype=float)
        self.dynamic_equity: pd.Series = pd.Series(dtype=float)

    def _initial_static(self) -> float:
        return float(
            self.config["INITIAL_CAPITAL"] * self.config["MASTER"]["STATIC_RATIO"]
        )

    def _initial_dynamic(self) -> float:
        return float(
            self.config["INITIAL_CAPITAL"] * self.config["MASTER"]["DYNAMIC_RATIO"]
        )

    def _rebalance_master(self, date: pd.Timestamp, prices: pd.Series) -> None:
        total_nav = self.static_engine.nav(prices) + self.dynamic_engine.nav(prices)
        target_static = total_nav * float(self.config["MASTER"]["STATIC_RATIO"])
        target_dynamic = total_nav * float(self.config["MASTER"]["DYNAMIC_RATIO"])

        static_nav = self.static_engine.nav(prices)
        dynamic_nav = self.dynamic_engine.nav(prices)

        if static_nav > target_static:
            delta = static_nav - target_static
            self.static_engine.withdraw_cash(delta, prices)
            self.dynamic_engine.add_cash(delta)
        elif dynamic_nav > target_dynamic:
            delta = dynamic_nav - target_dynamic
            self.dynamic_engine.withdraw_cash(delta, prices)
            self.static_engine.add_cash(delta)

    def run(self) -> pd.Series:
        equity = []
        static_equity = []
        dynamic_equity = []
        last_nav: float | None = None
        for date, row in self.prices.iterrows():
            date_ts = pd.Timestamp(cast(Any, date))
            self.static_engine.on_day(date_ts, row, self.static_rebalance_dates)
            self.dynamic_engine.on_day(
                date_ts,
                row,
                self.prices,
                self.dynamic_rebalance_dates,
                self.loader,
            )

            if date_ts in self.master_rebalance_dates:
                self._rebalance_master(date_ts, row)

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
