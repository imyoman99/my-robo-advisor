from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from core.allocators import normalize_weights
from core.strategies.base import Strategy
from core.strategies.dynamic.daa import DAAStrategy
from core.strategies.dynamic.faa import FAAStrategy
from core.strategies.dynamic.gtaa import GTAAStrategy
from core.strategies.dynamic.vaa import VAAStrategy
from core.strategies.static.all_weather import AllWeatherStrategy
from core.strategies.static.ams import AMSStrategy


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "all_weather": AllWeatherStrategy,
    "ams": AMSStrategy,
    "gtaa": GTAAStrategy,
    "faa": FAAStrategy,
    "vaa": VAAStrategy,
    "daa": DAAStrategy,
}


def build_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {sorted(STRATEGY_REGISTRY)}"
        )
    cls = STRATEGY_REGISTRY[name]
    params = params or {}
    return cls(**params)


@dataclass(frozen=True)
class SleeveConfig:
    name: str
    alloc: float
    params: dict[str, Any]


def build_sleeves(cfg: dict[str, Any]) -> list[SleeveConfig]:
    sleeves_cfg = cfg.get("sleeves", []) or []
    sleeves: list[SleeveConfig] = []
    for sleeve in sleeves_cfg:
        sleeves.append(
            SleeveConfig(
                name=str(sleeve.get("strategy")),
                alloc=float(sleeve.get("alloc", 0.0)),
                params=dict(sleeve.get("params", {}) or {}),
            )
        )
    return sleeves


def combined_weight_func(backtest_cfg: dict[str, Any]):
    sleeves = build_sleeves(backtest_cfg)
    if not sleeves:
        return None

    built = [(build_strategy(s.name, s.params), s.alloc) for s in sleeves]

    def _weight_func(dt: pd.Timestamp, prices: pd.DataFrame) -> pd.Series:
        # 각 슬리브가 같은 prices(전체 유니버스)에 대해 weights 산출
        combined = pd.Series(0.0, index=prices.columns)
        for strat, alloc in built:
            if alloc <= 0:
                continue
            w = strat.generate_weights(prices)
            w = w.reindex(prices.columns).fillna(0.0)
            combined = combined.add(w * alloc, fill_value=0.0)
        return normalize_weights(combined)

    return _weight_func
