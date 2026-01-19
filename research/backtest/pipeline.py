from __future__ import annotations

from typing import Any

import pandas as pd

from core.engine import (
    BacktestConfig,
    build_method,
    build_scorer,
    build_trigger,
    run_backtest,
)
from infra.data_manager.downloader import load_ohlcv_bulk


def resolve_universe(
    backtest_cfg: dict[str, Any], universe_cfg: dict[str, Any]
) -> list[str]:
    universe_key = backtest_cfg.get("universe_key")
    if universe_key:
        universe = universe_cfg.get(universe_key, [])
    else:
        universe = backtest_cfg.get("universe", [])
    if not universe:
        raise ValueError("backtest.yaml에 universe 또는 universe_key가 비어 있습니다.")
    return list(universe)


def load_prices(
    universe: list[str],
    data_cfg: dict[str, Any],
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    data = load_ohlcv_bulk(
        universe,
        start=start,
        end=end,
        source=data_cfg.get("source", "fdr"),
        use_cache=data_cfg.get("use_cache", True),
        cache_dir=data_cfg.get("cache_dir", "data/raw"),
        preprocess=data_cfg.get("preprocess", True),
    )
    return pd.concat({k: v["close"] for k, v in data.items()}, axis=1).dropna()


def build_backtest_config(backtest_cfg: dict[str, Any]) -> BacktestConfig:
    scoring_cfg = backtest_cfg.get("scoring", {})
    scorer = build_scorer(scoring_cfg)
    cash_asset = scoring_cfg.get("cash_asset", "CASH")
    trigger = build_trigger(backtest_cfg.get("rebalancing", {}).get("trigger", {}))
    method = build_method(backtest_cfg.get("rebalancing", {}).get("method", {}))
    return BacktestConfig(
        scorer=scorer, cash_asset=cash_asset, trigger=trigger, method=method
    )


def run_cio_backtest(
    backtest_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    universe_cfg: dict[str, Any],
    start: str | None,
    end: str | None,
):
    universe = resolve_universe(backtest_cfg, universe_cfg)
    prices = load_prices(universe, data_cfg, start, end)
    cfg = build_backtest_config(backtest_cfg)
    return run_backtest(prices, cfg)
