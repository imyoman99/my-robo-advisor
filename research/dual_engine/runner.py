from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from .config import apply_overrides, load_dual_engine_config
from .master import MasterPortfolio
from .metrics import performance_summary


@dataclass(frozen=True)
class DualEngineResult:
    config: Dict[str, Any]
    equity: pd.Series
    returns: pd.Series
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    performance: Dict[str, float]
    benchmark_performance: Dict[str, float]


def run_dual_engine_backtest(
    *,
    config_name: str = "dual_engine",
    config: Optional[Dict[str, Any]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> DualEngineResult:
    cfg = config if config is not None else load_dual_engine_config(config_name)
    cfg = apply_overrides(cfg, start=start, end=end)

    master = MasterPortfolio(cfg)
    equity = master.run()
    returns = equity.pct_change(fill_method=None).fillna(0.0)

    benchmark_ticker = cfg.get("BENCHMARK_TICKER")
    prices = master.prices
    if benchmark_ticker and benchmark_ticker in prices.columns:
        bench = prices[benchmark_ticker].dropna()
        benchmark_equity = (bench / bench.iloc[0]) * float(cfg["INITIAL_CAPITAL"])
        benchmark_returns = benchmark_equity.pct_change(fill_method=None).fillna(0.0)
    else:
        benchmark_equity = pd.Series(dtype=float)
        benchmark_returns = pd.Series(dtype=float)

    perf = performance_summary(equity, returns, int(cfg.get("TRADING_DAYS", 252)))
    bench_perf = performance_summary(
        benchmark_equity, benchmark_returns, int(cfg.get("TRADING_DAYS", 252))
    )

    return DualEngineResult(
        config=cfg,
        equity=equity,
        returns=returns,
        benchmark_equity=benchmark_equity,
        benchmark_returns=benchmark_returns,
        performance=perf,
        benchmark_performance=bench_perf,
    )
