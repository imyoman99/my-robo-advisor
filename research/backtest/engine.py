from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .costs import calc_costs


WeightFunc = Callable[[pd.Timestamp, pd.DataFrame], pd.Series]


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame


def _default_weight_func(date: pd.Timestamp, prices: pd.DataFrame) -> pd.Series:
    cols = prices.columns
    weight = pd.Series(1.0 / len(cols), index=cols)
    return weight


def backtest(
    prices: pd.DataFrame,
    weight_func: Optional[WeightFunc] = None,
    rebalance_freq: str = "M",
    initial_equity: float = 1.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    if prices.empty:
        raise ValueError("prices is empty")

    prices = prices.sort_index()
    returns = prices.pct_change().fillna(0.0)
    weight_func = weight_func or _default_weight_func

    rebalance_dates = prices.resample(rebalance_freq).last().index
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    for dt in rebalance_dates:
        if dt not in prices.index:
            dt = prices.index[prices.index.get_indexer([dt], method="nearest")[0]]
        w = weight_func(dt, prices.loc[:dt])
        weights.loc[dt] = w

    weights = weights.ffill().fillna(0.0)
    costs = calc_costs(weights, fee_bps=fee_bps, slippage_bps=slippage_bps)

    port_returns = (weights.shift(1) * returns).sum(axis=1) - costs
    equity = (1 + port_returns).cumprod() * initial_equity

    return BacktestResult(equity=equity, returns=port_returns, weights=weights)


def backtest_threshold(
    prices: pd.DataFrame,
    weight_func: Optional[WeightFunc] = None,
    rebalance_threshold: float = 0.05,
    initial_equity: float = 1.0,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestResult:
    if prices.empty:
        raise ValueError("prices is empty")
    if rebalance_threshold < 0:
        raise ValueError("rebalance_threshold must be >= 0")

    prices = prices.sort_index()
    returns = prices.pct_change().fillna(0.0)
    weight_func = weight_func or _default_weight_func

    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    last_rebalance_date = prices.index[0]
    last_weights = weight_func(last_rebalance_date, prices.loc[:last_rebalance_date])
    weights.loc[last_rebalance_date] = last_weights

    for dt in prices.index[1:]:
        price_base = prices.loc[last_rebalance_date]
        price_now = prices.loc[dt]

        rel = price_now / price_base
        current_values = last_weights * rel
        if current_values.sum() == 0:
            current_weights = last_weights
        else:
            current_weights = current_values / current_values.sum()

        drift = (current_weights - last_weights).abs().max()
        if drift >= rebalance_threshold:
            last_weights = weight_func(dt, prices.loc[:dt])
            last_rebalance_date = dt
        weights.loc[dt] = last_weights

    weights = weights.ffill().fillna(0.0)
    costs = calc_costs(weights, fee_bps=fee_bps, slippage_bps=slippage_bps)
    port_returns = (weights.shift(1) * returns).sum(axis=1) - costs
    equity = (1 + port_returns).cumprod() * initial_equity

    return BacktestResult(equity=equity, returns=port_returns, weights=weights)
