from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.rebalancing.methods import (
    AbsoluteScoreMethod,
    MethodContext,
    RankBasedMethod,
    RebalanceMethod,
    RiskParityMethod,
)
from core.rebalancing.triggers import (
    RebalanceContext,
    RebalanceTrigger,
    ThresholdTrigger,
    TimeTrigger,
)
from core.scoring import FactorConfig, FactorScorer
from core.types import BacktestResult


@dataclass(frozen=True)
class BacktestConfig:
    scorer: FactorScorer
    cash_asset: str
    trigger: RebalanceTrigger
    method: RebalanceMethod


def build_scorer(cfg: dict[str, Any]) -> FactorScorer:
    factors_cfg = cfg.get("factors", [])
    factors: list[FactorConfig] = []
    for f in factors_cfg:
        factors.append(
            FactorConfig(
                name=str(f.get("name")),
                weight=float(f.get("weight", 1.0)),
                params=dict(f.get("params", {}) or {}),
            )
        )
    normalize = cfg.get("normalize", "zscore")
    return FactorScorer(factors, normalize=normalize)


def build_trigger(cfg: dict[str, Any]) -> RebalanceTrigger:
    t = (cfg.get("type") or "time").lower()
    if t == "threshold":
        return ThresholdTrigger(threshold=float(cfg.get("threshold", 0.05)))
    return TimeTrigger(freq=str(cfg.get("freq", "M")))


def build_method(cfg: dict[str, Any]) -> RebalanceMethod:
    m = (cfg.get("type") or "absolute_score").lower()
    if m == "rank":
        return RankBasedMethod(top_n=int(cfg.get("top_n", 5)))
    if m == "risk_parity":
        return RiskParityMethod(window=int(cfg.get("window", 60)))
    return AbsoluteScoreMethod()


def _ensure_cash_prices(prices: pd.DataFrame, cash_asset: str) -> pd.DataFrame:
    if cash_asset in prices.columns:
        return prices
    cash = pd.Series(1.0, index=prices.index, name=cash_asset)
    return prices.join(cash)


def run_backtest(prices: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    if prices.empty:
        raise ValueError("prices is empty")

    prices = prices.sort_index()
    prices = _ensure_cash_prices(prices, cfg.cash_asset)

    cfg.trigger.prepare(prices)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    scorer = cfg.scorer

    last_rebalance_date = prices.index[0]
    last_target = pd.Series(0.0, index=prices.columns)

    for dt in prices.index:
        if dt == prices.index[0]:
            score = scorer.score(prices.loc[:dt]).scores
            method_ctx = MethodContext(
                prices=prices.loc[:dt], scores=score, cash_asset=cfg.cash_asset
            )
            last_target = (
                cfg.method.compute_target_weights(method_ctx)
                .reindex(prices.columns)
                .fillna(0.0)
            )
            weights.loc[dt] = last_target
            last_rebalance_date = dt
            continue

        price_base = prices.loc[last_rebalance_date]
        price_now = prices.loc[dt]
        rel = price_now / price_base
        current_values = last_target * rel
        current_weights = (
            current_values / current_values.sum()
            if current_values.sum() != 0
            else last_target
        )

        score = scorer.score(prices.loc[:dt]).scores
        context = RebalanceContext(
            date=dt,
            prices=prices.loc[:dt],
            last_rebalance_date=last_rebalance_date,
            last_target_weights=last_target,
            current_weights=current_weights,
        )

        if cfg.trigger.should_rebalance(context):
            method_ctx = MethodContext(
                prices=prices.loc[:dt], scores=score, cash_asset=cfg.cash_asset
            )
            last_target = (
                cfg.method.compute_target_weights(method_ctx)
                .reindex(prices.columns)
                .fillna(0.0)
            )
            last_rebalance_date = dt

        weights.loc[dt] = last_target

    returns = prices.pct_change().fillna(0.0)
    port_returns = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
    equity = (1 + port_returns).cumprod()

    return BacktestResult(equity=equity, returns=port_returns, weights=weights)
