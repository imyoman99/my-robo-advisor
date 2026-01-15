from __future__ import annotations

import pandas as pd


def calc_turnover(weights: pd.DataFrame) -> pd.Series:
    """일별 포트폴리오 회전율"""
    delta = weights.diff().abs().sum(axis=1)
    return delta.fillna(0.0)


def calc_costs(
    weights: pd.DataFrame,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.Series:
    total_bps = fee_bps + slippage_bps
    turnover = calc_turnover(weights)
    costs = turnover * (total_bps / 10000.0)
    return costs
