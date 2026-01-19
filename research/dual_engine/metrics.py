from __future__ import annotations

from typing import Dict

import pandas as pd


def performance_summary(
    equity: pd.Series, returns: pd.Series, trading_days: int = 252
) -> Dict[str, float]:
    if equity.empty:
        return {"CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0, "TotalReturn": 0.0}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    running_max = equity.cummax()
    mdd = (equity / running_max - 1).min()

    if returns.std() == 0 or pd.isna(returns.std()):
        sharpe = 0.0
    else:
        sharpe = (returns.mean() / returns.std()) * (trading_days**0.5)

    return {
        "CAGR": float(cagr),
        "MDD": float(mdd),
        "Sharpe": float(sharpe),
        "TotalReturn": float(total_return),
    }
