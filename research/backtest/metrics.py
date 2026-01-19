from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (1 + total_return) ** (1 / years) - 1


def mdd(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return drawdown.min()


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    excess = returns - (rf / periods_per_year)
    if excess.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def summary(equity: pd.Series, returns: pd.Series) -> dict:
    return {
        "CAGR": cagr(equity),
        "MDD": mdd(equity),
        "Sharpe": sharpe(returns),
        "TotalReturn": (
            equity.iloc[-1] / equity.iloc[0] - 1 if not equity.empty else 0.0
        ),
    }
