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


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 50,
    window: int | None = 252,
    periods_per_year: int = 252,
    ridge: float = 1e-6,
) -> pd.DataFrame:
    data = returns.dropna()
    if data.empty:
        return pd.DataFrame(columns=["target_return", "volatility", "weights"])
    if window is not None and window > 0:
        data = data.iloc[-window:]

    mu = data.mean() * periods_per_year
    cov = data.cov() * periods_per_year
    n = cov.shape[0]
    if n == 0:
        return pd.DataFrame(columns=["target_return", "volatility", "weights"])

    cov = cov.values + np.eye(n) * ridge
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    mu_vec = mu.values

    A = ones.T @ inv @ ones
    B = ones.T @ inv @ mu_vec
    C = mu_vec.T @ inv @ mu_vec
    denom = A * C - B**2
    if denom == 0:
        return pd.DataFrame(columns=["target_return", "volatility", "weights"])

    target_min = float(mu.min())
    target_max = float(mu.max())
    targets = np.linspace(target_min, target_max, n_points)

    rows: list[dict] = []
    for target in targets:
        g = (C - B * target) / denom
        h = (A * target - B) / denom
        w = inv @ (g * ones + h * mu_vec)
        vol = float(np.sqrt(w.T @ cov @ w))
        rows.append(
            {
                "target_return": float(target),
                "volatility": vol,
                "weights": pd.Series(w, index=data.columns),
            }
        )

    return pd.DataFrame(rows)


def plot_efficient_frontier(frontier: pd.DataFrame, ax=None):
    if frontier.empty:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(frontier["volatility"], frontier["target_return"], marker="o")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    ax.set_title("Efficient Frontier")
    return ax
