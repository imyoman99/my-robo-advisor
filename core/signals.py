from __future__ import annotations

import numpy as np
import pandas as pd


def calc_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return prices.pct_change(periods=periods)


def rolling_vol(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return returns.rolling(window).std()


def sma(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return prices.rolling(window).mean()


def ema(prices: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    return prices.ewm(span=span, adjust=False).mean()


def momentum(prices: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    return prices.pct_change(window)


def trend_signal(prices: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    ma = sma(prices, window)
    return (prices > ma).astype(float)


def cross_over_signal(short_ma: pd.DataFrame, long_ma: pd.DataFrame) -> pd.DataFrame:
    return (short_ma > long_ma).astype(float)


def rank_signal(values: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rank = values.rank(axis=1, ascending=False, method="first")
    return (rank <= top_n).astype(float)


def zscore(values: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    mean = values.rolling(window).mean()
    std = values.rolling(window).std()
    return (values - mean) / std.replace(0, np.nan)


def weighted_momentum(
    prices: pd.DataFrame,
    windows: tuple[int, ...] = (1, 3, 6, 12),
    weights: tuple[float, ...] = (12, 4, 2, 1),
    trading_days_per_month: int = 21,
) -> pd.DataFrame:
    if len(windows) != len(weights):
        raise ValueError("windows and weights length must match")
    score = prices.copy() * 0
    for m, w in zip(windows, weights):
        ret = prices.pct_change(m * trading_days_per_month)
        score = score + (ret * w)
    return score


def average_momentum_score(
    prices: pd.DataFrame,
    months: int = 12,
    trading_days_per_month: int = 21,
) -> pd.DataFrame:
    scores: list[pd.DataFrame] = []
    for m in range(1, months + 1):
        ret = prices.pct_change(m * trading_days_per_month)
        scores.append((ret > 0).astype(float))
    if not scores:
        return prices.copy() * 0
    total = scores[0]
    for s in scores[1:]:
        total = total + s
    return total / float(months)


def volatility_score(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    return returns.rolling(window).std().iloc[-1]


def correlation_sum(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    windowed = returns.tail(window)
    corr = windowed.corr()
    return corr.sum(axis=1)
