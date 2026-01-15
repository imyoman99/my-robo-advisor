from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_weights(weights: pd.Series) -> pd.Series:
    w = weights.copy()
    total = w.sum()
    if total == 0:
        return w
    return w / total


def equal_weight(assets: list[str]) -> pd.Series:
    if not assets:
        return pd.Series(dtype=float)
    w = pd.Series(1.0 / len(assets), index=assets)
    return w


def inverse_vol_weights(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    vol = returns.rolling(window).std().iloc[-1]
    vol = vol.replace(0, np.nan)
    inv = 1 / vol
    return normalize_weights(inv.fillna(0.0))


def risk_parity_weights(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    return inverse_vol_weights(returns, window=window)


def gmvp_weights(
    returns: pd.DataFrame,
    window: int = 60,
    ridge: float = 1e-6,
    long_only: bool = True,
) -> pd.Series:
    data = returns.dropna()
    if data.empty:
        return pd.Series(dtype=float)
    if window is not None and window > 0:
        data = data.iloc[-window:]

    cov = data.cov()
    n = cov.shape[0]
    if n == 0:
        return pd.Series(dtype=float)

    cov = cov.values + np.eye(n) * ridge
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    denom = ones.T @ inv @ ones
    if denom == 0:
        return equal_weight(list(data.columns))

    w = (inv @ ones) / denom
    weights = pd.Series(w, index=data.columns)
    if long_only:
        weights = weights.clip(lower=0.0)
    if weights.sum() == 0:
        return equal_weight(list(data.columns))
    return normalize_weights(weights)


def cap_weights(weights: pd.Series, cap: float = 0.3) -> pd.Series:
    w = weights.clip(upper=cap)
    return normalize_weights(w)


def apply_signal(weights: pd.Series, signal: pd.Series) -> pd.Series:
    aligned = weights.mul(signal, fill_value=0.0)
    return normalize_weights(aligned)


def rank_series(values: pd.Series, ascending: bool = False) -> pd.Series:
    return values.rank(ascending=ascending, method="first")


def select_top_n(values: pd.Series, top_n: int, ascending: bool = False) -> list[str]:
    ranked = values.sort_values(ascending=ascending)
    return list(ranked.head(top_n).index)


def single_asset_weight(asset: str, universe: list[str]) -> pd.Series:
    weights = pd.Series(0.0, index=universe)
    if asset in weights.index:
        weights.loc[asset] = 1.0
    return weights


def build_weights(targets: dict[str, float], universe: list[str]) -> pd.Series:
    weights = pd.Series(0.0, index=universe)
    for asset, weight in targets.items():
        if asset in weights.index:
            weights.loc[asset] = weight
    return normalize_weights(weights)
