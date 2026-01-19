from __future__ import annotations

import pandas as pd


def momentum_score(prices: pd.DataFrame, window: int = 126) -> pd.Series:
    if prices.empty:
        return pd.Series(dtype=float)
    return prices.pct_change(window).iloc[-1]


def volatility_score(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    return returns.rolling(window).std().iloc[-1]


def value_proxy_score(prices: pd.DataFrame) -> pd.Series:
    """값(밸류) 팩터 대용 스코어(스켈레톤).

    실제 밸류 지표(PER/PBR 등)가 없으므로 가격 역수 기반의 간단 proxy를 사용합니다.
    별도 밸류 데이터가 주입되면 이 함수를 대체하세요.
    """
    if prices.empty:
        return pd.Series(dtype=float)
    last = prices.iloc[-1]
    inv = 1.0 / last.replace(0, pd.NA)
    return inv.fillna(0.0)
