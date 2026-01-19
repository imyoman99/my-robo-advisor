from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .cache import read_cache, write_cache
from .preprocess import preprocess_ohlcv


def _load_csv(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    candidates = [
        Path("data/processed") / f"{symbol}.csv",
        Path("data/raw") / f"{symbol}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"CSV not found for {symbol}")
    df = pd.read_csv(path)
    if start or end:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
        if start:
            df = df.loc[df.index >= pd.to_datetime(start)]
        if end:
            df = df.loc[df.index <= pd.to_datetime(end)]
    return df


def _load_fdr(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FinanceDataReader가 필요합니다. `pip install finance-datareader` 실행 후 재시도하세요."
        ) from exc
    return fdr.DataReader(symbol, start, end)


def load_ohlcv_bulk(
    universe: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: str = "fdr",
    use_cache: bool = True,
    cache_dir: str = "data/raw",
    preprocess: bool = True,
) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}

    for symbol in universe:
        cache_key = f"{source}:{symbol}:{start}:{end}"
        if use_cache:
            cached = read_cache(cache_key, cache_dir=cache_dir)
            if cached is not None:
                data[symbol] = cached
                continue

        if source == "csv":
            df = _load_csv(symbol, start, end)
        else:
            df = _load_fdr(symbol, start, end)

        if preprocess:
            df = preprocess_ohlcv(df)

        data[symbol] = df
        if use_cache:
            write_cache(cache_key, df, cache_dir=cache_dir)

    return data
