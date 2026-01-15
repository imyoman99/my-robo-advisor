from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .cache import read_cache, write_cache
from .preprocess import preprocess_ohlcv


def _load_from_fdr(
    symbol: str, start: Optional[str], end: Optional[str]
) -> pd.DataFrame:
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "FinanceDataReader가 필요합니다. `pip install finance-datareader` 실행 후 재시도하세요."
        ) from exc

    return fdr.DataReader(symbol, start, end)


def load_ohlcv(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: str = "fdr",
    use_cache: bool = True,
    cache_dir: str = "data/raw",
    preprocess: bool = True,
) -> pd.DataFrame:
    """단일 종목 OHLCV 로드"""
    cache_key = f"{symbol}_{start}_{end}_{source}"
    if use_cache:
        cached = read_cache(cache_key, cache_dir=cache_dir)
        if cached is not None:
            return cached

    if source == "fdr":
        df = _load_from_fdr(symbol, start, end)
    elif source == "csv":
        csv_path = Path(cache_dir) / f"{symbol}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise ValueError(f"Unsupported source: {source}")

    if preprocess:
        df = preprocess_ohlcv(df)

    if use_cache:
        write_cache(cache_key, df, cache_dir=cache_dir)
    return df


def load_ohlcv_bulk(
    symbols: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: str = "fdr",
    use_cache: bool = True,
    cache_dir: str = "data/raw",
    preprocess: bool = True,
) -> dict[str, pd.DataFrame]:
    """다중 종목 OHLCV 로드"""
    data = {}
    for symbol in symbols:
        data[symbol] = load_ohlcv(
            symbol,
            start=start,
            end=end,
            source=source,
            use_cache=use_cache,
            cache_dir=cache_dir,
            preprocess=preprocess,
        )
    return data
