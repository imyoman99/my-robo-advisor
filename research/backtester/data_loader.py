from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _find_parquet_path(base: Path, symbol: str) -> Path | None:
    candidates = []
    if base.name in {"processed", "raw"}:
        candidates.append(base / f"{symbol}.parquet")
        candidates.append(base.parent / "processed" / f"{symbol}.parquet")
        candidates.append(base.parent / "raw" / f"{symbol}.parquet")
    else:
        candidates.extend(
            [
                base / "processed" / f"{symbol}.parquet",
                base / "raw" / f"{symbol}.parquet",
            ]
        )
    for p in candidates:
        if p.exists():
            return p

    exact = next(base.rglob(f"{symbol}.parquet"), None)
    if exact is not None:
        return exact

    matches = list(base.rglob(f"{symbol}*.parquet"))
    if not matches:
        matches = list(base.rglob(f"*{symbol}*.parquet"))
    if matches:
        matches.sort(key=lambda p: (len(p.as_posix()), p.name))
        return matches[0]

    return None


def _read_parquet_close(
    path: Path,
    start: str | None,
    end: str | None,
) -> pd.Series:
    df = pd.read_parquet(path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    if start:
        df = df.loc[df.index >= pd.to_datetime(start)]
    if end:
        df = df.loc[df.index <= pd.to_datetime(end)]

    close_col = (
        "close"
        if "close" in df.columns
        else ("Close" if "Close" in df.columns else None)
    )
    if close_col is None:
        raise ValueError(f"Close column not found in {path}")

    series = df[close_col].copy()
    series.name = path.stem
    return series


def load_local_parquet_prices(
    tickers: Iterable[str],
    *,
    data_root: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    ffill: bool = True,
) -> pd.DataFrame:
    base = Path(data_root).resolve()
    if not base.exists():
        raise FileNotFoundError(f"data root not found: {data_root}")

    frames = []
    for t in tickers:
        symbol = str(t)
        path = _find_parquet_path(base, symbol)
        if path is None:
            raise FileNotFoundError(f"Parquet not found for {symbol}")
        frames.append(_read_parquet_close(path, start, end))

    prices = pd.concat(frames, axis=1).sort_index()
    if ffill:
        prices = prices.ffill()
    return prices.dropna(how="all")
