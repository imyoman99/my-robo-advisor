from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str, index: bool = True) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=index)
    return file_path


def save_equity_curve(
    equity: pd.Series,
    path: str = "data/results/equity_curve.csv",
) -> Path:
    df = equity.to_frame(name="equity")
    return save_dataframe(df, path, index=True)
