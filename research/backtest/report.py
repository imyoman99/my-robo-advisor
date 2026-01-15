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


def save_metrics(metrics: dict, path: str = "data/results/metrics.csv") -> Path:
    df = pd.DataFrame([metrics])
    return save_dataframe(df, path, index=False)


def save_equity_plot(
    equity: pd.Series, path: str = "data/results/equity_curve.png"
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting") from exc

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    equity.plot(ax=ax, title="Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)
    return file_path
