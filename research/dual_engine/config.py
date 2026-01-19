from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from infra.config import load_yaml_config


DEFAULT_CONFIG: Dict[str, Any] = {
    "START_DATE": "2015-01-01",
    "END_DATE": None,
    "INITIAL_CAPITAL": 100000000,
    "FEES": 0.0015,
    "TRADING_DAYS": 252,
    "FFILL": True,
    "MASTER": {
        "STATIC_RATIO": 0.4,
        "DYNAMIC_RATIO": 0.6,
        "REBALANCE_FREQ": "Quarterly",
    },
    "STATIC": {
        "REBALANCE_FREQ": "Monthly",
        "ASSETS": {
            "KOSPI200": "069500",
            "BOND10Y": "148070",
            "GOLD": "411060",
        },
        "WEIGHTS": [0.4, 0.4, 0.2],
    },
    "DYNAMIC": {
        "REBALANCE_FREQ": "Daily",
        "MOMENTUM_WINDOW": 60,
        "CORRELATION_WINDOW": 60,
        "SEASONS": [
            {"name": "Spring", "start_md": "03-01", "end_md": "05-31", "tickers": []},
            {"name": "Summer", "start_md": "06-01", "end_md": "08-31", "tickers": []},
            {"name": "Fall", "start_md": "09-01", "end_md": "11-30", "tickers": []},
            {"name": "Winter", "start_md": "12-01", "end_md": "02-28", "tickers": []},
        ],
        "LOGIC": {
            "correlation_filter": {
                "type": "average",
                "threshold": 0.4,
                "window": 60,
            },
            "momentum_ranker": {
                "type": "simple",
                "window": 60,
            },
            "selector": {
                "type": "rank_range",
                "rank_range": [4, 7],
                "replacement_order": [1, 2, 3, 8, 9, 10],
            },
            "allocator": {
                "type": "slot_weights",
                "slot_weights": [0.3, 0.3, 0.2, 0.2],
            },
            "stop_loss": {
                "type": "trailing_pct",
                "pct": 0.05,
            },
        },
    },
    "BENCHMARK_TICKER": "069500",
    "DATA": {
        "source": "fdr",
        "use_cache": True,
        "cache_dir": "data/raw",
        "preprocess": True,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_dual_engine_config(name: str = "dual_engine") -> Dict[str, Any]:
    """Load config/{name}.yaml and merge onto DEFAULT_CONFIG."""
    user_cfg = load_yaml_config(name)
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


def apply_overrides(
    cfg: Dict[str, Any], *, start: Optional[str], end: Optional[str]
) -> Dict[str, Any]:
    out = deepcopy(cfg)
    if start:
        out["START_DATE"] = start
    if end:
        out["END_DATE"] = end
    return out
