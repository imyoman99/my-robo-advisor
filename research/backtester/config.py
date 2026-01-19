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
        "FEES": 0.0004,
        "ASSETS": [],
        "WEIGHTS": [],
    },
    "DYNAMIC": {
        "REBALANCE_FREQ": "Daily",
        "FEES": 0.0022,
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
        "root": "data",
    },
    "RESULTS": {
        "root": "results",
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


def _apply_logic_profile(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """If DYNAMIC.LOGIC_PROFILE is provided, load config/logic_{profile}.yaml.

    - If user also provided DYNAMIC.LOGIC, it wins.
    - If the profile file is missing, raise a clear error.
    """
    dynamic = (
        user_cfg.get("DYNAMIC") if isinstance(user_cfg.get("DYNAMIC"), dict) else {}
    )
    profile = (dynamic or {}).get("LOGIC_PROFILE")
    if not profile:
        return user_cfg

    if isinstance((dynamic or {}).get("LOGIC"), dict) and (dynamic or {}).get("LOGIC"):
        return user_cfg

    profile_name = str(profile).strip()
    try:
        logic_cfg = load_yaml_config(f"logic_{profile_name}")
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"LOGIC_PROFILE='{profile_name}' 설정을 찾지 못했습니다. "
            f"config/logic_{profile_name}.yaml 파일을 생성하세요."
        ) from e

    out = deepcopy(user_cfg)
    out.setdefault("DYNAMIC", {})
    out["DYNAMIC"].setdefault("LOGIC", {})
    if not isinstance(out["DYNAMIC"]["LOGIC"], dict):
        out["DYNAMIC"]["LOGIC"] = {}
    if not isinstance(logic_cfg, dict):
        raise ValueError(f"logic_{profile_name}.yaml은 dict 구조여야 합니다.")

    out["DYNAMIC"]["LOGIC"] = _deep_merge(out["DYNAMIC"]["LOGIC"], logic_cfg)
    return out


def load_dual_engine_config(name: str = "backtester") -> Dict[str, Any]:
    """Load config/{name}.yaml and merge onto DEFAULT_CONFIG."""
    user_cfg = load_yaml_config(name)
    user_cfg = _apply_logic_profile(user_cfg)
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
