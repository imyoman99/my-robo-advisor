from __future__ import annotations

from typing import Any, Dict, Tuple

from .allocation import (
    EqualWeightAllocator,
    InverseVolatilityAllocator,
    SlotWeightAllocator,
)
from .correlation import AverageCorrelationFilter, NoCorrelationFilter
from .momentum import SimpleMomentumRanker, VolAdjustedMomentumRanker
from .selection import RankRangeWithReplacementSelector, TopNSelector
from .stoploss import NoStopLoss, TrailingStopLoss
from .types import Allocator, CorrelationFilter, MomentumRanker, Selector, StopLoss


def _legacy_to_plugin_cfg(
    logic: Dict[str, Any], dynamic: Dict[str, Any]
) -> Dict[str, Any]:
    # 기존 키(CORRELATION_THRESHOLD, MOMENTUM_RANK_RANGE, SLOT_WEIGHTS, STOP_LOSS_PCT, REPLACEMENT_ORDER)를
    # 새 플러그인 스키마로 자동 변환
    if any(
        k in logic
        for k in [
            "correlation_filter",
            "momentum_ranker",
            "selector",
            "allocator",
            "stop_loss",
        ]
    ):
        return logic

    corr_window = int(dynamic.get("CORRELATION_WINDOW", 60))
    mom_window = int(dynamic.get("MOMENTUM_WINDOW", 60))

    return {
        "correlation_filter": {
            "type": "average",
            "threshold": float(logic.get("CORRELATION_THRESHOLD", 0.0)),
            "window": int(corr_window),
        },
        "momentum_ranker": {
            "type": "simple",
            "window": int(mom_window),
        },
        "selector": {
            "type": "rank_range",
            "rank_range": list(logic.get("MOMENTUM_RANK_RANGE", [1, 1])),
            "replacement_order": list(logic.get("REPLACEMENT_ORDER", [])),
        },
        "allocator": {
            "type": "slot_weights",
            "slot_weights": list(logic.get("SLOT_WEIGHTS", [])),
        },
        "stop_loss": {
            "type": (
                "trailing_pct" if float(logic.get("STOP_LOSS_PCT", 0.0)) > 0 else "none"
            ),
            "pct": float(logic.get("STOP_LOSS_PCT", 0.0)),
        },
    }


def build_dynamic_components(
    config: Dict[str, Any], *, fee: float, slot_count: int
) -> Tuple[
    CorrelationFilter,
    MomentumRanker,
    Selector,
    Allocator,
    StopLoss,
]:
    dynamic = config["DYNAMIC"]
    legacy_logic = dynamic.get("LOGIC", {}) or {}
    logic = _legacy_to_plugin_cfg(legacy_logic, dynamic)

    # correlation
    corr_cfg = logic.get("correlation_filter", {}) or {}
    corr_type = str(corr_cfg.get("type", "average")).lower()
    if corr_type in ["none", "off", "disabled"]:
        correlation: CorrelationFilter = NoCorrelationFilter()
    else:
        correlation = AverageCorrelationFilter(
            threshold=float(corr_cfg.get("threshold", 0.0)),
            window=int(corr_cfg.get("window", dynamic.get("CORRELATION_WINDOW", 60))),
        )

    # momentum
    mom_cfg = logic.get("momentum_ranker", {}) or {}
    mom_type = str(mom_cfg.get("type", "simple")).lower()
    if mom_type in ["vol_adj", "voladjusted", "risk_adj", "sharpe_like"]:
        momentum = VolAdjustedMomentumRanker(
            window=int(mom_cfg.get("window", dynamic.get("MOMENTUM_WINDOW", 60))),
            vol_window=mom_cfg.get("vol_window"),
            min_vol=float(mom_cfg.get("min_vol", 1e-8)),
        )
    else:
        momentum = SimpleMomentumRanker(
            window=int(mom_cfg.get("window", dynamic.get("MOMENTUM_WINDOW", 60)))
        )

    # selector
    sel_cfg = logic.get("selector", {}) or {}
    sel_type = str(sel_cfg.get("type", "rank_range")).lower()
    if sel_type in ["topn", "top_n", "top"]:
        selector: Selector = TopNSelector(n=int(sel_cfg.get("n", slot_count)))
    else:
        rr = sel_cfg.get("rank_range", [1, slot_count])
        start = int(rr[0]) if len(rr) >= 1 else 1
        end = int(rr[1]) if len(rr) >= 2 else start
        selector = RankRangeWithReplacementSelector(
            rank_range=(start, end),
            replacement_order=list(sel_cfg.get("replacement_order", [])),
        )

    # allocator
    alloc_cfg = logic.get("allocator", {}) or {}
    alloc_type = str(alloc_cfg.get("type", "slot_weights")).lower()
    if alloc_type in ["equal", "equal_weight", "equalweight"]:
        allocator: Allocator = EqualWeightAllocator()
    elif alloc_type in [
        "inv_vol",
        "inverse_vol",
        "inverse_volatility",
        "risk_parity_1overvol",
    ]:
        allocator = InverseVolatilityAllocator(
            window=int(alloc_cfg.get("window", dynamic.get("MOMENTUM_WINDOW", 60))),
            min_vol=float(alloc_cfg.get("min_vol", 1e-8)),
            max_weight=alloc_cfg.get("max_weight"),
        )
    else:
        allocator = SlotWeightAllocator(
            slot_weights=list(
                alloc_cfg.get("slot_weights", legacy_logic.get("SLOT_WEIGHTS", []))
            )
        )

    # stop loss
    sl_cfg = logic.get("stop_loss", {}) or {}
    sl_type = str(sl_cfg.get("type", "trailing_pct")).lower()
    if sl_type in ["none", "off", "disabled"] or float(sl_cfg.get("pct", 0.0)) <= 0:
        stop_loss: StopLoss = NoStopLoss()
    else:
        stop_loss = TrailingStopLoss(
            pct=float(sl_cfg.get("pct", 0.0)),
            fee=float(fee),
            selector=selector,
            slot_count=int(slot_count),
        )

    return correlation, momentum, selector, allocator, stop_loss
