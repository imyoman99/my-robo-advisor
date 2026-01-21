from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from core.engine import build_dynamic_components
from core.types import DynamicState


def _month_day_in_season(date: pd.Timestamp, start_md: str, end_md: str) -> bool:
    md = date.strftime("%m-%d")
    if start_md <= end_md:
        return start_md <= md <= end_md
    return md >= start_md or md <= end_md


def _freq_to_pandas(freq: str) -> str:
    mapping = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "ME",
        "Quarterly": "QE",
    }
    return mapping.get(freq, freq)


@dataclass
class LiveDynamicStrategy:
    config: Dict[str, object]

    def __post_init__(self) -> None:
        config = self.config if isinstance(self.config, dict) else {}
        dynamic_cfg = config.get("DYNAMIC", {}) or {}
        if not isinstance(dynamic_cfg, dict):
            dynamic_cfg = {}
        self.logic = dynamic_cfg.get("LOGIC", {}) or {}
        if not isinstance(self.logic, dict):
            self.logic = {}
        self.rebalance_freq = str(dynamic_cfg.get("REBALANCE_FREQ", "Daily"))

        self.seasons = dynamic_cfg.get("SEASONS", []) or []

        self.selection_cfg = dynamic_cfg.get("SELECTION", {}) or {}
        if not isinstance(self.selection_cfg, dict):
            self.selection_cfg = {}
        self.selection_mode = str(self.selection_cfg.get("mode", "season")).lower()
        self.selection_top_n = int(self.selection_cfg.get("top_n", 10))
        self.selection_buy_n = int(
            self.selection_cfg.get("buy_n", self.selection_top_n)
        )
        if self.selection_buy_n <= 0:
            self.selection_buy_n = self.selection_top_n
        self.selection_buy_n = min(self.selection_buy_n, self.selection_top_n)

        self.selection_weighting = str(
            self.selection_cfg.get("weighting", "equal")
        ).lower()
        self.filters_cfg = self.selection_cfg.get("FILTERS", {}) or {}
        rank_weights = self.selection_cfg.get("rank_weights", []) or []
        if isinstance(rank_weights, list):
            self.selection_rank_weights = [float(w) for w in rank_weights]
        else:
            self.selection_rank_weights = []

        fee = dynamic_cfg.get("FEES", None)
        if fee is None:
            fee = config.get("FEES", 0.0)
        try:
            self.fee = float(fee)
        except (TypeError, ValueError):
            self.fee = 0.0

        self.slot_count = self._infer_slot_count()
        (
            self.correlation_filter,
            self.momentum_ranker,
            self.selector,
            self.allocator,
            self.stop_loss,
        ) = build_dynamic_components(
            self.config, fee=self.fee, slot_count=self.slot_count
        )

    def _infer_slot_count(self) -> int:
        alloc_cfg = self.logic.get("allocator", {}) or {}
        if isinstance(alloc_cfg, dict):
            slot_weights = alloc_cfg.get("slot_weights")
            if isinstance(slot_weights, list) and slot_weights:
                return len(slot_weights)

        legacy_weights = self.logic.get("SLOT_WEIGHTS")
        if isinstance(legacy_weights, list) and legacy_weights:
            return len(legacy_weights)

        return 4

    def rebalance_dates(self, index: pd.Index) -> set[pd.Timestamp]:
        freq = _freq_to_pandas(self.rebalance_freq)
        anchor = pd.Series(1, index=pd.DatetimeIndex(index))
        return set(anchor.resample(freq).last().index)

    def compute_targets(
        self,
        prices: pd.DataFrame,
        universe: List[str],
        as_of: pd.Timestamp,
        *,
        volumes: pd.DataFrame | None = None,
    ) -> Tuple[Dict[str, float], pd.Series, List[str]]:
        tickers = [t for t in universe if t in prices.columns]
        if not tickers:
            return {}, pd.Series(dtype=float), []

        tickers = self._filter_candidates(prices, volumes, tickers, as_of)
        if not tickers:
            return {}, pd.Series(dtype=float), []

        if self.selection_mode == "season" and self.seasons:
            season = self._current_season(as_of)
            season_tickers = season.get("tickers", []) or []
            if season_tickers:
                tickers = [t for t in tickers if t in season_tickers]
            if not tickers:
                return {}, pd.Series(dtype=float), []

        filtered = self.correlation_filter.filter(prices, tickers, as_of)
        ranked = self.momentum_ranker.rank(prices, filtered, as_of)
        if ranked.empty:
            return {}, ranked, []

        if self.selection_mode == "auto":
            ranked = ranked.head(self.selection_top_n)
            selected = list(ranked.index[: self.selection_buy_n])
            targets = self._auto_targets(selected, prices, as_of)
        else:
            selected = self.selector.select(ranked, self.slot_count)
            targets = self.allocator.targets(
                prices=prices, tickers=selected, as_of=as_of
            )
        return targets, ranked, selected

    def _auto_targets(
        self, selected: List[str], prices: pd.DataFrame, as_of: pd.Timestamp
    ) -> Dict[str, float]:
        if not selected:
            return {}

        if self.selection_weighting in {"rank", "rank_weights"}:
            weights = list(self.selection_rank_weights)[: len(selected)]
            if len(weights) < len(selected):
                weights += [0.0] * (len(selected) - len(weights))
            total = float(sum(weights))
            if total > 0:
                return {t: float(w) / total for t, w in zip(selected, weights)}

        if self.selection_weighting in {"allocator", "alloc"}:
            return self.allocator.targets(prices=prices, tickers=selected, as_of=as_of)

        w = 1.0 / float(len(selected))
        return {t: w for t in selected}

    def apply_stop_loss(
        self,
        state: DynamicState,
        prices: pd.Series,
        ranked: pd.Series,
        *,
        allowed_symbols: List[str] | None = None,
    ) -> DynamicState:
        if allowed_symbols is not None:
            allowed = set(allowed_symbols)
            holdings = {k: v for k, v in state.holdings.items() if k in allowed}
            high_water = {k: v for k, v in state.high_water.items() if k in allowed}
            state = DynamicState(
                cash=state.cash, holdings=holdings, high_water=high_water
            )
        return self.stop_loss.apply(state=state, prices=prices, ranked=ranked)

    def _current_season(self, date: pd.Timestamp) -> dict:
        for season in self.seasons:
            if _month_day_in_season(
                date, season.get("start_md", "01-01"), season.get("end_md", "12-31")
            ):
                return season
        return (
            self.seasons[0]
            if self.seasons
            else {
                "name": "Season",
                "start_md": "01-01",
                "end_md": "12-31",
                "tickers": [],
            }
        )

    def _filter_candidates(
        self,
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None,
        tickers: List[str],
        as_of: pd.Timestamp,
    ) -> List[str]:
        if not tickers:
            return []

        min_age = int(self.filters_cfg.get("min_age_days", 0))
        min_price = float(self.filters_cfg.get("min_price", 0))
        min_turnover = float(self.filters_cfg.get("min_turnover", 0))
        turnover_window = int(self.filters_cfg.get("turnover_window", 20))

        if as_of not in prices.index:
            idx = prices.index[prices.index <= as_of]
            if idx.empty:
                return []
            current_idx = idx[-1]
        else:
            current_idx = as_of

        current_prices = prices.loc[current_idx].reindex(tickers)
        mask_price = (current_prices >= min_price) & current_prices.notna()

        mask_age = pd.Series(True, index=tickers)
        if min_age > 0:
            counts = prices.loc[:as_of, tickers].count()
            mask_age = counts >= min_age

        mask_turnover = pd.Series(True, index=tickers)
        if volumes is not None and min_turnover > 0:
            vol_subset = volumes.loc[:as_of, tickers].tail(turnover_window)
            px_subset = prices.loc[:as_of, tickers].tail(turnover_window)
            if not vol_subset.empty and not px_subset.empty:
                turnover = (vol_subset * px_subset).mean()
                mask_turnover = turnover >= min_turnover

        final = mask_price & mask_age & mask_turnover
        return [t for t in tickers if bool(final.get(t, False))]
