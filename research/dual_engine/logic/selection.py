from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class TopNSelector:
    n: int

    def select(self, ranked: pd.Series, slot_count: int) -> List[str]:
        n = min(int(self.n), int(slot_count))
        return list(ranked.index[:n])


@dataclass(frozen=True)
class RankRangeWithReplacementSelector:
    rank_range: Tuple[int, int] = (1, 1)
    replacement_order: Sequence[int] = ()

    def select(self, ranked: pd.Series, slot_count: int) -> List[str]:
        start, end = self.rank_range
        ranked_list = list(ranked.index)
        selected: List[str] = []

        for r in range(int(start), int(end) + 1):
            idx = r - 1
            if 0 <= idx < len(ranked_list):
                selected.append(ranked_list[idx])

        for r in self.replacement_order:
            idx = int(r) - 1
            if 0 <= idx < len(ranked_list):
                t = ranked_list[idx]
                if t not in selected:
                    selected.append(t)
            if len(selected) >= int(slot_count):
                break

        if len(selected) < int(slot_count):
            for t in ranked_list:
                if t not in selected:
                    selected.append(t)
                if len(selected) >= int(slot_count):
                    break

        return selected[: int(slot_count)]
