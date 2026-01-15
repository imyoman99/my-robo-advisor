from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class LiveBroker(ABC):
    @abstractmethod
    def get_price(self, symbol: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def place_order(
        self, symbol: str, qty: int, side: str, order_type: str = "market"
    ) -> dict:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict:
        raise NotImplementedError
