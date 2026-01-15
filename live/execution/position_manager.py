from __future__ import annotations

from typing import Dict

from live.broker.base import LiveBroker


class PositionManager:
    def __init__(self, broker: LiveBroker) -> None:
        self.broker = broker
        self.positions: Dict[str, int] = {}

    def refresh(self) -> Dict[str, int]:
        self.positions = self.broker.get_positions()
        return self.positions
