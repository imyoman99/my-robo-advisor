from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


class KISClient:
    def __init__(
        self,
        *,
        mock: bool | None = None,
        state_path: str | None = None,
        initial_cash: float = 10_000_000,
    ) -> None:
        self.mock = True if mock is None else bool(mock)
        self.state_path = Path(state_path or "live/mock_account.json")
        self.initial_cash = float(initial_cash)

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"cash": self.initial_cash, "positions": {}}
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"cash": self.initial_cash, "positions": {}}

        cash = float(data.get("cash", self.initial_cash))
        positions = data.get("positions", {}) or {}
        cleaned = {str(k): int(v) for k, v in positions.items() if int(v) != 0}
        return {"cash": cash, "positions": cleaned}

    def _save_state(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": float(state.get("cash", self.initial_cash)),
            "positions": {k: int(v) for k, v in (state.get("positions") or {}).items()},
        }
        self.state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get_ohlcv(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        if not self.mock:
            raise NotImplementedError(
                "KIS 시세 API 연동이 필요합니다. 실계좌용 OHLCV 호출을 추가하세요."
            )
        try:
            import FinanceDataReader as fdr
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FinanceDataReader가 필요합니다. `pip install finance-datareader` 실행 후 재시도하세요."
            ) from exc
        return fdr.DataReader(symbol, start, end)

    def get_price(self, symbol: str) -> float:
        df = self.get_ohlcv(symbol, None, None)
        if df.empty:
            return 0.0
        for col in ("Close", "close"):
            if col in df.columns:
                return float(df[col].iloc[-1])
        return float(df.iloc[-1].iloc[0])

    def get_positions(self) -> Dict[str, int]:
        if not self.mock:
            raise NotImplementedError(
                "실계좌 포지션 조회는 KIS 실계좌 API 연동이 필요합니다."
            )
        state = self._load_state()
        return {k: int(v) for k, v in (state.get("positions") or {}).items()}

    def get_cash(self) -> float:
        if not self.mock:
            raise NotImplementedError(
                "실계좌 현금 조회는 KIS 실계좌 API 연동이 필요합니다."
            )
        state = self._load_state()
        return float(state.get("cash", self.initial_cash))

    def list_universe(self, *, market: str = "KRX") -> list[str]:
        if not self.mock:
            raise NotImplementedError("KIS 종목 리스트 API 연동이 필요합니다.")
        return []

    def place_order(
        self, symbol: str, qty: int, side: str, order_type: str = "market"
    ) -> dict:
        if not self.mock:
            raise NotImplementedError("실계좌 주문은 KIS 실계좌 API 연동이 필요합니다.")

        if qty <= 0:
            return {
                "status": "rejected",
                "symbol": symbol,
                "qty": int(qty),
                "side": side,
                "reason": "qty<=0",
            }

        state = self._load_state()
        positions = state.get("positions", {}) or {}
        cash = float(state.get("cash", self.initial_cash))
        price = float(self.get_price(symbol))
        trade_value = float(price * qty)

        side_lower = str(side).lower()
        if side_lower == "buy":
            cash -= trade_value
            positions[symbol] = int(positions.get(symbol, 0)) + int(qty)
        elif side_lower == "sell":
            cash += trade_value
            positions[symbol] = int(positions.get(symbol, 0)) - int(qty)
            if positions[symbol] == 0:
                positions.pop(symbol, None)
        else:
            return {
                "status": "rejected",
                "symbol": symbol,
                "qty": int(qty),
                "side": side,
                "reason": "unknown side",
            }

        state["cash"] = cash
        state["positions"] = positions
        self._save_state(state)

        return {
            "status": "filled",
            "symbol": symbol,
            "qty": int(qty),
            "side": side_lower,
            "price": price,
            "order_type": order_type,
        }

    def cancel_order(self, order_id: str) -> dict:
        if not self.mock:
            raise NotImplementedError(
                "실계좌 주문 취소는 KIS 실계좌 API 연동이 필요합니다."
            )
        return {"status": "canceled", "order_id": order_id}
