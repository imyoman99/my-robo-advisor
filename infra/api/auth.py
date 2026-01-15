from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class Token:
    access_token: str
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.utcnow() >= self.expires_at


class TokenManager:
    def __init__(self) -> None:
        self._token: Optional[Token] = None

    def set_token(self, access_token: str, expires_in: int) -> None:
        self._token = Token(
            access_token=access_token,
            expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
        )

    def get_token(self) -> Optional[str]:
        if self._token is None or self._token.is_expired():
            return None
        return self._token.access_token
