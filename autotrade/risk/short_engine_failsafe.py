"""Independent long/short failsafe lane helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class FailsafeLane:
    wr: float = 0.0
    pf: float = 0.0
    sample_count: int = 0
    level: str = "normal"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FailsafeLane":
        data = payload or {}
        return cls(
            wr=float(data.get("wr", data.get("win_rate", 0.0)) or 0.0),
            pf=float(data.get("pf", data.get("profit_factor", 0.0)) or 0.0),
            sample_count=int(data.get("sample_count", data.get("sample_size", 0)) or 0),
            level=str(data.get("level", "normal") or "normal"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def allows_new_entries(
        self, *, pf_floor: float = 1.1, min_samples: int = 30
    ) -> bool:
        if str(self.level or "").lower() in {"halted", "critical"}:
            return False
        if self.sample_count >= min_samples and self.pf < pf_floor:
            return False
        return True


@dataclass
class SideFailsafeState:
    long: FailsafeLane
    short: FailsafeLane
    global_drawdown_pct: float = 0.0
    global_circuit_breaker_active: bool = False

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SideFailsafeState":
        payload = payload or {}
        return cls(
            long=FailsafeLane.from_dict(payload.get("long", {})),
            short=FailsafeLane.from_dict(payload.get("short", {})),
            global_drawdown_pct=float(payload.get("global_drawdown_pct", 0.0) or 0.0),
            global_circuit_breaker_active=bool(
                payload.get("global_circuit_breaker_active", False)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "long": self.long.to_dict(),
            "short": self.short.to_dict(),
            "global_drawdown_pct": self.global_drawdown_pct,
            "global_circuit_breaker_active": self.global_circuit_breaker_active,
        }

    def allows_new_entries(self, side: str) -> bool:
        if self.global_circuit_breaker_active or self.global_drawdown_pct > 10.0:
            return False
        side_key = str(side or "").lower().strip()
        if side_key == "short":
            return self.short.allows_new_entries()
        return self.long.allows_new_entries()
