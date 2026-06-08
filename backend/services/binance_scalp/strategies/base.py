"""Base types for paper scalp strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScalpSetupSignal:
    symbol: str
    side: str
    score: float
    setup_name: str
    confidence: float
    entry_reason: str
    invalidation_reason: str | None
    required_target_pct: float
    expected_move_pct: float
    spread_pct: float
    impact_pct: float
    depth_sufficient: bool
    limit_buy_price: float
    passed: bool
    reject_reason: str | None
    setup_context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "score": self.score,
            "setup_name": self.setup_name,
            "confidence": self.confidence,
            "entry_reason": self.entry_reason,
            "invalidation_reason": self.invalidation_reason,
            "required_target_pct": self.required_target_pct,
            "expected_move_pct": self.expected_move_pct,
            "spread_pct": self.spread_pct,
            "impact_pct": self.impact_pct,
            "depth_sufficient": self.depth_sufficient,
            "limit_buy_price": self.limit_buy_price,
            "passed": self.passed,
            "reject_reason": self.reject_reason,
            "setup_context": self.setup_context,
        }


@dataclass
class StrategyMarketContext:
    """Shared inputs for all scalp strategies."""

    symbol: str
    snap: Any
    mom: Any
    bars_1m: list[dict]
    econ: Any
    config: Any
    notional_usd: float
