"""SCALP V2 opportunity identity — prevents same-move repeated entries.

A ScalpOpportunityId is derived from:
  symbol + setup_family + structural_anchor + entry_bar_15m

Two signals in the same 15m bar for the same symbol/setup family share
an opportunity ID. Re-entry is only permitted after a new bar AND a
change in either setup_family or structural_anchor.

This is a data class + hash function, no execution logic.
"""

from __future__ import annotations

import dataclasses
import hashlib


@dataclasses.dataclass(frozen=True)
class ScalpOpportunityId:
    symbol: str
    setup_family: str
    structural_anchor: str  # e.g. "VWAP_ABOVE" or "SUPPORT_1.42"
    entry_bar_15m: str  # ISO timestamp of the 15m bar, rounded

    @property
    def canonical_id(self) -> str:
        raw = f"{self.symbol}:{self.setup_family}:{self.structural_anchor}:{self.entry_bar_15m}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def from_intent(cls, symbol: str, setup: str, bar_timestamp: str) -> ScalpOpportunityId:
        """Derive from intent fields available at arm time."""
        from datetime import datetime, timezone

        try:
            dt = datetime.fromisoformat(bar_timestamp.replace("Z", "+00:00"))
            # Round down to 15m boundary
            mins = (dt.minute // 15) * 15
            bar_15m = dt.replace(minute=mins, second=0, microsecond=0).isoformat()
        except Exception:
            bar_15m = bar_timestamp[:16]

        setup_family = setup.split("_", maxsplit=1)[0] if setup else "UNKNOWN"
        anchor = "LIVE"  # Simplified — real version would compute from candle context
        return cls(
            symbol=symbol,
            setup_family=setup_family,
            structural_anchor=anchor,
            entry_bar_15m=bar_15m,
        )
