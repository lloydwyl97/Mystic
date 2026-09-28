"""Engine-scoped position keys: '<engine_id>::<symbol>' (e.g. 'SCALP_V2::BTC/USDT').

The composite key is internal identity only. Exchange-facing formatters call
`venue_symbol` so engine identity never reaches Binance.US. No imports: the
symbol formatters depend on this module.
"""

from __future__ import annotations

POSITION_KEY_SEP = "::"


def split_engine_key(key: object) -> tuple[str, str]:
    """'<engine>::<symbol>' -> (engine, symbol); a bare symbol returns ('', symbol) unchanged."""
    s = str(key or "")
    engine, sep, symbol = s.rpartition(POSITION_KEY_SEP)
    return (engine, symbol) if sep else ("", s)


def venue_symbol(value: object) -> str:
    """Market symbol with any engine prefix removed; bare symbols pass through unchanged."""
    return split_engine_key(value)[1]
