"""Redis key helpers — scalp: namespace only; never write DAY keys."""

from __future__ import annotations

FORBIDDEN_PREFIXES = (
    "ai_signal:day:",
    "ai_context:",
    "bwl:",
    "paper:position:",
    "paper:",
)

FORBIDDEN_WRITE_TABLES = (
    "portfolio_engine_positions",
    "portfolio_engine_scoreboard_daily",
)


def normalize_prefix(prefix: str) -> str:
    p = (prefix or "scalp").strip().rstrip(":")
    if not p:
        raise ValueError("SCALP_REDIS_PREFIX must not be empty")
    return p


def market_key(prefix: str, symbol_bus: str) -> str:
    sym = symbol_bus.strip().upper()
    key = f"{normalize_prefix(prefix)}:market:{sym}"
    assert_key_allowed(key, prefix=prefix)
    return key


def orderbook_key(prefix: str, symbol_bus: str) -> str:
    sym = symbol_bus.strip().upper()
    key = f"{normalize_prefix(prefix)}:orderbook:{sym}"
    assert_key_allowed(key, prefix=prefix)
    return key


def signal_key(prefix: str, symbol_bus: str) -> str:
    sym = symbol_bus.strip().upper()
    key = f"{normalize_prefix(prefix)}:signal:{sym}"
    assert_key_allowed(key, prefix=prefix)
    return key


def position_key(prefix: str, symbol_bus: str) -> str:
    sym = symbol_bus.strip().upper()
    key = f"{normalize_prefix(prefix)}:position:{sym}"
    assert_key_allowed(key, prefix=prefix)
    return key


def scan_key(prefix: str, symbol_bus: str) -> str:
    sym = symbol_bus.strip().upper()
    key = f"{normalize_prefix(prefix)}:scan:{sym}"
    assert_key_allowed(key, prefix=prefix)
    return key


def runner_state_key(prefix: str) -> str:
    key = f"{normalize_prefix(prefix)}:runner:state"
    assert_key_allowed(key, prefix=prefix)
    return key


def last_decision_key(prefix: str) -> str:
    """Canonical pre-order decision snapshot — the single source of truth the
    status/dashboard endpoint must read instead of running its own independent
    ranking simulation."""
    key = f"{normalize_prefix(prefix)}:runner:last_decision"
    assert_key_allowed(key, prefix=prefix)
    return key


def status_snapshot_key(prefix: str, warm_rounds: int = 0) -> str:
    key = f"{normalize_prefix(prefix)}:status:snapshot:w{int(warm_rounds)}"
    assert_key_allowed(key, prefix=prefix)
    return key


def assert_key_allowed(key: str, *, prefix: str = "scalp") -> None:
    expected = f"{normalize_prefix(prefix)}:"
    if not key.startswith(expected):
        raise ValueError(f"scalp redis key must start with {expected!r}, got {key!r}")
    for forbidden in FORBIDDEN_PREFIXES:
        if key.startswith(forbidden):
            raise ValueError(f"refusing to write forbidden redis namespace: {key!r}")
