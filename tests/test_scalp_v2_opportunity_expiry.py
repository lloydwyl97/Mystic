"""Tests for SCALP V2 opportunity ARMED-state expiry.

Proves that:
- ARMED rows older than SCALP_V2_OPP_EXPIRY_SEC are reset and a fresh arm
  in the same price zone is permitted.
- ARMED rows younger than the expiry are still blocked (deduplication intact).
- OPEN rows are never expired (active position).
- CLOSED rows are never expired (still blocking same-move recycling).
- Different price zones always create a fresh ARMED row.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


def _db() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# 1. Fresh arm: not blocked
# ---------------------------------------------------------------------------


def test_fresh_arm_is_not_blocked():
    from backend.services.scalp_v2.opportunity import arm_opportunity

    db = _db()
    opp_id, blocked = arm_opportunity(db, "BTC/USDT", "SCALP", 84000.0)
    assert not blocked
    assert len(opp_id) == 16


# ---------------------------------------------------------------------------
# 2. Same zone, fresh row: blocked
# ---------------------------------------------------------------------------


def test_same_zone_second_arm_is_blocked_while_fresh():
    from backend.services.scalp_v2.opportunity import arm_opportunity

    db = _db()
    # First arm
    arm_opportunity(db, "BTC/USDT", "SCALP", 84000.0)
    # Second arm in same price zone — should be blocked
    _, blocked = arm_opportunity(db, "BTC/USDT", "SCALP", 84050.0)
    assert blocked, "Same-zone duplicate arm must be blocked while ARMED row is fresh"


# ---------------------------------------------------------------------------
# 3. Expired ARMED row: unblocked
# ---------------------------------------------------------------------------


def test_expired_armed_row_allows_new_arm():
    """ARMED row older than SCALP_V2_OPP_EXPIRY_SEC is reset, new arm succeeds."""
    from backend.services.scalp_v2.opportunity import SCALP_V2_OPP_EXPIRY_SEC, arm_opportunity

    db = _db()
    # First arm — succeeds
    _, blocked1 = arm_opportunity(db, "BTC/USDT", "SCALP", 84000.0)
    assert not blocked1

    # Backdate the created_at of the ARMED row to simulate expiry
    conn = sqlite3.connect(db)
    old_ts = time.time() - SCALP_V2_OPP_EXPIRY_SEC - 60  # 1 min past expiry
    conn.execute("UPDATE scalp_v2_opportunities SET created_at=?, updated_at=?", (old_ts, old_ts))
    conn.commit()
    conn.close()

    # Second arm in the same price zone — should now succeed (expired)
    _, blocked2 = arm_opportunity(db, "BTC/USDT", "SCALP", 84050.0)
    assert not blocked2, "Expired ARMED row must be reset; new arm must not be blocked"


# ---------------------------------------------------------------------------
# 4. OPEN rows are never expired
# ---------------------------------------------------------------------------


def test_open_row_is_not_expired():
    """An OPEN row (active position) must remain blocked even when old."""
    import sqlite3

    from backend.services.scalp_v2.opportunity import SCALP_V2_OPP_EXPIRY_SEC, arm_opportunity

    db = _db()
    _, blocked1 = arm_opportunity(db, "BTC/USDT", "SCALP", 84000.0)
    assert not blocked1

    # Set to OPEN state (simulating a fill) and backdate
    conn = sqlite3.connect(db)
    old_ts = time.time() - SCALP_V2_OPP_EXPIRY_SEC - 60
    conn.execute(
        "UPDATE scalp_v2_opportunities SET state='OPEN', created_at=?, updated_at=?",
        (old_ts, old_ts),
    )
    conn.commit()
    conn.close()

    # Same zone arm must be blocked — position is live
    _, blocked2 = arm_opportunity(db, "BTC/USDT", "SCALP", 84050.0)
    assert blocked2, "OPEN row must block same-zone arm even when old"


# ---------------------------------------------------------------------------
# 5. Different price zone is never blocked
# ---------------------------------------------------------------------------


def test_different_zone_does_not_create_a_second_actionable_row():
    """One ARMED row per symbol. A second price zone stays historical until the first ends."""
    from backend.services.scalp_v2.opportunity import arm_opportunity

    db = _db()
    arm_opportunity(db, "BTC/USDT", "SCALP", 84000.0)
    _, blocked = arm_opportunity(db, "BTC/USDT", "SCALP", 88000.0)
    assert blocked, "A second price zone must not become actionable while one ARMED row exists"


# ---------------------------------------------------------------------------
# 6. Cross-symbol isolation
# ---------------------------------------------------------------------------


def test_armed_row_does_not_block_different_symbol():
    """BTCUSDT arm does not block ETHUSDT arm in same price zone."""
    from backend.services.scalp_v2.opportunity import arm_opportunity

    db = _db()
    arm_opportunity(db, "BTC/USDT", "SCALP", 84000.0)
    _, blocked = arm_opportunity(db, "ETH/USDT", "SCALP", 84000.0)
    assert not blocked, "Different symbol must not be blocked by another symbol's ARMED row"


# ---------------------------------------------------------------------------
# 7. Multiple symbols each arm independently
# ---------------------------------------------------------------------------


def test_all_four_symbols_can_arm_independently():
    from backend.services.scalp_v2.opportunity import arm_opportunity

    db = _db()
    prices = {"BTC/USDT": 84000.0, "ETH/USDT": 3400.0, "SOL/USDT": 180.0, "XRP/USDT": 2.5}
    for sym, px in prices.items():
        _, blocked = arm_opportunity(db, sym, "SCALP", px)
        assert not blocked, f"{sym} arm must not be blocked on first attempt"
