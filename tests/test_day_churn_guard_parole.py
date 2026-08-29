"""CHURN_GUARD 4h parole: identical frozen 20-SELL window must not re-lock."""

from __future__ import annotations

import os
import tempfile

from backend.services.day_churn_guard import (
    CHURN_GUARD_MAX_SEC,
    ChurnGuardState,
    evaluate_churn_transition,
    fingerprint_sell_ids,
    load_churn_guard_state,
    persist_churn_guard_state,
)

# 20-SELL window: costs 23.13 / win 41.99 = 0.551 > 0.50
_BAD_IDS = list(range(100, 120))
_BAD_ROWS = [(1.1565, 0.0, 2.0995, i) for i in _BAD_IDS]  # 20 * 1.1565 = 23.13; 20 * 2.0995 = 41.99


def _rows(ids, fee_each=1.1565, win_each=2.0995):
    return [(fee_each, 0.0, win_each, i) for i in ids]


def test_case1_new_bad_window_activates():
    st = evaluate_churn_transition(now=1_000.0, rows=_BAD_ROWS, state=ChurnGuardState())
    assert st.active is True
    assert st.activated_at == 1_000.0
    assert st.parole_window_fp == fingerprint_sell_ids(_BAD_IDS)
    assert st.ratio > 0.50


def test_case2_same_window_before_4h_stays_blocked():
    armed = evaluate_churn_transition(now=1_000.0, rows=_BAD_ROWS, state=ChurnGuardState())
    mid = evaluate_churn_transition(now=1_000.0 + 3_600.0, rows=_BAD_ROWS, state=armed)
    assert mid.active is True
    assert mid.activated_at == 1_000.0
    assert mid.parole_window_fp == armed.parole_window_fp


def test_case3_same_window_after_4h_clears():
    armed = evaluate_churn_transition(now=1_000.0, rows=_BAD_ROWS, state=ChurnGuardState())
    after = evaluate_churn_transition(
        now=1_000.0 + CHURN_GUARD_MAX_SEC + 1.0,
        rows=_BAD_ROWS,
        state=armed,
    )
    assert after.active is False
    assert after.activated_at == 0.0
    assert after.parole_window_fp == fingerprint_sell_ids(_BAD_IDS)


def test_case4_same_window_repeated_after_clear_must_not_rearm():
    armed = evaluate_churn_transition(now=1_000.0, rows=_BAD_ROWS, state=ChurnGuardState())
    cleared = evaluate_churn_transition(
        now=1_000.0 + CHURN_GUARD_MAX_SEC + 1.0,
        rows=_BAD_ROWS,
        state=armed,
    )
    again = evaluate_churn_transition(
        now=1_000.0 + CHURN_GUARD_MAX_SEC + 60.0,
        rows=_BAD_ROWS,
        state=cleared,
    )
    third = evaluate_churn_transition(
        now=1_000.0 + CHURN_GUARD_MAX_SEC + 120.0,
        rows=_BAD_ROWS,
        state=again,
    )
    assert cleared.active is False
    assert again.active is False
    assert third.active is False
    assert again.parole_window_fp == fingerprint_sell_ids(_BAD_IDS)


def test_case5_new_sell_changes_window_may_activate_again():
    armed = evaluate_churn_transition(now=1_000.0, rows=_BAD_ROWS, state=ChurnGuardState())
    cleared = evaluate_churn_transition(
        now=1_000.0 + CHURN_GUARD_MAX_SEC + 1.0,
        rows=_BAD_ROWS,
        state=armed,
    )
    new_ids = list(range(101, 121))  # dropped 100, added 120
    new_rows = _rows(new_ids)
    rearmed = evaluate_churn_transition(
        now=1_000.0 + CHURN_GUARD_MAX_SEC + 90.0,
        rows=new_rows,
        state=cleared,
    )
    assert fingerprint_sell_ids(new_ids) != fingerprint_sell_ids(_BAD_IDS)
    assert rearmed.active is True
    assert rearmed.parole_window_fp == fingerprint_sell_ids(new_ids)
    assert rearmed.ratio > 0.50


def test_case6_new_window_ratio_at_or_below_limit_stays_clear():
    armed = evaluate_churn_transition(now=1_000.0, rows=_BAD_ROWS, state=ChurnGuardState())
    cleared = evaluate_churn_transition(
        now=1_000.0 + CHURN_GUARD_MAX_SEC + 1.0,
        rows=_BAD_ROWS,
        state=armed,
    )
    # Same costs, much larger winning invariant → ratio ~0.10
    healthy = _rows(list(range(200, 220)), fee_each=1.1565, win_each=20.0)
    stay = evaluate_churn_transition(
        now=1_000.0 + CHURN_GUARD_MAX_SEC + 90.0,
        rows=healthy,
        state=cleared,
    )
    assert stay.active is False
    assert stay.ratio <= 0.50


def test_case7_restart_while_active_preserves_identity():
    armed = evaluate_churn_transition(now=5_000.0, rows=_BAD_ROWS, state=ChurnGuardState())
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        persist_churn_guard_state(path, armed)
        restored = load_churn_guard_state(path)
        assert restored.active is True
        assert restored.activated_at == 5_000.0
        assert restored.parole_window_fp == fingerprint_sell_ids(_BAD_IDS)
        still = evaluate_churn_transition(now=5_000.0 + 60.0, rows=_BAD_ROWS, state=restored)
        assert still.active is True
        assert still.activated_at == 5_000.0
    finally:
        os.unlink(path)


def test_threshold_and_window_constants_unchanged():
    from backend.services.day_churn_guard import CHURN_RATIO_LIMIT, CHURN_TRADE_WINDOW

    assert CHURN_RATIO_LIMIT == 0.50
    assert CHURN_TRADE_WINDOW == 20
    assert CHURN_GUARD_MAX_SEC == 14400
