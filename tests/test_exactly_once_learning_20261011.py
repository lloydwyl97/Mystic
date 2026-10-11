"""Every live learner folds a label exactly once: a crash mid-fold leaves no
partial state and the retry writes it once; a repeated delivery is a no-op."""

from __future__ import annotations

import sqlite3

import pytest

from backend.services import adaptive_learning as al
from backend.services.adaptive_learning import learn_from_close, record_candidate, resolve_markouts
from backend.services.strategy_version import DAY_STRATEGY_VERSION, SCALP_STRATEGY_VERSION, exit_policy_anchor

SCALP = "SCALP_V2"
DAY = "DAY_V2"


def _state(db: str) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT engine_id, symbol, setup, regime, metric, round(n, 9), round(ewma, 12) FROM adaptive_metric_state ORDER BY 1,2,3,4,5").fetchall()
    finally:
        conn.close()


def _linear(db: str) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT engine_id, model, payload FROM adaptive_linear_model ORDER BY 1,2").fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _scalp_row(db: str) -> None:
    record_candidate(
        db,
        engine=SCALP,
        symbol="ETHUSDT",
        setup="VWAP_EMA_RECLAIM",
        regime="btcup_vollo",
        ref_price=100.0,
        roundtrip_cost=0.0006,
        signaled=True,
        evaluated_at=1_000.0,
        raw_expected_move=0.003,
    )


def test_markout_crash_mid_fold_leaves_nothing_and_retry_learns_once(tmp_path, monkeypatch):
    crashed, clean = str(tmp_path / "crash.db"), str(tmp_path / "clean.db")
    for db in (crashed, clean):
        _scalp_row(db)

    real = al._fold_observation
    calls = {"n": 0}

    def _boom(conn, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("injected crash between folds")
        return real(conn, **kwargs)

    monkeypatch.setattr(al, "_fold_observation", _boom)
    assert resolve_markouts(crashed, lambda _s, _t: 100.4, now=3_000.0) == 0
    assert _state(crashed) == []
    conn = sqlite3.connect(crashed)
    assert conn.execute("SELECT learned FROM adaptive_candidate_markouts").fetchone()[0] == 0
    conn.close()

    monkeypatch.setattr(al, "_fold_observation", real)
    assert resolve_markouts(crashed, lambda _s, _t: 100.4, now=3_000.0) == 1
    assert resolve_markouts(crashed, lambda _s, _t: 100.4, now=3_000.0) == 0
    assert resolve_markouts(clean, lambda _s, _t: 100.4, now=3_000.0) == 1
    assert _state(crashed) == _state(clean) != []
    assert _linear(crashed) == _linear(clean)


def _close(db: str, *, key: str) -> bool:
    return learn_from_close(
        db,
        engine=DAY,
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_vollo",
        strategy_version=DAY_STRATEGY_VERSION,
        net_pct=0.004,
        mfe_pct=0.006,
        mae_pct=-0.002,
        hold_min=30.0,
        continuation=0.5,
        version_current=True,
        is_dust=False,
        entered_at=float(exit_policy_anchor(DAY)["epoch"]) + 60.0,
        now=float(exit_policy_anchor(DAY)["epoch"]) + 3_600.0,
        unrealized_marks=[0.001, 0.002, -0.001],
        exit_reason="LEARNED_CONTINUATION_EXIT",
        close_key=key,
    )


def test_close_is_learned_once_per_lot(tmp_path):
    once, twice = str(tmp_path / "once.db"), str(tmp_path / "twice.db")
    assert _close(once, key="mystic_BTC/USDT_1")
    assert _close(twice, key="mystic_BTC/USDT_1")
    assert not _close(twice, key="mystic_BTC/USDT_1")
    assert _state(once) == _state(twice) != []


def test_close_crash_mid_fold_rolls_back_and_retry_learns_once(tmp_path, monkeypatch):
    crashed, clean = str(tmp_path / "crash.db"), str(tmp_path / "clean.db")
    real = al._fold_observation
    calls = {"n": 0}

    def _boom(conn, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise sqlite3.OperationalError("injected crash between folds")
        return real(conn, **kwargs)

    monkeypatch.setattr(al, "_fold_observation", _boom)
    with pytest.raises(sqlite3.OperationalError):
        _close(crashed, key="lot-7")
    assert _state(crashed) == []
    monkeypatch.setattr(al, "_fold_observation", real)
    assert _close(crashed, key="lot-7")
    assert not _close(crashed, key="lot-7")
    assert _close(clean, key="lot-7")
    assert _state(crashed) == _state(clean)


def test_continuation_advantage_folds_once(tmp_path, monkeypatch):
    from backend.services import continuation_surface as cs
    from backend.services.continuation_backfill import AdvantageLabel, fold_advantages_once

    db = str(tmp_path / "adv.db")
    clean = str(tmp_path / "adv_clean.db")
    al._connect(db).close()
    al._connect(clean).close()
    horizon = cs.DAY_ADVANTAGE_HORIZONS[0]
    label = AdvantageLabel(
        engine=DAY,
        trade_id="mystic_BTC/USDT_9",
        symbol="BTCUSDT",
        setup="BREAKOUT_CONTINUATION",
        regime="btcup_vollo",
        snapshot_time=1_000.0,
        horizon=horizon,
        features=dict.fromkeys(cs.FEATURE_NAMES, 0.001),
        advantage=-0.0004,
        weight=1.0,
        label_time=1_000.0 + horizon,
    )
    real = cs._update_ridge
    calls = {"n": 0}

    def _boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("injected crash after the metric fold")
        return real(*args, **kwargs)

    monkeypatch.setattr(cs, "_update_ridge", _boom)
    with pytest.raises(sqlite3.OperationalError):
        fold_advantages_once(db, [label])
    assert _state(db) == []
    monkeypatch.setattr(cs, "_update_ridge", real)
    assert fold_advantages_once(db, [label]) == 1
    assert fold_advantages_once(db, [label]) == 0
    assert fold_advantages_once(clean, [label]) == 1
    assert _state(db) == _state(clean) != []
    assert _linear(db) == _linear(clean)


def test_startup_calibration_repair_never_refolds_a_trade_net_close(tmp_path):
    db = str(tmp_path / "cal.db")
    key = ("BTCUSDT", "BREAKOUT_CONTINUATION__MARKET", "btcup_vollo")
    moment = float(exit_policy_anchor(DAY)["epoch"]) + 600.0
    row_id = al.record_candidate(
        db,
        engine=DAY,
        symbol=key[0],
        setup=key[1],
        regime=key[2],
        ref_price=100.0,
        roundtrip_cost=0.00066,
        signaled=True,
        evaluated_at=moment,
        economic={"policy_base": "trade_net", "uncalibrated_policy_value": -0.0004, "policy_value": -0.0004, "policy_calibration": 0.0},
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE adaptive_candidate_markouts SET filled=1, realized_net=-0.0005 WHERE id=?", (row_id,))
        conn.commit()
    conn = al._connect(db)
    try:
        row = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()
        assert al._learn_policy_calibration(conn, db, row, -0.0005, moment)
        conn.commit()
        before = _state(db)
        assert al.repair_marked_calibration_without_observation(conn) == (0, 0)
        conn.commit()
        flag = conn.execute("SELECT calibration_learned FROM adaptive_candidate_markouts WHERE id=?", (row_id,)).fetchone()[0]
    finally:
        conn.close()
    assert _state(db) == before
    assert [m for *_, m, _n, _e in before] == ["trade_net_calibration"]
    assert flag == 1


def test_scalp_labels_resolve_without_a_day_candidate():
    import inspect

    from backend.services import portfolio_engine_integration as pei

    day = inspect.getsource(pei.PortfolioEngineIntegration._process_day_v2_signals)
    assert day.index("resolve_learning_labels(db_path)") < day.index("if not candidates:")
    loop = inspect.getsource(pei.PortfolioEngineIntegration._scalp_v2_live_loop)
    assert "resolve_learning_labels" in loop
    _ = SCALP_STRATEGY_VERSION
