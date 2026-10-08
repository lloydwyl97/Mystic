"""Group opportunity and coin residual stay research-only and causal."""

from __future__ import annotations

import inspect
import json
import sqlite3

import pytest

from backend.services import adaptive_learning as al
from backend.services import policy_episode as pe
from backend.services import policy_group_learner as gl
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

DAY = "DAY_V2"
SCALP = "SCALP_V2"
T0 = 1_700_000_000.0
COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def test_group_mean_plus_residual_is_the_policy_net():
    nets = {"BTCUSDT": 0.002, "ETHUSDT": -0.004, "SOLUSDT": 0.001, "XRPUSDT": 0.001}
    assert gl.reconstructs(nets)
    made = gl.targets(nets)
    assert made["best"] == pytest.approx(0.002)
    assert made["spread"] == pytest.approx(0.006)
    assert made["positive_value"] == pytest.approx(0.002)


def test_temporal_lag_uses_the_prior_decision_not_the_outcome():
    first = {coin: {"ofi_5s": 1.0, "bar_return": 0.0} for coin in COINS}
    second = {coin: {"ofi_5s": 3.0, "bar_return": 0.0} for coin in COINS}
    previous = gl.cross_section(first, SCALP)
    previous["btc_lead"] = 1.0
    vector = gl.group_vector(second, SCALP, previous)
    names = gl.group_feature_names(SCALP)
    assert vector[names.index("d_mean_ofi_5s")] == pytest.approx(2.0)
    # A resolved net is not an input to the feature function.
    assert "net" not in gl.group_feature_names(SCALP)


def test_prediction_is_written_while_the_group_is_open_and_engines_stay_separate(tmp_path):
    db = str(tmp_path / "g.db")
    for index, symbol in enumerate(COINS):
        pe.open_episode(
            db,
            candidate_id=index + 1,
            engine=SCALP,
            symbol=symbol,
            setup="S",
            regime="r",
            entry_ask=10.0,
            roundtrip_cost=0.00066,
            features={"ofi_5s": 0.2},
            economics={"policy_value": 0.001},
            now=T0,
            decision_group_id="scalp-1",
        )
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT status, snapshot_json FROM policy_episodes").fetchall()
    snaps = [json.loads(raw) for _status, raw in rows]
    assert all(status == pe.OPEN for status, _raw in rows)
    assert all(snap.get("group_schema") == gl.SCHEMA for snap in snaps)
    assert all(snap["group_predictions"]["n"] == 0 for snap in snaps)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE policy_episodes SET status=?, net=?, exit_at=?, learned=1", (pe.CLOSED, 0.001, T0 + 30.0))
    for index, symbol in enumerate(COINS):
        pe.open_episode(
            db,
            candidate_id=index + 11,
            engine=SCALP,
            symbol=symbol,
            setup="S",
            regime="r",
            entry_ask=10.0,
            roundtrip_cost=0.00066,
            features={"ofi_5s": -0.1},
            economics={"policy_value": -0.002},
            now=T0 + 100.0,
            decision_group_id="scalp-2",
        )
    for index, symbol in enumerate(COINS):
        pe.open_episode(
            db,
            candidate_id=index + 21,
            engine=DAY,
            symbol=symbol,
            setup="S",
            regime="r",
            entry_ask=10.0,
            roundtrip_cost=0.00066,
            features={"bar_return": 0.01},
            economics={"policy_value": 0.001},
            now=T0 + 100.0,
            decision_group_id="day-1",
        )
    with sqlite3.connect(db) as conn:
        scalp = json.loads(conn.execute("SELECT snapshot_json FROM policy_episodes WHERE decision_group_id='scalp-2' LIMIT 1").fetchone()[0])
        day = json.loads(conn.execute("SELECT snapshot_json FROM policy_episodes WHERE decision_group_id='day-1' LIMIT 1").fetchone()[0])
        first = json.loads(conn.execute("SELECT snapshot_json FROM policy_episodes WHERE decision_group_id='scalp-1' LIMIT 1").fetchone()[0])
    assert scalp["group_predictions"]["n"] == 1
    assert day["group_predictions"]["n"] == 0
    assert first["group_predictions"]["n"] == 0


def test_no_gate_and_no_promotion():
    import re

    src = inspect.getsource(gl)
    assert "0.60" not in src
    assert re.search(r"probability\s*>", src, re.IGNORECASE) is None
    for word in ("min_hold", "min_trades", "blacklist"):
        assert re.search(rf"\b{word}\b", src, re.IGNORECASE) is None
    fund = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    assert "policy_group_learner" not in fund
    assert "policy_group_learner" not in inspect.getsource(al.day_net_expectancy)
