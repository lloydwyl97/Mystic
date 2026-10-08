"""Outside-venue state is causal, optional, and not a live entry input."""

from __future__ import annotations

import inspect
import json
import sqlite3

import pytest

from backend.services import adaptive_learning as al
from backend.services import external_policy_learn as epl
from backend.services.external_venue_feed import discovery_record
from backend.services.external_venue_tape import VenueTape
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

NOW = 100.0


def _ready(symbol: str = "BTCUSDT") -> dict:
    tapes = {symbol: {"coinbase": VenueTape(), "kraken": VenueTape()}}
    for venue in tapes[symbol].values():
        venue.add_book(NOW - 20.0, 100.0, 100.2)
        venue.add_book(NOW, 101.0, 101.2)
        venue.add_trade(NOW - 1.0, 2.0, True)
    mids = {symbol: __import__("collections").deque([(NOW - 20.0, 100.0), (NOW, 100.0)])}
    return discovery_record(tapes, mids, symbol, NOW)


def test_a_later_print_is_not_in_the_snapshot():
    tape = VenueTape()
    tape.add_book(NOW - 20.0, 100.0, 100.2)
    tape.add_book(NOW, 101.0, 101.2)
    tape.add_trade(NOW + 1.0, 9.0, True)
    feat = tape.features(NOW, 100.0, 0.0)
    assert feat is not None
    assert feat["flow_5s"] == 0.0
    assert feat["lead_5s"] == pytest.approx(feat["ret_5s"])


def test_a_stale_book_is_missing_not_zero():
    tape = VenueTape()
    tape.add_book(NOW - 30.0, 100.0, 100.2)
    tape.add_book(NOW - 20.0, 101.0, 101.2)
    assert tape.features(NOW, 100.0, 0.0) is None
    assert epl.feature_vector({"venues_fresh": 0}) is None
    assert epl.feature_vector({"venues_fresh": 1, "coinbase_ret_5s": 0.01}) is None


def test_both_venues_are_required_and_the_lead_is_the_outside_move():
    record = _ready()
    assert record is not None
    assert record["coinbase_lead_5s"] == pytest.approx(record["coinbase_ret_5s"])
    tapes = {"BTCUSDT": {"coinbase": VenueTape(), "kraken": VenueTape()}}
    tapes["BTCUSDT"]["coinbase"].add_book(NOW - 20.0, 100.0, 100.2)
    tapes["BTCUSDT"]["coinbase"].add_book(NOW, 101.0, 101.2)
    assert discovery_record(tapes, {"BTCUSDT": __import__("collections").deque([(NOW, 100.0)])}, "BTCUSDT", NOW) is None


def test_the_update_happens_once_after_the_stored_prediction(tmp_path):
    db = str(tmp_path / "e.db")
    epl.ensure_schema(db)
    features = _ready()
    frozen = epl.freeze_external(db, "SCALP_V2", features)
    assert frozen is not None and frozen["prediction"] == 0.0
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE policy_episodes (id INTEGER PRIMARY KEY, engine_id TEXT, snapshot_json TEXT)")
    snap = json.dumps({"external_discovery": frozen})
    conn.execute("INSERT INTO policy_episodes VALUES (1, 'SCALP_V2', ?)", (snap,))
    row = conn.execute("SELECT * FROM policy_episodes").fetchone()
    epl.apply_external_outcome(conn, row, 0.01, NOW)
    epl.apply_external_outcome(conn, row, 0.01, NOW + 1)
    conn.commit()
    n, taken, selected = conn.execute("SELECT n, taken, selected_sum FROM external_policy_model").fetchone()
    assert n == 1.0 and taken == 0 and selected == 0.0
    second = epl.freeze_external(db, "SCALP_V2", features)
    assert second is not None
    conn.close()


def test_outside_venues_are_not_a_live_entry_input():
    import re

    src = inspect.getsource(epl) + inspect.getsource(VenueTape)
    assert "0.60" not in src
    assert re.search(r"probability\s*>", src, re.IGNORECASE) is None
    fund = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    entry = inspect.getsource(al.day_net_expectancy)
    assert "external_policy_learn" not in fund
    assert "external_policy_learn" not in entry
    assert "external_venue_feed" not in entry
