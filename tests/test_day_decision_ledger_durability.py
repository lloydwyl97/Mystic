"""Guards for the DAY decision-ledger hole found 2026-09-18.

Three separate defects are covered:

1. day_decision_records was only written on the legacy direct-execute path. Under
   DAY_ENTRY_EXECUTION_MODE=trailing_buy the ranked-stream branch returns early, so
   the live BUY authority recorded nothing and the table sat empty for six days.
2. Structured HOLD state was a latest-per-symbol JSON snapshot, overwritten every
   cycle, so no runtime decision audit was possible after the fact.
3. That snapshot was a read-modify-write of one shared blob without an exclusive
   transaction, so concurrent per-symbol writers dropped each other's records.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from backend.services import day_decision_state as dds


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "ledger.db")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE operational_state(key TEXT PRIMARY KEY, value_json TEXT, updated_ts INTEGER)")
    dds._episode_ready.discard(path)
    return path


def _rec(symbol, category, reason, ts=None, **kw):
    return dds.build_hold_record(
        symbol=symbol,
        category=category,
        reason=reason,
        authority="test",
        now=ts if ts is not None else time.time(),
        **kw,
    )


def test_episode_extends_then_opens_new_on_state_change(db):
    for _ in range(4):
        dds.persist_hold_record(db, _rec("BTC/USDT", dds.WAITING_FOR_DIP, "WAIT_DIP"))
    dds.persist_hold_record(db, _rec("BTC/USDT", dds.TRAILING_LOW, "NEW_LOW"))
    dds.persist_hold_record(db, _rec("BTC/USDT", dds.TRAILING_LOW, "NEW_LOW"))

    eps = dds.load_hold_episodes(db, symbol="BTC/USDT")
    assert len(eps) == 2, f"expected 2 episodes, got {[e['category'] for e in eps]}"
    newest, oldest = eps[0], eps[1]
    assert newest["category"] == dds.TRAILING_LOW
    assert newest["observation_count"] == 2
    assert oldest["category"] == dds.WAITING_FOR_DIP
    assert oldest["observation_count"] == 4
    assert oldest["first_seen_ts"] <= oldest["last_seen_ts"]


def test_episodes_are_durable_across_snapshot_overwrite(db):
    """The snapshot keeps only the latest state; the ledger must keep the history."""
    for cat, reason in (
        (dds.WAITING_FOR_DIP, "WAIT_DIP"),
        (dds.TRAILING_LOW, "NEW_LOW"),
        (dds.WAITING_FOR_REBOUND, "NO_REBOUND"),
        (dds.CAPITAL_OR_SLOT_BLOCK, "INSUFFICIENT_CASH"),
    ):
        dds.persist_hold_record(db, _rec("ETH/USDT", cat, reason))

    snapshot = dds.load_hold_records(db)
    assert len(snapshot) == 1, "snapshot is latest-per-symbol by design"
    assert snapshot[0]["category"] == dds.CAPITAL_OR_SLOT_BLOCK

    cats = [e["category"] for e in dds.load_hold_episodes(db, symbol="ETH/USDT")]
    assert cats == [
        dds.CAPITAL_OR_SLOT_BLOCK,
        dds.WAITING_FOR_REBOUND,
        dds.TRAILING_LOW,
        dds.WAITING_FOR_DIP,
    ]


def test_blocks_live_execution_is_persisted(db):
    dds.persist_hold_record(db, _rec("XRP/USDT", dds.OPERATOR_CONTROL_BLOCK, "KILL_SWITCH"))
    dds.persist_hold_record(db, _rec("SOL/USDT", dds.WAITING_FOR_DIP, "WAIT_DIP"))
    by_sym = {e["symbol"]: e for e in dds.load_hold_episodes(db)}
    assert by_sym["XRP/USDT"]["blocks_live_execution"] == 1
    assert by_sym["SOL/USDT"]["blocks_live_execution"] == 0


def test_concurrent_symbols_do_not_lose_snapshot_entries(db):
    """Without BEGIN IMMEDIATE the second writer drops the first writer's symbol."""
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
    barrier = threading.Barrier(len(symbols))

    def worker(sym):
        barrier.wait()
        for _ in range(12):
            dds.persist_hold_record(db, _rec(sym, dds.WAITING_FOR_DIP, "WAIT_DIP"))

    threads = [threading.Thread(target=worker, args=(s,)) for s in symbols]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    present = {r["symbol"] for r in dds.load_hold_records(db)}
    assert present == set(symbols), f"lost snapshot entries: {set(symbols) - present}"


def test_path_ev_key_is_not_a_tradable_symbol():
    from backend.config.trading_universe import TRADING_SYMBOLS

    key = dds.PATH_EV_TELEMETRY_KEY
    assert key not in TRADING_SYMBOLS
    assert "/" not in key and key != "HOLD"
    for sym in TRADING_SYMBOLS:
        assert key != sym.replace("/", "")


def test_ranked_stream_records_every_candidate(tmp_path, monkeypatch):
    """The trailing-buy branch must record armed and non-armed ranked symbols alike."""
    from backend.services.portfolio_engine import PortfolioEngine

    captured: list[dict] = []

    def fake_record(db_path, **kw):
        captured.append(kw)

    monkeypatch.setattr("backend.services.day_gate_telemetry.record_day_decision", fake_record)

    cands = [
        SimpleNamespace(
            symbol=s,
            decision_id=f"day_{s.replace('/', '')}_1",
            decision_data={"setup_type": "HTF_TREND_PULLBACK", "buy_margin": 0.5},
        )
        for s in ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")
    ]
    engine = SimpleNamespace(db_path=str(tmp_path / "x.db"))
    bound = PortfolioEngine._record_ranked_stream_decisions.__get__(engine, PortfolioEngine)

    bound(
        ranked_candidates=cands,
        bar_timestamp=1789739280,
        arm_result={
            "trailing_buy_armed": True,
            "intents": [{"decision_id": "day_BTCUSDT_1", "intent_id": "tb1", "arm_ask": 80000.0}],
            "preserved": [],
        },
        path_ev_decision={"path_ev_winner": "HOLD"},
    )

    assert len(captured) == 4, "every ranked symbol needs a record, armed or not"
    by_sym = {c["symbol"]: c for c in captured}
    assert by_sym["BTC/USDT"]["final_decision"] == "arm"
    assert by_sym["BTC/USDT"]["detail"]["intent_id"] == "tb1"
    for sym in ("ETH/USDT", "SOL/USDT", "XRP/USDT"):
        assert by_sym[sym]["final_decision"] == "reject"
        assert by_sym[sym]["first_hard_block"] == "NOT_ARMED_THIS_BAR"
    for c in captured:
        assert c["detail"]["authority"] == "DAY_TRAILING_BUY_RANKED_STREAM"
        assert c["detail"]["path_ev_telemetry_only"] is True
        # mode must never be pinned; record_day_decision resolves the real one.
        assert "mode" not in c


def test_ranked_stream_records_hard_block_reason(tmp_path, monkeypatch):
    from backend.services.portfolio_engine import PortfolioEngine

    captured: list[dict] = []
    monkeypatch.setattr(
        "backend.services.day_gate_telemetry.record_day_decision",
        lambda _db_path, **kw: captured.append(kw),
    )

    engine = SimpleNamespace(db_path=str(tmp_path / "x.db"))
    bound = PortfolioEngine._record_ranked_stream_decisions.__get__(engine, PortfolioEngine)
    bound(
        ranked_candidates=[SimpleNamespace(symbol="BTC/USDT", decision_id="d1", decision_data={})],
        bar_timestamp=1,
        arm_result={"trailing_buy_armed": False, "intents": [], "blocked": "KILL_OR_PAUSE"},
        path_ev_decision=None,
    )
    assert captured[0]["final_decision"] == "reject"
    assert captured[0]["first_hard_block"] == "KILL_OR_PAUSE"


def test_episode_table_survives_json_payloads(db):
    rec = _rec(
        "BTC/USDT",
        dds.WAITING_FOR_DIP,
        "WAIT_DIP",
        observed={"ask": 80403.87, "action": "watch"},
        required={"min_dip_bps": 14.0, "rebound_bps": 4.0},
    )
    dds.persist_hold_record(db, rec)
    ep = dds.load_hold_episodes(db)[0]
    assert json.loads(ep["required_json"])["min_dip_bps"] == 14.0
    assert json.loads(ep["last_observed_json"])["ask"] == 80403.87
