"""Regressions for the 2026-10-04 live audit: event-loop starvation, SCALP_V2
liveness/heartbeat, live SCALP dashboard data, operator SCALP label, and
dust-aware position invariants."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import backend.services.portfolio_engine_integration as pei
import backend.services.task_health_monitor as thm
from backend.services.portfolio_engine import strategy_slot_invariants


class _SyncRedis:
    def __init__(self, hashes: dict[str, dict[str, str]] | None = None) -> None:
        self.hashes = hashes or {}

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def ping(self) -> bool:
        return True


class _AsyncRedis:
    def __init__(self, hashes: dict[str, dict[str, str]]) -> None:
        self.hashes = hashes

    async def scan(self, cursor: int = 0, match: str | None = None, count: int = 100):
        return 0, list(self.hashes)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))


def _beat_hash(age_sec: float, **extra: str) -> dict[str, dict[str, str]]:
    return {f"task_heartbeat:{thm.SCALP_V2_LIVE_LOOP_TASK}": {"last_beat_epoch": str(time.time() - age_sec), **extra}}


# --- event loop starvation -------------------------------------------------


async def test_scalp_router_work_runs_off_the_event_loop_one_call_at_a_time():
    integ = pei.PortfolioEngineIntegration.__new__(pei.PortfolioEngineIntegration)
    integ._scalp_router_executor = None
    loop_thread = threading.get_ident()
    worker_threads: list[int] = []
    active = 0
    overlapped = False
    guard = threading.Lock()

    def blocking(tag: str) -> str:
        nonlocal active, overlapped
        with guard:
            active += 1
            overlapped = overlapped or active > 1
        worker_threads.append(threading.get_ident())
        time.sleep(0.2)
        with guard:
            active -= 1
        return tag

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    ticking = asyncio.create_task(ticker())
    try:
        results = await asyncio.gather(integ._run_scalp_router(lambda: blocking("a")), integ._run_scalp_router(lambda: blocking("b")))
    finally:
        ticking.cancel()
        integ._scalp_router_executor.shutdown(wait=True)
    assert results == ["a", "b"]
    assert ticks >= 10
    assert loop_thread not in worker_threads
    assert overlapped is False


def test_momentum_sample_is_stamped_when_the_worker_runs_it(monkeypatch):
    seen: dict[str, float] = {}

    class _Router:
        def sample_momentum(self, *, epoch: float) -> int:
            seen["epoch"] = epoch
            return 4

    monkeypatch.setattr(pei.time, "time", lambda: 1234.5)
    assert pei._sample_scalp_momentum(_Router()) == 4
    assert seen["epoch"] == 1234.5


def test_scalp_cycle_blocking_work_is_offloaded():
    cycle = inspect.getsource(pei.PortfolioEngineIntegration._process_scalp_v2_signals)
    sampler = inspect.getsource(pei.PortfolioEngineIntegration._scalp_momentum_sampler_loop)
    assert re.search(r"await self\._run_scalp_router\(\s*functools\.partial\(\s*router\.evaluate_all,", cycle)
    assert not re.search(r"=\s*router\.evaluate_all\(", cycle)
    assert re.search(r"await _asyncio\.to_thread\(\s*resolve_markouts,", cycle)
    assert "await self._run_scalp_router(functools.partial(_sample_scalp_momentum, router))" in sampler
    assert "router.sample_momentum(" not in sampler


def test_markout_resolver_lookups_use_indexes(tmp_path):
    from backend.services import adaptive_learning as al

    conn = al._connect(str(tmp_path / "markouts.db"))
    try:
        key_plan = " ".join(
            str(r[-1])
            for r in conn.execute(
                "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM adaptive_candidate_markouts WHERE learned=1 AND engine_id=? AND symbol=? AND setup=? AND regime=?",
                ("SCALP_V2", "BTCUSDT", "vwap", "range"),
            )
        )
        unresolved_plan = " ".join(str(r[-1]) for r in conn.execute("EXPLAIN QUERY PLAN SELECT * FROM adaptive_candidate_markouts WHERE resolved=0 ORDER BY id ASC LIMIT 200"))
    finally:
        conn.close()
    assert "idx_adaptive_markouts_key" in key_plan
    assert "idx_adaptive_markouts_unresolved" in unresolved_plan


# --- SCALP_V2 heartbeat / task health --------------------------------------


async def test_scalp_v2_loop_beats_only_after_a_completed_cycle(monkeypatch):
    integ = pei.PortfolioEngineIntegration.__new__(pei.PortfolioEngineIntegration)
    integ._scalp_v2_entry_interval = 60
    integ._scalp_v2_halt_reason = ""
    integ._scalp_v2_halt_until = ""
    integ.redis_client = object()
    cycles = 0

    async def _process() -> None:
        nonlocal cycles
        cycles += 1
        if cycles == 1:
            raise RuntimeError("cycle failed")
        integ._scalp_v2_halt_reason = "CONSECUTIVE_LOSSES"

    beats: list[tuple[str, dict]] = []

    async def _beat(task, client, *, extra=None, ttl_sec=600):
        beats.append((task, dict(extra or {})))

    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _sleep(seconds, *args, **kwargs):
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            raise asyncio.CancelledError
        await real_sleep(0)

    integ._process_scalp_v2_signals = _process
    monkeypatch.setattr(thm, "beat", _beat)
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    await integ._scalp_v2_live_loop()
    assert cycles == 2
    assert beats == [(thm.SCALP_V2_LIVE_LOOP_TASK, {"interval_sec": 60, "halt_reason": "CONSECUTIVE_LOSSES", "halt_until": ""})]


def test_task_health_watches_the_scalp_v2_loop_not_the_retired_runner():
    assert thm.SCALP_V2_LIVE_LOOP_TASK in thm.CRITICAL_TASK_THRESHOLDS_SEC
    assert "scalp_runner:tick" not in thm.CRITICAL_TASK_THRESHOLDS_SEC


async def test_task_health_ok_when_live_tasks_beat_and_degraded_without_scalp_v2():
    fresh = {f"task_heartbeat:{name}": {"last_beat_epoch": str(time.time() - 5)} for name in thm.CRITICAL_TASK_THRESHOLDS_SEC}
    report = await thm.get_task_health(_AsyncRedis(fresh))
    assert report["overall_status"] == "OK"
    fresh.pop(f"task_heartbeat:{thm.SCALP_V2_LIVE_LOOP_TASK}")
    report = await thm.get_task_health(_AsyncRedis(fresh))
    assert report["overall_status"] == "DEGRADED"
    assert report["unknown_critical_tasks"] == [thm.SCALP_V2_LIVE_LOOP_TASK]


def test_sync_heartbeat_reader():
    assert thm.heartbeat_age_sync(thm.SCALP_V2_LIVE_LOOP_TASK, None) is None
    assert thm.heartbeat_age_sync(thm.SCALP_V2_LIVE_LOOP_TASK, _SyncRedis()) is None
    age = thm.heartbeat_age_sync(thm.SCALP_V2_LIVE_LOOP_TASK, _SyncRedis(_beat_hash(12.0)))
    assert age is not None
    assert 11.0 <= age <= 20.0


async def test_process_health_reports_scalp_v2_loop_from_heartbeat(monkeypatch):
    import backend.routes.system_health as sh

    monkeypatch.setattr(sh, "_process_running", lambda _pattern: False)
    monkeypatch.setattr(sh, "get_shared_redis_sync", lambda: _SyncRedis(_beat_hash(30.0)))
    loop = (await sh.get_process_health())["optional_processes"]["scalp_v2_live_loop"]
    assert loop["running"] is True
    assert loop["classification"] == "in_process"
    monkeypatch.setattr(sh, "get_shared_redis_sync", lambda: _SyncRedis(_beat_hash(900.0)))
    assert (await sh.get_process_health())["optional_processes"]["scalp_v2_live_loop"]["running"] is False


# --- operator SCALP label ---------------------------------------------------


def test_operator_scalp_label_follows_structural_mode(monkeypatch):
    from backend.services.operator_account_status import account_operator_labels

    for key in ("SCALP_STRUCTURAL_MODE", "SCALP_TRADING_MODE", "SCALP_MODE", "BINANCE_SCALP_MODE", "SCALP_ALLOW_MARKET_ORDERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SCALP_LIVE", "true")
    monkeypatch.setenv("SCALP_LIVE_ARMED", "true")
    labels = account_operator_labels()
    assert labels["scalp_mode_display"] == "SCALP LIVE"
    assert labels["operator_mode_labels"]["scalp"] == "SCALP LIVE"
    monkeypatch.setenv("SCALP_LIVE_ARMED", "false")
    assert account_operator_labels()["scalp_mode_display"] == "SCALP PAPER"


# --- position invariants ----------------------------------------------------


def _lot(symbol: str, engine: str, status: str = "ACTIVE", qty: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, engine_id=engine, status=status, quantity=qty)


PRODUCTION_DUST = [
    ("XRP/USDT", "DAY_V2", "DUST_PENDING", 0.07096),
    ("XRP/USDT", "SCALP_V2", "DUST_PENDING", 0.07662),
    ("SOL/USDT", "SCALP_V2", "DUST_PENDING", 0.0007776),
    ("ETH/USDT", "SCALP_V2", "DUST_PENDING", 8.964e-05),
]


def test_slot_invariants_ignore_dust_and_allow_cross_engine_symbols():
    dust = strategy_slot_invariants((sym, _lot(sym, eng, st, q)) for sym, eng, st, q in PRODUCTION_DUST)
    assert dust["position_limit"] == {
        "ok": True,
        "current": 0,
        "max": 8,
        "by_engine": {"DAY_V2": {"current": 0, "max": 4}, "SCALP_V2": {"current": 0, "max": 4}},
    }
    assert dust["no_stacking"] == {"ok": True, "symbols": []}

    cross = strategy_slot_invariants([("BTC/USDT", _lot("BTC/USDT", "DAY_V2")), ("BTC/USDT", _lot("BTC/USDT", "SCALP_V2"))])
    assert cross["no_stacking"]["ok"] is True
    assert cross["position_limit"]["current"] == 2

    stacked = strategy_slot_invariants([("BTC/USDT", _lot("BTC/USDT", "SCALP_V2")), ("BTC/USDT", _lot("BTC/USDT", "SCALP_V2"))])
    assert stacked["no_stacking"]["ok"] is False

    five_scalp = strategy_slot_invariants((s, _lot(s, "SCALP_V2")) for s in ("A/USDT", "B/USDT", "C/USDT", "D/USDT", "E/USDT"))
    assert five_scalp["position_limit"]["ok"] is False


def test_engine_invariants_status_counts_slots_not_dust(tmp_path):
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine(db_path=str(tmp_path / "inv.db"), principal=500.0, test_mode=True)
    engine.open_positions = {f"{eng}::{sym}": _lot(sym, eng, st, q) for sym, eng, st, q in PRODUCTION_DUST}
    inv = engine.get_invariants_status()
    assert inv["position_limit"]["current"] == 0
    assert inv["position_limit"]["max"] == 8
    assert inv["no_stacking"]["ok"] is True
    assert "position_limit" not in inv["snapshot_failed_keys"]
    assert "no_stacking" not in inv["snapshot_failed_keys"]


async def test_invariants_detail_override_excludes_dust_rows(monkeypatch, tmp_path):
    import backend.endpoints.portfolio_engine_endpoints as pee

    db = tmp_path / "positions.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE portfolio_engine_positions (symbol TEXT, quantity REAL, engine_id TEXT, status TEXT)")
    conn.executemany("INSERT INTO portfolio_engine_positions (symbol, engine_id, status, quantity) VALUES (?,?,?,?)", PRODUCTION_DUST)
    conn.commit()
    conn.close()

    class _Engine:
        def get_invariants_status(self) -> dict:
            return {
                "all_ok": False,
                "position_limit": {"ok": False, "current": 4, "max": 4},
                "no_stacking": {"ok": False, "symbols": []},
                "snapshot_failed_keys": ["position_limit", "no_stacking"],
                "snapshot_failed_count": 2,
            }

    monkeypatch.setattr(pee, "get_portfolio_engine", _Engine)
    monkeypatch.setattr(pee, "DATABASE_PATH", str(db))
    data = (await pee.get_invariants_detail())["data"]
    assert data["position_limit"]["current"] == 0
    assert data["position_limit"]["max"] == 8
    assert data["no_stacking"] == {"ok": True, "symbols": []}
    assert data["all_ok"] is True
    assert data["snapshot_failed_keys"] == []

    conn = sqlite3.connect(db)
    conn.executemany(
        "INSERT INTO portfolio_engine_positions (symbol, engine_id, status, quantity) VALUES (?,?,?,?)",
        [("BTC/USDT", "SCALP_V2", "ACTIVE", 0.001), ("BTC/USDT", "SCALP_V2", "ACTIVE", 0.002)],
    )
    conn.commit()
    conn.close()
    data = (await pee.get_invariants_detail())["data"]
    assert data["no_stacking"]["ok"] is False
    assert data["all_ok"] is False
    assert data["snapshot_failed_keys"] == ["no_stacking"]


# --- live SCALP dashboard ---------------------------------------------------


@pytest.fixture
def live_db(tmp_path):
    from backend.database_schema import initialize_paper_trading_schema
    from backend.services.portfolio_engine import PortfolioEngine
    from backend.services.scalp_v2 import opportunity
    from backend.services.scalp_v2.decision_log import record_scalp_decision

    db = str(tmp_path / "live.db")
    engine = PortfolioEngine(db_path=db, principal=500.0, test_mode=True)
    engine._ensure_db_schema()
    initialize_paper_trading_schema(db)
    now = datetime.now(timezone.utc)
    today = now.isoformat()
    yesterday = (now - timedelta(days=1)).isoformat()

    def _created(iso: str) -> str:
        return datetime.fromisoformat(iso).strftime("%Y-%m-%d %H:%M:%S")

    trades = [
        # side, symbol, pnl, ts, engine, counts, synthetic, opp, exit, hold, fees, qty, price
        ("BUY", "BTC/USDT", None, today, "SCALP_V2", 1, 0, "o-a", None, None, 0.05, 0.001, 60000.0),
        ("SELL", "BTC/USDT", 0.50, today, "SCALP_V2", 1, 0, "o-a", "NET_PROFIT_EXIT", 240.0, 0.05, 0.001, 60600.0),
        ("SELL", "ETH/USDT", -0.20, today, "SCALP_V2", 1, 0, "o-b", "STOP_LOSS_EXIT", 45.0, 0.04, 0.02, 2500.0),
        ("SELL", "SOL/USDT", -5.00, today, "SCALP_V2", 0, 0, "o-c", "DUST_WRITEOFF", None, 0.0, 0.1, 150.0),
        ("SELL", "XRP/USDT", -7.00, today, "SCALP_V2", 1, 1, "o-d", "SYNTHETIC", None, 0.0, 10.0, 2.0),
        ("SELL", "BTC/USDT", 3.00, today, "DAY_V2", 1, 0, "", "NET_PROFIT_EXIT", None, 0.1, 0.001, 61000.0),
        ("SELL", "XRP/USDT", -0.10, yesterday, "SCALP_V2", 1, 0, "o-e", "TIME_STOP_EXIT", 700.0, 0.03, 20.0, 2.5),
    ]
    conn = sqlite3.connect(db)
    have = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
    production_cols = {
        "is_synthetic": "INTEGER",
        "engine_id": "TEXT",
        "scalp_opportunity_id": "TEXT",
        "counts_toward_realized": "INTEGER DEFAULT 1",
        "hold_time_seconds": "REAL",
        "regime": "TEXT",
        "fees_paid": "REAL",
    }
    for name, decl in production_cols.items():
        if name not in have:
            conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {name} {decl}")
    for i, (side, sym, pnl, ts, eng, counts, synth, opp, exit_reason, hold, fees, qty, price) in enumerate(trades):
        conn.execute(
            "INSERT INTO paper_trades (trade_id, paper_run_id, symbol, side, quantity, price, pnl, timestamp, created_at, exit_reason, fees_paid, "
            "is_synthetic, engine_id, scalp_opportunity_id, counts_toward_realized, hold_time_seconds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"t{i}", "live", sym, side, qty, price, pnl, ts, _created(ts), exit_reason, fees, synth, eng, opp, counts, hold),
        )
    conn.executemany(
        "INSERT INTO portfolio_engine_positions (symbol, quantity, entry_price, entry_time, trade_id, stop_price, take_profit_1_price, take_profit_2_price, "
        "highest_price, atr_at_entry, last_updated, engine_id, status, scalp_opportunity_id) VALUES (?,?,?,?,?,0,0,0,?,0,'',?,?,?)",
        [
            ("BTC/USDT", 0.001, 60000.0, time.time() - 120, "p1", 60000.0, "SCALP_V2", "ACTIVE", "o-btc"),
            ("XRP/USDT", 0.07, 2.5, time.time() - 900, "p2", 2.5, "SCALP_V2", "DUST_PENDING", "o-x"),
            ("ETH/USDT", 0.02, 2500.0, time.time() - 600, "p3", 2500.0, "DAY_V2", "ACTIVE", ""),
        ],
    )
    opportunity._ensure(conn)
    conn.executemany(
        "INSERT INTO scalp_v2_opportunities (symbol, opportunity_id, setup_family, structural_anchor, state, engine_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        [("BTC/USDT", "o-btc", "vwap", "", "FILLED", "SCALP_V2", 0.0, 0.0), ("BTC/USDT", "o-a", "range", "", "CLOSED", "SCALP_V2", 0.0, 0.0)],
    )
    conn.commit()
    conn.close()
    record_scalp_decision(db, "BTC/USDT", "FILLED", "FILLED", cycle_ts=1000.0)
    for sym, result, reason in (
        ("BTC/USDT", "REJECTED:NO_EXECUTABLE_EDGE_ESTIMATE", "NO_EXECUTABLE_EDGE_ESTIMATE"),
        ("ETH/USDT", "REJECTED:NO_EXECUTABLE_EDGE_ESTIMATE", "NO_EXECUTABLE_EDGE_ESTIMATE"),
        ("SOL/USDT", "REJECTED:SYMBOL_OCCUPIED", "SYMBOL_OCCUPIED"),
        ("XRP/USDT", "REJECTED:NO_EXECUTABLE_EDGE_ESTIMATE", "NO_EXECUTABLE_EDGE_ESTIMATE"),
    ):
        record_scalp_decision(db, sym, result, reason, cycle_ts=2000.0)
    return db


@pytest.fixture
def scalp_live_env(monkeypatch):
    for key in ("SCALP_STRUCTURAL_MODE", "SCALP_ALLOW_MARKET_ORDERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SCALP_LIVE", "true")
    monkeypatch.setenv("SCALP_LIVE_ARMED", "true")


def test_live_status_reports_heartbeat_decisions_lots_and_realized_pnl(live_db, scalp_live_env):
    from backend.services.scalp_v2.live_dashboard import live_status

    st = live_status(live_db, _SyncRedis(_beat_hash(5.0, halt_reason="", halt_until="")))
    assert st["runner_active"] is True
    assert st["structural_mode"] == "LIVE"
    assert st["entry_armed"] is True
    assert st["open_scalp_positions"] == 1
    assert st["operational_summary"] == {"operational_mode": "entry_scan_active", "entry_blocked_reason": None}
    assert st["overall_decision"] == "NO_ENTRY"
    assert st["top_blocker"] == "NO_EXECUTABLE_EDGE_ESTIMATE"
    assert st["latest_cycle"]["cycle_ts"] == 2000.0
    assert st["pnl_summary"]["today"] == {"sells": 2, "realized_pnl_usd": 0.3, "wins": 1, "losses": 1}
    assert st["pnl_summary"]["all_time"]["sells"] == 3
    assert st["pnl_summary"]["all_time"]["realized_pnl_usd"] == pytest.approx(0.2)
    assert st["structural_breaker"]["status"] == "CLOSED"
    assert "note" not in st


def test_live_status_flags_stale_missing_and_halted_loops(live_db, scalp_live_env):
    from backend.services.scalp_v2.live_dashboard import live_status

    stale = live_status(live_db, _SyncRedis(_beat_hash(1000.0)))
    assert stale["runner_active"] is False
    assert stale["operational_summary"]["operational_mode"] == "runner_dead"
    assert stale["note"]

    missing = live_status(live_db, _SyncRedis())
    assert missing["operational_summary"]["entry_blocked_reason"] == "SCALP_V2_LOOP_NEVER_BEAT"

    halted = live_status(live_db, _SyncRedis(_beat_hash(5.0, halt_reason="CONSECUTIVE_LOSSES", halt_until="2026-10-05T01:00:00+00:00")))
    assert halted["runner_active"] is True
    assert halted["entry_armed"] is False
    assert halted["operational_summary"] == {"operational_mode": "breaker_halt", "entry_blocked_reason": "CONSECUTIVE_LOSSES"}
    assert halted["structural_breaker"] == {"status": "OPEN", "reason": "CONSECUTIVE_LOSSES", "recovery_until": "2026-10-05T01:00:00+00:00"}


def test_scalp_status_endpoint_serves_live_view(live_db, scalp_live_env, monkeypatch):
    import backend.endpoints.scalp_status_endpoints as se

    monkeypatch.setattr(se, "_live_db_path", lambda: live_db)
    monkeypatch.setattr(se, "_live_redis", lambda: _SyncRedis(_beat_hash(5.0)))
    st = se.scalp_status()
    assert st["runner_active"] is True
    assert st["source"] == "scalp_v2_live"
    assert st["pnl_summary"]["today"]["sells"] == 2


def test_live_positions_trades_scoreboard_attribution_and_learning_count(live_db, monkeypatch):
    import backend.endpoints.scalp_status_endpoints as se

    monkeypatch.setattr(se, "_live_db_path", lambda: live_db)

    pos = se.scalp_positions()
    assert pos["open_count"] == 1
    assert pos["positions"][0]["symbol"] == "BTC/USDT"
    assert pos["positions"][0]["setup"] == "vwap"
    assert 100 <= pos["positions"][0]["hold_seconds"] <= 200

    trades = se.scalp_trades(limit=50, days=None)
    assert trades["count"] == 5
    assert {t["symbol"] for t in trades["trades"]} == {"BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"}
    assert all(t["created_at"] for t in trades["trades"])
    assert se.scalp_trades(limit=2, days=None)["count"] == 2

    board = se.scalp_scoreboard(days=7)["rows"]
    assert [(r["trades"], r["wins"], r["losses"]) for r in board] == [(2, 1, 1), (1, 0, 1)]
    assert board[0]["net_pnl"] == pytest.approx(0.3)

    attr = se.scalp_attribution(days=None)
    assert attr["closed_sells"] == 3
    assert attr["total_net_pnl_usd"] == pytest.approx(0.2)
    assert {row["key"] for row in attr["by_setup"]} == {"range", "unknown"}
    assert {row["key"] for row in attr["by_fee_burden"]} == {"moderate (0.1-0.2%)", "unknown"}

    from backend.services.scalp_v2.live_dashboard import live_closed_sell_count

    assert live_closed_sell_count(live_db) == 3


# ------------------------------------------- canonical weekly completed-bar window


def test_weekly_completed_bar_is_the_last_closed_monday_bar_every_hour_of_the_week():
    from backend.config.canonical_candle_intervals import align_open_ms, interval_ms
    from backend.services.canonical_candle_pipeline import CanonicalCandlePipeline

    pipe = CanonicalCandlePipeline()
    pipe._cached_start_ms = 0
    week = interval_ms("1w")
    monday = int(datetime(2026, 10, 5, tzinfo=timezone.utc).timestamp() * 1000)
    for hour in range(-24 * 7, 24 * 7):
        now_ms = monday + hour * 3_600_000 + 7 * 60_000
        pipe._now_ms = lambda now_ms=now_ms: now_ms
        completed = pipe._completed_open_ms("1w")
        assert completed == align_open_ms(completed, "1w"), "weekly bars open Monday 00:00 UTC"
        assert completed + week <= now_ms < completed + 2 * week, "must be the newest closed weekly bar"
        assert pipe._recent_window_start_ms("1w") <= completed, "integrity/gap-repair window must not be empty"
        for tf in ("1m", "15m", "1h", "4h", "1d"):
            width = interval_ms(tf)
            assert pipe._completed_open_ms(tf) == (now_ms // width) * width - width


@pytest.mark.asyncio
async def test_weekly_depth_check_counts_the_newest_closed_bar_on_monday(monkeypatch):
    from backend.config.canonical_candle_intervals import align_open_ms, interval_ms
    from backend.services import canonical_candle_pipeline as ccp

    week = interval_ms("1w")
    monday_now = int(datetime(2026, 10, 5, 0, 7, tzinfo=timezone.utc).timestamp() * 1000)
    newest_closed = align_open_ms(monday_now, "1w") - week
    stored: dict[str, set[int]] = {"1d": set(), "1w": {newest_closed - i * week for i in range(ccp.htf_history_depth("1w"))}}
    day = interval_ms("1d")
    stored["1d"] = {align_open_ms(monday_now, "1d") - (i + 1) * day for i in range(ccp.htf_history_depth("1d"))}

    def _load(_symbol, interval, *, start_ms, end_ms):
        return [{"open_ms": ts} for ts in sorted(stored[interval]) if start_ms <= ts <= end_ms]

    async def _never_backfill(*_a, **_k):
        raise AssertionError("store already holds full depth; nothing to backfill")

    monkeypatch.setattr(ccp, "load_aligned_candles", _load)
    pipe = ccp.CanonicalCandlePipeline()
    monkeypatch.setattr(pipe, "_now_ms", lambda: monday_now)
    monkeypatch.setattr(pipe, "backfill_range", _never_backfill)

    out = await pipe.ensure_htf_history_depth(["BTCUSDT"], force=True)

    by_tf = {row["interval"]: row for row in out}
    assert by_tf["1w"]["before"] == ccp.htf_history_depth("1w")
    assert {row["action"] for row in out} == {"ok"}


# ------------------------------------------- order-book collector backlog / stale depth


def _depth_frame(stream: str, uid: int, bid: float) -> str:
    return json.dumps({"stream": stream, "data": {"lastUpdateId": uid, "bids": [[str(bid), "1.0"]], "asks": [[str(bid + 0.1), "1.0"]]}})


def _fake_ws_connect(frames_source):
    class _Socket:
        async def send(self, _msg) -> None:
            return None

        def __aiter__(self):
            return frames_source()

    class _Connect:
        async def __aenter__(self):
            return _Socket()

        async def __aexit__(self, *_exc) -> bool:
            return False

    return lambda *_a, **_k: _Connect()


@pytest.mark.asyncio
async def test_order_book_collector_publishes_the_newest_snapshot_after_a_backlog(monkeypatch):
    from backend.services import order_book_collector as obc

    queued = [json.dumps({"result": None, "id": 1})]
    queued += [_depth_frame("solusdt@depth20@100ms", uid, 121.0 + uid / 1000) for uid in range(1, 201)]
    queued += [_depth_frame("btcusdt@depth20@100ms", uid, 60000.0 + uid) for uid in range(1, 4)]

    async def _frames():
        for frame in queued:
            yield frame
        await asyncio.sleep(0.2)

    monkeypatch.setattr(obc.websockets, "connect", _fake_ws_connect(_frames))
    processed: list[tuple[str, int]] = []

    async def _process(message):
        payload = json.loads(message)
        processed.append((payload["stream"], payload["data"]["lastUpdateId"]))
        await asyncio.sleep(0.01)

    collector = obc.OrderBookCollector()
    collector.is_running = True
    monkeypatch.setattr(collector, "_process_message", _process)

    await collector._connect_and_listen()

    assert sorted(processed) == [("btcusdt@depth20@100ms", 3), ("solusdt@depth20@100ms", 200)]
    assert collector.stats["messages_received"] == 203
    assert collector.stats["snapshots_superseded"] == 201


@pytest.mark.asyncio
async def test_order_book_collector_keeps_draining_the_socket_while_processing_is_slow(monkeypatch):
    from backend.services import order_book_collector as obc

    received_at: list[int] = []

    async def _frames():
        for uid in range(1, 101):
            received_at.append(uid)
            yield _depth_frame("ethusdt@depth20@100ms", uid, 2700.0 + uid / 100)
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.3)

    monkeypatch.setattr(obc.websockets, "connect", _fake_ws_connect(_frames))
    processed: list[int] = []
    pulled_when_processing: list[int] = []

    async def _slow_process(message):
        pulled_when_processing.append(len(received_at))
        processed.append(json.loads(message)["data"]["lastUpdateId"])
        await asyncio.sleep(0.05)

    collector = obc.OrderBookCollector()
    collector.is_running = True
    monkeypatch.setattr(collector, "_process_message", _slow_process)

    await collector._connect_and_listen()

    assert processed[-1] == 100, "the newest book must be the one left published"
    assert processed == sorted(processed), "an older snapshot must never be processed after a newer one"
    assert len(processed) < 25, "slow processing must not force the reader to consume frames one by one"
    assert pulled_when_processing[-1] == 100


def test_stream_name_reads_the_combined_stream_envelope():
    from backend.services.order_book_collector import _stream_name

    assert _stream_name(_depth_frame("xrpusdt@depth20@100ms", 7, 1.5)) == "xrpusdt@depth20@100ms"
    assert _stream_name(_depth_frame("xrpusdt@depth20@100ms", 7, 1.5).encode()) == "xrpusdt@depth20@100ms"
    compact = '{"stream":"btcusdt@depth20@100ms","data":{"lastUpdateId":1,"bids":[["1","1"]],"asks":[["2","1"]]}}'
    assert _stream_name(compact) == "btcusdt@depth20@100ms"
    assert _stream_name(json.dumps({"result": None, "id": 1})) is None
    assert _stream_name('{"stream":null,"data":{"x":"y"}}') is None


def test_publish_ws_depth_reuses_one_redis_client(monkeypatch):
    from backend.services.binance_scalp import market_reader

    class _Mem:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, ex=None):
            self.store[key] = value

    mem = _Mem()
    created: list[int] = []

    def _from_url(*_a, **_k):
        created.append(1)
        return mem

    monkeypatch.setattr(market_reader.redis, "from_url", _from_url)
    monkeypatch.setattr(market_reader, "_WS_DEPTH_REDIS", None)
    for uid in range(1, 41):
        market_reader.publish_ws_depth("SOLUSDT", [[121.8, 1.0]], [[121.9, 1.0]], last_update_id=uid)

    assert len(created) == 1
    assert json.loads(mem.store["scalp:ws_depth:SOLUSDT"])["last_update_id"] == 40


# ------------------------------------------- provable fill identity backfill


def _recon_fill(order: str, trade: str, side: str, qty: float, cost: float, ts_ms: int) -> dict:
    return {"symbol": "XRP/USDT", "id": trade, "order": order, "side": side, "qty": qty, "cost": cost, "ts": ts_ms, "fee_cost": 0.0099, "fee_ccy": "USDT"}


def test_backfill_records_the_venue_timestamp_from_reconciliation_fills(tmp_path, caplog):
    from backend.services.live_exchange_equity import backfill_provable_fill_identities
    from backend.services.live_order_identity import fills_for_order

    db = str(tmp_path / "fills.db")
    ts_ms = int(datetime(2026, 10, 2, 14, 26, 21, tzinfo=timezone.utc).timestamp() * 1000)
    recorded = [{"symbol": "XRP/USDT", "side": "SELL", "order_id": "511504197", "trade_id": "mystic_sell_XRP/USDT_1790951179571"}]
    with caplog.at_level("ERROR"):
        written = backfill_provable_fill_identities(db, recorded=recorded, venue_fills=[_recon_fill("511504197", "90001", "SELL", 9.8, 14.996, ts_ms)])

    assert [w["exchange_order_id"] for w in written] == ["511504197"]
    rows = fills_for_order(db, "511504197")
    assert len(rows) == 1
    assert datetime.fromisoformat(rows[0]["event_ts_exchange"]) == datetime(2026, 10, 2, 14, 26, 21, tzinfo=timezone.utc)
    assert "LIVE_FILL_IDENTITY_INCOMPLETE" not in caplog.text


def test_backfill_never_duplicates_an_order_the_live_path_already_recorded(tmp_path, caplog, monkeypatch):
    import backend.services.live_order_identity as identity_mod
    from backend.services import live_exchange_equity
    from backend.services.live_order_identity import OrderIdentity, fills_for_order, record_fill

    db = str(tmp_path / "fills.db")
    record_fill(
        db,
        OrderIdentity(
            symbol="XRP/USDT",
            side="SELL",
            exchange_order_id="511504197",
            fill_ids=["90001", "90002"],
            venue_trade_ids=["90001", "90002"],
            executed_qty=32.5,
            avg_fill_price=1.53027,
            cost_quote=49.73,
            fee_amount=0.00994675,
            fee_asset="USDT",
            mystic_trade_id="scalp_v2_XRPUSDT_1790951065796",
        ),
    )
    caplog.clear()
    ts_ms = int(datetime(2026, 10, 2, 14, 26, 21, tzinfo=timezone.utc).timestamp() * 1000)
    recorded = [{"symbol": "XRP/USDT", "side": "SELL", "order_id": "511504197", "trade_id": "mystic_sell_XRP/USDT_1790951179571"}]
    fills = [_recon_fill("511504197", "90001", "SELL", 22.7, 34.74, ts_ms)]

    attempts: list[str] = []

    def _counting_record(path, identity):
        attempts.append(identity.exchange_order_id)
        return record_fill(path, identity)

    monkeypatch.setattr(identity_mod, "record_fill", _counting_record)
    with caplog.at_level("ERROR"):
        first = live_exchange_equity.backfill_provable_fill_identities(db, recorded=recorded, venue_fills=fills)
        second = live_exchange_equity.backfill_provable_fill_identities(db, recorded=recorded, venue_fills=fills)

    rows = fills_for_order(db, "511504197")
    assert len(rows) == 1, "the live-path SELL identity must stay the only row for this venue order"
    assert rows[0]["mystic_trade_id"] == "scalp_v2_XRPUSDT_1790951065796"
    assert datetime.fromisoformat(rows[0]["event_ts_exchange"]) == datetime(2026, 10, 2, 14, 26, 21, tzinfo=timezone.utc)
    assert attempts == []
    assert first == second == [{"symbol": "XRP/USDT", "side": "SELL", "exchange_order_id": "511504197", "venue_trade_id": "90001"}]
    assert "LIVE_FILL_IDENTITY_INCOMPLETE" not in caplog.text


def test_blank_timestamp_fill_never_overwrites_a_recorded_venue_time(tmp_path):
    from backend.services.live_order_identity import OrderIdentity, fill_blank_exchange_timestamps, fills_for_order, record_fill, recorded_order_sides

    db = str(tmp_path / "fills.db")
    record_fill(
        db,
        OrderIdentity(
            symbol="SOL/USDT",
            side="BUY",
            exchange_order_id="77",
            fill_ids=["1"],
            executed_qty=1.0,
            avg_fill_price=121.0,
            event_ts_exchange="2026-10-04T22:02:59.039000+00:00",
            mystic_trade_id="scalp_v2_SOLUSDT_1",
        ),
    )
    assert recorded_order_sides(db) == {("77", "BUY"): False}
    assert fill_blank_exchange_timestamps(db, [("2030-01-01T00:00:00+00:00", "77", "BUY")]) == 0
    assert fills_for_order(db, "77")[0]["event_ts_exchange"] == "2026-10-04T22:02:59.039000+00:00"
