"""Research-only policy episodes. No orders, cash, positions, or live entry authority.

Every recorded candidate can open a virtual position at the executable ask from
that decision. Later checks call the same continuation function a live position
uses and store the advantage, terminal, and action. A replay reads those rows.
It does not recompute them against a later model.

A candidate that actually fills is closed from its accounting realized net and
tagged REAL_POLICY_OUTCOME. Every other resolution is COUNTERFACTUAL_POLICY_OUTCOME.
Neither tag is account P&L. ``direct_policy_net`` is updated from the resolved
net and is not an input to entry ranking.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from backend.services.adaptive_learning import (
    DAY_ENGINE,
    SCALP_ENGINE,
    _connect,
    _fold_observation,
    continuation_terminal,
    current_economic_version,
    current_strategy_version,
    executable_bid_net,
    learned_hold_or_exit,
)
from backend.services.continuation_surface import ADVANTAGE_VERSION, horizons_for, installed_aggregator, state_features

logger = logging.getLogger(__name__)

REAL = "REAL_POLICY_OUTCOME"
COUNTERFACTUAL = "COUNTERFACTUAL_POLICY_OUTCOME"
OPEN = "OPEN"
CLOSED = "CLOSED"
EXIT_REASON = "LEARNED_CONTINUATION_EXIT"
# One virtual check is enough to see the exit that live positions see on the
# next monitor pass. A second check inside this window is the same pass.
MIN_CHECK_GAP_SEC = 20.0
ADVANCE_LIMIT = 40

EPISODE_DDL = """
CREATE TABLE IF NOT EXISTS policy_episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    engine_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    setup TEXT NOT NULL,
    regime TEXT NOT NULL,
    decided_at REAL NOT NULL,
    entry_ask REAL NOT NULL,
    roundtrip_cost REAL NOT NULL,
    snapshot_json TEXT NOT NULL,
    status TEXT NOT NULL,
    exit_bid REAL,
    exit_at REAL,
    exit_reason TEXT,
    gross REAL,
    net REAL,
    mfe REAL NOT NULL DEFAULT 0,
    mae REAL NOT NULL DEFAULT 0,
    hold_sec REAL,
    learned INTEGER NOT NULL DEFAULT 0
)
"""
CHECK_DDL = """
CREATE TABLE IF NOT EXISTS policy_episode_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER NOT NULL,
    checked_at REAL NOT NULL,
    unrealized REAL NOT NULL,
    advantage REAL,
    terminal REAL,
    action TEXT NOT NULL,
    UNIQUE (episode_id, checked_at)
)
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(EPISODE_DDL)
    conn.execute(CHECK_DDL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_episodes_open ON policy_episodes(status, decided_at)")


def storage_estimate(*, episodes_per_hour: float = 80.0, checks_per_episode: float = 2.0) -> dict[str, float]:
    """Rough size of the research tables. Not a cap and not a trading limit."""
    row_bytes = 1200.0 + checks_per_episode * 180.0
    mb_hour = episodes_per_hour * row_bytes / 1e6
    return {"rows_per_hour": episodes_per_hour, "mb_per_hour": mb_hour, "gb_per_day": mb_hour * 24.0 / 1000.0}


def _horizon_snapshot(db_path: str, engine: str, symbol: str, setup: str, regime: str, now: float) -> dict[str, dict[str, float]]:
    from backend.services.adaptive_learning import estimate

    out: dict[str, dict[str, float]] = {}
    for horizon in horizons_for(engine):
        view = estimate(db_path, engine, symbol, setup, regime, f"hold_adv_{int(horizon)}", now=now, include_engine=True)
        out[str(int(horizon))] = {"mean": float(view["mean"]), "evidence": float(view.get("level_weights", {}).get("engine") or 0.0)}
    return out


def entry_snapshot(
    db_path: str,
    *,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    economics: dict[str, Any] | None,
    rank_position: Any,
    size_mult: Any,
    now: float,
) -> dict[str, Any]:
    """Immutable description of the continuation policy at the decision.

    Horizon means are the posteriors a decision at ``now`` would read. Later
    checks store their own advantage instead of rereading these means.
    """
    econ = economics if isinstance(economics, dict) else {}
    horizons = _horizon_snapshot(db_path, engine, symbol, setup, regime, now)
    version, source = "", ""
    try:
        meta = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        found = meta.execute(
            "SELECT learning_version, source FROM continuation_learning_meta WHERE engine_id=? AND economic_version=?",
            (str(engine), current_economic_version(engine)),
        ).fetchone()
        meta.close()
        if found:
            version, source = str(found[0] or ""), str(found[1] or "")
    except sqlite3.Error:
        found = None
    return {
        "economic_version": current_economic_version(engine),
        "strategy_version": current_strategy_version(engine),
        "continuation_version": version,
        "continuation_source": source,
        "advantage_authority": version == ADVANTAGE_VERSION,
        "aggregator": installed_aggregator(db_path, engine),
        "horizons": horizons,
        "issued": {
            "market_alpha": econ.get("market_alpha"),
            "policy_gap": econ.get("policy_gap"),
            "uncalibrated_policy_value": econ.get("uncalibrated_policy_value"),
            "policy_calibration": econ.get("policy_calibration"),
            "policy_value": econ.get("policy_value"),
            "rank_position": rank_position,
            "size_mult": size_mult,
        },
    }


def open_episode(
    db_path: str,
    *,
    candidate_id: int,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    entry_ask: float,
    roundtrip_cost: float,
    economics: dict[str, Any] | None = None,
    rank_position: Any = None,
    size_mult: Any = None,
    now: float | None = None,
) -> int | None:
    """Open one research episode. Invalid asks are skipped. A second call is a no-op."""
    ask = float(entry_ask or 0.0)
    if candidate_id is None or ask <= 0.0 or not (ask < float("inf")):
        return None
    moment = float(now if now is not None else time.time())
    snap = entry_snapshot(
        db_path,
        engine=engine,
        symbol=symbol,
        setup=setup,
        regime=regime,
        economics=economics,
        rank_position=rank_position,
        size_mult=size_mult,
        now=moment,
    )
    conn = _connect(db_path)
    try:
        ensure_schema(conn)
        conn.execute(
            """
            INSERT OR IGNORE INTO policy_episodes (
                candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask,
                roundtrip_cost, snapshot_json, status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(candidate_id),
                COUNTERFACTUAL,
                str(engine),
                str(symbol),
                str(setup or ""),
                str(regime or ""),
                moment,
                ask,
                float(roundtrip_cost or 0.0),
                json.dumps(snap, separators=(",", ":"), default=str),
                OPEN,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT id FROM policy_episodes WHERE candidate_id=?", (int(candidate_id),)).fetchone()
        return None if row is None else int(row[0])
    finally:
        conn.close()


def decision_from_check(unrealized: float, advantage: float | None) -> str:
    """The stored check is the decision. It does not read a later model."""
    if advantage is None:
        return "hold"
    terminal = float(unrealized) + float(advantage)
    return learned_hold_or_exit(expected_terminal_net=terminal, unrealized_net=float(unrealized))


def _learn(conn: sqlite3.Connection, row: sqlite3.Row, net: float, moment: float) -> None:
    folded = _fold_observation(
        conn,
        engine=str(row["engine_id"]),
        symbol=str(row["symbol"]),
        setup=str(row["setup"]),
        regime=str(row["regime"]),
        metric="direct_policy_net",
        value=float(net),
        strategy_version=current_strategy_version(str(row["engine_id"])),
        now=moment,
    )
    if not folded:
        raise RuntimeError("policy episode observation rejected")
    conn.execute("UPDATE policy_episodes SET learned=1 WHERE id=? AND learned=0", (int(row["id"]),))


def _close(conn: sqlite3.Connection, row: sqlite3.Row, *, kind: str, net: float, moment: float, bid: float | None, reason: str, gross: float | None, hold: float | None) -> None:
    conn.execute("SAVEPOINT mystic_policy_episode")
    try:
        cur = conn.execute(
            """
            UPDATE policy_episodes
            SET status=?, kind=?, exit_bid=?, exit_at=?, exit_reason=?, gross=?, net=?, hold_sec=?
            WHERE id=? AND status=? AND learned=0
            """,
            (CLOSED, kind, bid, moment, reason, gross, float(net), hold, int(row["id"]), OPEN),
        )
        if cur.rowcount != 1:
            conn.execute("ROLLBACK TO mystic_policy_episode")
            conn.execute("RELEASE mystic_policy_episode")
            return
        _learn(conn, row, float(net), moment)
        conn.execute("RELEASE mystic_policy_episode")
    except Exception:
        conn.execute("ROLLBACK TO mystic_policy_episode")
        conn.execute("RELEASE mystic_policy_episode")
        raise


def advance_episodes(
    db_path: str,
    bid_at: Callable[[str], float | None],
    *,
    now: float | None = None,
    limit: int = ADVANCE_LIMIT,
) -> dict[str, int]:
    """Walk open episodes one check. Missing bids are skipped, not invented.

    A filled candidate with an accounting realized net closes as a real outcome
    and is not also simulated. The research mean updates in that same transaction.
    """
    moment = float(now if now is not None else time.time())
    stats = {"checked": 0, "exited": 0, "real": 0, "skipped": 0}
    conn = _connect(db_path)
    try:
        ensure_schema(conn)
        rows = conn.execute("SELECT * FROM policy_episodes WHERE status=? ORDER BY decided_at, id LIMIT ?", (OPEN, int(limit))).fetchall()
        for row in rows:
            mark = conn.execute("SELECT filled, realized_net FROM adaptive_candidate_markouts WHERE id=?", (int(row["candidate_id"]),)).fetchone()
            if mark is not None and int(mark["filled"] or 0) and mark["realized_net"] is not None:
                _close(conn, row, kind=REAL, net=float(mark["realized_net"]), moment=moment, bid=None, reason="REAL_FILL", gross=None, hold=moment - float(row["decided_at"]))
                stats["real"] += 1
                conn.commit()
                continue
            if mark is not None and int(mark["filled"] or 0):
                stats["skipped"] += 1
                continue
            previous = conn.execute("SELECT checked_at, unrealized FROM policy_episode_checks WHERE episode_id=? ORDER BY checked_at DESC LIMIT 1", (int(row["id"]),)).fetchone()
            if previous is not None and moment - float(previous["checked_at"]) < MIN_CHECK_GAP_SEC:
                continue
            bid = bid_at(str(row["symbol"]))
            if bid is None or float(bid) <= 0.0:
                stats["skipped"] += 1
                continue
            ask = float(row["entry_ask"])
            cost = float(row["roundtrip_cost"] or 0.0)
            unrealized = executable_bid_net(ask, float(bid), cost)
            gross = (float(bid) - ask) / ask
            favorable = max(float(row["mfe"] or 0.0), gross, 0.0)
            adverse = max(float(row["mae"] or 0.0), -min(gross, 0.0), 0.0)
            prev_net = None if previous is None else float(previous["unrealized"])
            features = state_features(
                entry=ask,
                mark=float(bid),
                net=unrealized,
                mfe=favorable,
                mae=adverse,
                high_water=ask * (1.0 + favorable),
                prev_net=prev_net,
                age_sec=moment - float(row["decided_at"]),
            )
            terminal = continuation_terminal(db_path, str(row["engine_id"]), str(row["symbol"]), str(row["setup"]), str(row["regime"]), unrealized, now=moment, features=features)
            advantage = None if terminal is None else float(terminal) - unrealized
            action = learned_hold_or_exit(expected_terminal_net=terminal, unrealized_net=unrealized)
            conn.execute(
                "INSERT OR IGNORE INTO policy_episode_checks (episode_id, checked_at, unrealized, advantage, terminal, action) VALUES (?,?,?,?,?,?)",
                (int(row["id"]), moment, unrealized, advantage, terminal, action),
            )
            conn.execute("UPDATE policy_episodes SET mfe=?, mae=? WHERE id=?", (favorable, adverse, int(row["id"])))
            stats["checked"] += 1
            if action == "exit":
                _close(
                    conn,
                    row,
                    kind=COUNTERFACTUAL,
                    net=unrealized,
                    moment=moment,
                    bid=float(bid),
                    reason=EXIT_REASON,
                    gross=gross,
                    hold=moment - float(row["decided_at"]),
                )
                stats["exited"] += 1
            # Release the write lock before the next continuation read. That read
            # opens the database itself, and an open write here would wait on it.
            conn.commit()
        conn.commit()
    finally:
        conn.close()
    return stats


def live_executable_bid(symbol: str) -> float | None:
    """Current bid when the book is fresh. Stale or missing books stay missing."""
    try:
        from backend.config.day_entry_execution import BOOK_STALE_SEC
        from backend.config.redis_config import get_redis_client
        from backend.services.spread_book_telemetry import read_market_book

        book = read_market_book(get_redis_client(), symbol)
    except Exception:
        return None
    if not book:
        return None
    try:
        age = float(book.get("freshness_sec") or 0.0)
        bid = float(book.get("bid") or 0.0)
    except (TypeError, ValueError):
        return None
    if age > float(BOOK_STALE_SEC) or bid <= 0.0:
        return None
    return bid


def advance_live(db_path: str, *, now: float | None = None) -> dict[str, int]:
    """One live research pass. Failures are logged by the caller."""
    return advance_episodes(db_path, live_executable_bid, now=now)


def open_recorded_day(db_path: str, cand: dict[str, Any], candidate_id: int | None) -> int | None:
    if not candidate_id:
        return None
    adaptive = cand.get("adaptive") or {}
    cost = float((adaptive.get("economic") or {}).get("expected_cost") or 0.0)
    if cost <= 0.0:
        from backend.config.trading_economics import canonical_roundtrip_cost_pct

        cost = float(canonical_roundtrip_cost_pct())
    return open_episode(
        db_path,
        candidate_id=int(candidate_id),
        engine=DAY_ENGINE,
        symbol=str(cand.get("symbol") or ""),
        setup=str(cand.get("learned_setup") or getattr(cand.get("signal"), "setup", "") or ""),
        regime=str(cand.get("regime_tag") or ""),
        entry_ask=float(cand.get("ask_price") or 0.0),
        roundtrip_cost=cost,
        economics=adaptive.get("economic") if isinstance(adaptive.get("economic"), dict) else None,
        rank_position=(cand.get("rank") or {}).get("position"),
        size_mult=adaptive.get("size_mult"),
        now=float(cand.get("as_of") or time.time()),
    )


def symbol_outcomes(db_path: str) -> list[dict[str, Any]]:
    """Resolved policy net per symbol at each decision second. Not a fixed-horizon mark."""
    conn = _connect(db_path)
    try:
        ensure_schema(conn)
        rows = conn.execute("SELECT decided_at, symbol, net, kind FROM policy_episodes WHERE status=? AND net IS NOT NULL", (CLOSED,)).fetchall()
    finally:
        conn.close()
    grouped: dict[tuple[int, str], float] = {}
    for row in rows:
        key = (int(float(row["decided_at"])), str(row["symbol"]))
        net = float(row["net"])
        grouped[key] = net if key not in grouped else max(grouped[key], net)
    return [{"t": t, "symbol": symbol, "net": net} for (t, symbol), net in sorted(grouped.items())]


def open_recorded_scalp(
    db_path: str,
    *,
    candidate_id: int | None,
    symbol: str,
    setup: str,
    regime: str,
    entry_ask: float,
    roundtrip_cost: float,
    economics: dict[str, Any] | None,
    now: float | None = None,
) -> int | None:
    if not candidate_id:
        return None
    return open_episode(
        db_path,
        candidate_id=int(candidate_id),
        engine=SCALP_ENGINE,
        symbol=symbol,
        setup=setup,
        regime=regime,
        entry_ask=float(entry_ask or 0.0),
        roundtrip_cost=float(roundtrip_cost or 0.0),
        economics=economics,
        now=now,
    )
