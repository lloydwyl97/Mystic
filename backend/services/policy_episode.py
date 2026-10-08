"""Research-only policy episodes. No orders, cash, positions, or live entry authority.

Every recorded candidate can open a virtual position at the executable ask from
that decision. Later checks call the same continuation function a live position
uses and store the advantage, terminal, and action. A replay reads those rows.
It does not recompute them against a later model.

A candidate that actually fills is closed from its accounting realized net and
tagged REAL_POLICY_OUTCOME. Every other resolution is COUNTERFACTUAL_POLICY_OUTCOME.
Neither tag is account P&L. ``direct_policy_net`` is updated from the resolved
net and is not an input to entry ranking.

The entry snapshot, including the direct-policy prediction, is written when the
episode opens. The learner is updated only after that episode's net is stored.
A replay reads the stored continuation checks. Those checks were made on the
learner state a live position would have seen at that moment, not on a frozen
entry-time copy and not on a later retrained model.
"""

from __future__ import annotations

import json
import logging
import math
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
    estimate,
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
UNIVERSE = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
PREDICTION_SOURCE = "hierarchical_direct_policy_net"

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
    learned INTEGER NOT NULL DEFAULT 0,
    decision_group_id TEXT,
    prediction_at REAL,
    funded INTEGER,
    reject_reason TEXT,
    live_error REAL,
    direct_error REAL,
    recovered INTEGER NOT NULL DEFAULT 0
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
SCORE_DDL = """
CREATE TABLE IF NOT EXISTS policy_group_scores (
    decision_group_id TEXT PRIMARY KEY,
    engine_id TEXT NOT NULL,
    decided_at REAL NOT NULL,
    n_symbols INTEGER NOT NULL,
    best_net REAL,
    mean_net REAL,
    any_positive INTEGER NOT NULL,
    live_selected_net REAL,
    direct_selected_net REAL,
    oracle_net REAL,
    live_regret REAL,
    direct_regret REAL,
    live_spearman REAL,
    direct_spearman REAL,
    live_capture REAL,
    direct_capture REAL,
    predictions_complete INTEGER NOT NULL,
    detail_json TEXT NOT NULL,
    scored_at REAL NOT NULL
)
"""
SUMMARY_DDL = """
CREATE TABLE IF NOT EXISTS policy_research_summary (
    engine_id TEXT PRIMARY KEY,
    summary_json TEXT NOT NULL,
    updated_at REAL NOT NULL
)
"""
PARITY_DDL = """
CREATE TABLE IF NOT EXISTS policy_fill_parity (
    episode_id INTEGER PRIMARY KEY,
    exit_at REAL,
    accounting_net REAL,
    last_check_at REAL,
    last_check_unrealized REAL,
    last_check_advantage REAL,
    last_check_action TEXT,
    net_delta REAL
)
"""
_EPISODE_COLUMNS = (
    ("decision_group_id", "TEXT"),
    ("prediction_at", "REAL"),
    ("funded", "INTEGER"),
    ("reject_reason", "TEXT"),
    ("live_error", "REAL"),
    ("direct_error", "REAL"),
    ("recovered", "INTEGER NOT NULL DEFAULT 0"),
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(EPISODE_DDL)
    conn.execute(CHECK_DDL)
    conn.execute(SCORE_DDL)
    conn.execute(SUMMARY_DDL)
    conn.execute(PARITY_DDL)
    present = {str(row[1]) for row in conn.execute("PRAGMA table_info(policy_episodes)")}
    for name, decl in _EPISODE_COLUMNS:
        if name not in present:
            conn.execute(f"ALTER TABLE policy_episodes ADD COLUMN {name} {decl}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_episodes_open ON policy_episodes(status, decided_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_episodes_group ON policy_episodes(decision_group_id, symbol)")
    from backend.services.policy_state_learner import ensure_schema as ensure_state_schema

    ensure_state_schema(conn)
    conn.execute(
        """
        UPDATE policy_episodes
        SET decision_group_id = engine_id || ':' || printf('%.6f', decided_at)
        WHERE decision_group_id IS NULL OR decision_group_id = ''
        """
    )


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def canon_symbol(symbol: str) -> str:
    text = str(symbol or "").upper().replace("-", "").replace("/", "")
    if text in {"BTC", "ETH", "SOL", "XRP"}:
        return text + "USDT"
    return text


def economics_of(edge: Any) -> dict[str, Any] | None:
    """The decision-time economic dict, whether the edge is a dict or an object."""
    if isinstance(edge, dict):
        inner = edge.get("economic")
        if isinstance(inner, dict):
            return inner
        final = _num(edge.get("final_executable_edge_pct"))
        if final is None:
            final = _num(edge.get("policy_value"))
        if final is None and edge.get("policy_gap") is None and edge.get("policy_gap_pct") is None:
            return None
        calibration = edge.get("policy_calibration_pct", edge.get("policy_calibration"))
        uncalibrated = edge.get("uncalibrated_policy_value")
        if uncalibrated is None and final is not None and _num(calibration) is not None:
            uncalibrated = float(final) - float(calibration)
        market = edge.get("market_edge_pct", edge.get("market_edge", edge.get("market_alpha")))
        return {
            "market_alpha": market,
            "market_edge": market,
            "policy_gap": edge.get("policy_gap_pct", edge.get("policy_gap")),
            "policy_calibration": calibration,
            "uncalibrated_policy_value": uncalibrated,
            "policy_value": final,
            "eligible": edge.get("eligible"),
            "size_mult": edge.get("size_mult"),
            "expected_cost": edge.get("live_cost_pct", edge.get("expected_cost")),
        }
    inner = getattr(edge, "economic", None)
    if callable(inner):
        try:
            inner = inner()
        except Exception:
            return None
    return inner if isinstance(inner, dict) else None


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


def _issued(economics: dict[str, Any] | None, *, rank_position: Any, size_mult: Any, funded: bool, reject_reason: str) -> dict[str, Any]:
    econ = economics if isinstance(economics, dict) else {}
    market = econ.get("market_alpha")
    market_source = "market_alpha"
    if _num(market) is None and _num(econ.get("market_edge")) is not None:
        market = econ.get("market_edge")
        market_source = "market_edge"
    return {
        "market_alpha": market,
        "market_alpha_source": market_source,
        "market_edge": econ.get("market_edge"),
        "policy_gap": econ.get("policy_gap"),
        "uncalibrated_policy_value": econ.get("uncalibrated_policy_value"),
        "policy_calibration": econ.get("policy_calibration"),
        "policy_value": econ.get("policy_value"),
        "rank_position": rank_position,
        "size_mult": size_mult if size_mult is not None else econ.get("size_mult"),
        "funded": bool(funded),
        "reject_reason": str(reject_reason or ""),
        "eligible": econ.get("eligible"),
    }


def _direct_prediction(db_path: str, engine: str, symbol: str, setup: str, regime: str, now: float) -> dict[str, Any]:
    """Hierarchical mean of policy nets already resolved. This episode is not one of them."""
    view = estimate(db_path, engine, symbol, setup, regime, "direct_policy_net", now=now, include_engine=False)
    return {
        "mean": view["mean"],
        "prior": view["prior"],
        "n": view["n"],
        "levels": view["levels"],
        "level_weights": view["level_weights"],
        "source": PREDICTION_SOURCE,
        "include_engine": False,
        "prediction_at": now,
    }


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
    candidate_id: int | None = None,
    decision_group_id: str = "",
    funded: bool = False,
    reject_reason: str = "",
    entry_ask: float | None = None,
    roundtrip_cost: float | None = None,
) -> dict[str, Any]:
    """Immutable description of the decision, written before this episode resolves.

    Horizon means are the posteriors a decision at ``now`` would read. Later
    checks store their own advantage instead of rereading these means. The
    direct-policy block is the learner's estimate before this net exists.
    """
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
        "engine": str(engine),
        "candidate_id": candidate_id,
        "decision_group_id": decision_group_id,
        "symbol": canon_symbol(symbol),
        "decided_at": now,
        "economic_version": current_economic_version(engine),
        "strategy_version": current_strategy_version(engine),
        "continuation_version": version,
        "continuation_source": source,
        "continuation_semantics": "contemporaneous",
        "advantage_authority": version == ADVANTAGE_VERSION,
        "aggregator": installed_aggregator(db_path, engine),
        "horizons": horizons,
        "entry_ask": entry_ask,
        "expected_cost": roundtrip_cost,
        "features": {"source": "adaptive_candidate_markouts.features_json", "candidate_id": candidate_id},
        "issued": _issued(economics, rank_position=rank_position, size_mult=size_mult, funded=funded, reject_reason=reject_reason),
        "direct_policy": _direct_prediction(db_path, engine, symbol, setup, regime, now),
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
    decision_group_id: str | None = None,
    funded: bool = False,
    reject_reason: str = "",
    features: dict[str, Any] | None = None,
    peers: list[float] | None = None,
) -> int | None:
    """Open one research episode. Invalid asks are skipped. A second call is a no-op."""
    ask = float(entry_ask or 0.0)
    if candidate_id is None or ask <= 0.0 or not (ask < float("inf")):
        return None
    moment = float(now if now is not None else time.time())
    group = str(decision_group_id or f"{engine}:{moment:.6f}")
    cost = float(roundtrip_cost or 0.0)
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
        candidate_id=int(candidate_id),
        decision_group_id=group,
        funded=bool(funded),
        reject_reason=str(reject_reason or ""),
        entry_ask=ask,
        roundtrip_cost=cost,
    )
    try:
        from backend.services.policy_state_learner import freeze_prediction

        issued = dict(snap.get("issued") or {})
        issued["expected_cost"] = cost
        frozen = freeze_prediction(db_path, engine=engine, features=features, issued=issued, symbol=symbol, peers=peers)
        snap["state_features"] = frozen["features"]
        snap["state_predictions"] = frozen["predictions"]
    except Exception:
        logger.debug("POLICY_STATE_PREDICT_FAILED", exc_info=True)
    prediction_at = _num((snap.get("direct_policy") or {}).get("prediction_at"))
    conn = _connect(db_path)
    try:
        ensure_schema(conn)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO policy_episodes (
                candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask,
                roundtrip_cost, snapshot_json, status, decision_group_id, prediction_at,
                funded, reject_reason, recovered
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
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
                cost,
                json.dumps(snap, separators=(",", ":"), default=str),
                OPEN,
                group,
                prediction_at,
                1 if funded else 0,
                str(reject_reason or ""),
            ),
        )
        if cur.rowcount == 1:
            episode_id = int(cur.lastrowid)
            snap["episode_id"] = episode_id
            conn.execute(
                "UPDATE policy_episodes SET snapshot_json=? WHERE id=? AND learned=0 AND exit_at IS NULL",
                (json.dumps(snap, separators=(",", ":"), default=str), episode_id),
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


def _forecast_errors(row: sqlite3.Row, net: float) -> tuple[float | None, float | None]:
    try:
        snap = json.loads(row["snapshot_json"] or "{}")
    except json.JSONDecodeError:
        snap = {}
    issued = snap.get("issued") if isinstance(snap, dict) else None
    direct = snap.get("direct_policy") if isinstance(snap, dict) else None
    live_v = _num((issued or {}).get("policy_value")) if isinstance(issued, dict) else None
    direct_v = _num((direct or {}).get("mean")) if isinstance(direct, dict) else None
    return (None if live_v is None else float(net) - live_v, None if direct_v is None else float(net) - direct_v)


def _close(conn: sqlite3.Connection, row: sqlite3.Row, *, kind: str, net: float, moment: float, bid: float | None, reason: str, gross: float | None, hold: float | None) -> None:
    live_error, direct_error = _forecast_errors(row, float(net))
    conn.execute("SAVEPOINT mystic_policy_episode")
    try:
        cur = conn.execute(
            """
            UPDATE policy_episodes
            SET status=?, kind=?, exit_bid=?, exit_at=?, exit_reason=?, gross=?, net=?, hold_sec=?,
                live_error=?, direct_error=?
            WHERE id=? AND status=? AND learned=0
            """,
            (CLOSED, kind, bid, moment, reason, gross, float(net), hold, live_error, direct_error, int(row["id"]), OPEN),
        )
        if cur.rowcount != 1:
            conn.execute("ROLLBACK TO mystic_policy_episode")
            conn.execute("RELEASE mystic_policy_episode")
            return
        _learn(conn, row, float(net), moment)
        if kind == REAL:
            _record_parity(conn, row, float(net), moment)
        conn.execute("RELEASE mystic_policy_episode")
    except Exception:
        conn.execute("ROLLBACK TO mystic_policy_episode")
        conn.execute("RELEASE mystic_policy_episode")
        raise


def _record_parity(conn: sqlite3.Connection, row: sqlite3.Row, accounting_net: float, moment: float) -> None:
    last = conn.execute(
        "SELECT checked_at, unrealized, advantage, action FROM policy_episode_checks WHERE episode_id=? ORDER BY checked_at DESC LIMIT 1",
        (int(row["id"]),),
    ).fetchone()
    last_net = None if last is None else float(last["unrealized"])
    conn.execute(
        """
        INSERT OR REPLACE INTO policy_fill_parity (
            episode_id, exit_at, accounting_net, last_check_at, last_check_unrealized,
            last_check_advantage, last_check_action, net_delta
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            int(row["id"]),
            moment,
            accounting_net,
            None if last is None else float(last["checked_at"]),
            last_net,
            None if last is None else last["advantage"],
            None if last is None else str(last["action"]),
            None if last_net is None else accounting_net - last_net,
        ),
    )


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    xs, ys = _ranks(left), _ranks(right)
    n = float(len(xs))
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    dx = sum((a - mx) ** 2 for a in xs)
    dy = sum((b - my) ** 2 for b in ys)
    if dx <= 0.0 or dy <= 0.0:
        return None
    return num / ((dx ** 0.5) * (dy ** 0.5))


def _snap(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _selected(rows: list[dict[str, Any]], key: str) -> float | None:
    usable = [row for row in rows if row[key] is not None]
    if len(usable) != len(rows) or not rows:
        return None
    best = max(usable, key=lambda row: float(row[key]))
    if float(best[key]) <= 0.0:
        return 0.0
    return float(best["net"])


def _score_group(conn: sqlite3.Connection, group_id: str, moment: float) -> None:
    rows = conn.execute(
        "SELECT id, engine_id, symbol, decided_at, net, snapshot_json, status FROM policy_episodes WHERE decision_group_id=?",
        (group_id,),
    ).fetchall()
    if not rows or any(str(row["status"]) != CLOSED or row["net"] is None for row in rows):
        return
    by_symbol: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_symbol.setdefault(canon_symbol(str(row["symbol"])), []).append(row)
    if any(symbol not in by_symbol for symbol in UNIVERSE):
        return
    primaries = []
    for symbol in UNIVERSE:
        choices = by_symbol[symbol]

        def _live_value(row: sqlite3.Row) -> float | None:
            issued = _snap(str(row["snapshot_json"])).get("issued")
            return _num(issued.get("policy_value")) if isinstance(issued, dict) else None

        chosen = max(choices, key=lambda row: ((_live_value(row) is not None), _live_value(row) or -1e99, -int(row["id"])))
        snap = _snap(str(chosen["snapshot_json"]))
        issued = snap.get("issued") if isinstance(snap.get("issued"), dict) else {}
        direct = snap.get("direct_policy") if isinstance(snap.get("direct_policy"), dict) else {}
        primaries.append(
            {
                "symbol": symbol,
                "engine": str(chosen["engine_id"]),
                "decided_at": float(chosen["decided_at"]),
                "episode_id": int(chosen["id"]),
                "net": float(chosen["net"]),
                "live": _num(issued.get("policy_value")),
                "direct": _num(direct.get("mean")),
                "state": snap.get("state_predictions") if isinstance(snap.get("state_predictions"), dict) else {},
            }
        )
    nets = [row["net"] for row in primaries]
    best = max(float(row["net"]) for row in rows)
    mean_net = sum(nets) / len(nets)
    live_selected = _selected(primaries, "live")
    direct_selected = _selected(primaries, "direct")
    complete = all(row["live"] is not None and row["direct"] is not None for row in primaries)
    live_spearman = _spearman([float(row["live"]) for row in primaries], nets) if complete else None
    direct_spearman = _spearman([float(row["direct"]) for row in primaries], nets) if complete else None
    detail = {
        "symbols": primaries,
        "best_any_net": best,
        "live_order": [row["symbol"] for row in sorted(primaries, key=lambda row: (row["live"] is not None, row["live"] or -1e99), reverse=True)],
        "direct_order": [row["symbol"] for row in sorted(primaries, key=lambda row: (row["direct"] is not None, row["direct"] or -1e99), reverse=True)],
        "actual_order": [row["symbol"] for row in sorted(primaries, key=lambda row: row["net"], reverse=True)],
    }
    conn.execute(
        """
        INSERT OR IGNORE INTO policy_group_scores (
            decision_group_id, engine_id, decided_at, n_symbols, best_net, mean_net, any_positive,
            live_selected_net, direct_selected_net, oracle_net, live_regret, direct_regret,
            live_spearman, direct_spearman, live_capture, direct_capture, predictions_complete,
            detail_json, scored_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            group_id,
            str(rows[0]["engine_id"]),
            float(rows[0]["decided_at"]),
            len(primaries),
            best,
            mean_net,
            1 if best > 0.0 else 0,
            live_selected,
            direct_selected,
            best,
            None if live_selected is None else best - live_selected,
            None if direct_selected is None else best - direct_selected,
            live_spearman,
            direct_spearman,
            None if live_selected is None or best <= 0.0 else live_selected / best,
            None if direct_selected is None or best <= 0.0 else direct_selected / best,
            1 if complete else 0,
            json.dumps(detail, separators=(",", ":")),
            moment,
        ),
    )
    try:
        from backend.services.policy_state_learner import score_group

        score_group(conn, group_id, primaries, moment)
    except Exception:
        logger.debug("POLICY_CHALLENGER_SCORE_FAILED", exc_info=True)


def _refresh_summary(conn: sqlite3.Connection, engine: str, moment: float) -> None:
    rows = conn.execute(
        """
        SELECT best_net, mean_net, any_positive, live_selected_net, direct_selected_net, oracle_net,
               live_regret, direct_regret, live_spearman, direct_spearman, predictions_complete
        FROM policy_group_scores WHERE engine_id=? ORDER BY decided_at
        """,
        (engine,),
    ).fetchall()
    n = len(rows)
    positive = sum(int(row["any_positive"] or 0) for row in rows)
    bests = [float(row["best_net"]) for row in rows if row["best_net"] is not None]
    complete = [row for row in rows if int(row["predictions_complete"] or 0)]

    def _avg(values: list[float]) -> float | None:
        return None if not values else sum(values) / len(values)

    def _median(values: list[float]) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    def _pf(values: list[float]) -> float | None:
        gains = sum(v for v in values if v > 0.0)
        losses = -sum(v for v in values if v < 0.0)
        if losses <= 0.0:
            return None
        return gains / losses

    def _drawdown(values: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        worst = 0.0
        for value in values:
            equity += value
            peak = max(peak, equity)
            worst = max(worst, peak - equity)
        return worst

    live_selected = [float(row["live_selected_net"]) for row in complete if row["live_selected_net"] is not None]
    direct_selected = [float(row["direct_selected_net"]) for row in complete if row["direct_selected_net"] is not None]
    summary = {
        "complete_groups": n,
        "prediction_groups": len(complete),
        "positive_groups": positive,
        "positive_rate": None if n == 0 else positive / n,
        "mean_best_net": _avg(bests),
        "median_best_net": _median(bests),
        "live_selected_net": _avg(live_selected),
        "direct_selected_net": _avg(direct_selected),
        "oracle_net": _avg([float(row["oracle_net"]) for row in rows if row["oracle_net"] is not None]),
        "live_regret": _avg([float(row["live_regret"]) for row in complete if row["live_regret"] is not None]),
        "direct_regret": _avg([float(row["direct_regret"]) for row in complete if row["direct_regret"] is not None]),
        "live_spearman": _avg([float(row["live_spearman"]) for row in complete if row["live_spearman"] is not None]),
        "direct_spearman": _avg([float(row["direct_spearman"]) for row in complete if row["direct_spearman"] is not None]),
        "live_pf": _pf(live_selected),
        "direct_pf": _pf(direct_selected),
        "live_drawdown": _drawdown(live_selected),
        "direct_drawdown": _drawdown(direct_selected),
    }
    conn.execute(
        """
        INSERT INTO policy_research_summary (engine_id, summary_json, updated_at) VALUES (?,?,?)
        ON CONFLICT(engine_id) DO UPDATE SET summary_json=excluded.summary_json, updated_at=excluded.updated_at
        """,
        (engine, json.dumps(summary, separators=(",", ":")), moment),
    )


def _score_ready_groups(conn: sqlite3.Connection, moment: float) -> None:
    pending = conn.execute(
        """
        SELECT decision_group_id, MIN(engine_id) AS engine_id FROM policy_episodes
        WHERE decision_group_id IS NOT NULL AND decision_group_id != ''
          AND decision_group_id NOT IN (SELECT decision_group_id FROM policy_group_scores)
        GROUP BY decision_group_id
        HAVING SUM(CASE WHEN status != ? THEN 1 ELSE 0 END) = 0
           AND COUNT(DISTINCT CASE
                WHEN REPLACE(REPLACE(UPPER(symbol), '-', ''), '/', '') IN ('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT')
                THEN REPLACE(REPLACE(UPPER(symbol), '-', ''), '/', '')
           END) = 4
        LIMIT 20
        """,
        (CLOSED,),
    ).fetchall()
    engines = set()
    for row in pending:
        before = conn.execute("SELECT 1 FROM policy_group_scores WHERE decision_group_id=?", (row["decision_group_id"],)).fetchone()
        _score_group(conn, str(row["decision_group_id"]), moment)
        after = conn.execute("SELECT 1 FROM policy_group_scores WHERE decision_group_id=?", (row["decision_group_id"],)).fetchone()
        if before is None and after is not None:
            engines.add(str(row["engine_id"]))
    for engine in engines:
        _refresh_summary(conn, engine, moment)


def _decision_reason(conn: sqlite3.Connection, symbol: str, decided_at: float) -> str | None:
    try:
        found = conn.execute(
            "SELECT symbol, reason FROM scalp_v2_decisions WHERE ABS(cycle_ts - ?) < 0.001",
            (float(decided_at),),
        ).fetchall()
    except sqlite3.Error:
        return None
    matched = [str(row[1]) for row in found if canon_symbol(str(row[0])) == canon_symbol(symbol)]
    if len(matched) != 1 or not matched[0]:
        return None
    return matched[0]


def _recover_issued(conn: sqlite3.Connection) -> None:
    """Fill null entry fields from state written at the decision. Never writes a prediction."""
    rows = conn.execute(
        "SELECT id, candidate_id, symbol, decided_at, snapshot_json FROM policy_episodes WHERE recovered=0 LIMIT 40"
    ).fetchall()
    for row in rows:
        snap = _snap(str(row["snapshot_json"]))
        issued = snap.get("issued") if isinstance(snap.get("issued"), dict) else {}
        snap["issued"] = issued
        try:
            mark = conn.execute("SELECT economic_json FROM adaptive_candidate_markouts WHERE id=?", (int(row["candidate_id"]),)).fetchone()
            econ = json.loads(mark[0] or "{}") if mark is not None else {}
        except (sqlite3.Error, json.JSONDecodeError, TypeError, IndexError):
            econ = {}
        if isinstance(econ, dict):
            for key in ("policy_gap", "uncalibrated_policy_value", "policy_calibration", "policy_value"):
                if issued.get(key) is None and econ.get(key) is not None:
                    issued[key] = econ[key]
            if issued.get("market_alpha") is None:
                if econ.get("market_alpha") is not None:
                    issued["market_alpha"] = econ.get("market_alpha")
                    issued["market_alpha_source"] = "market_alpha"
                elif econ.get("market_edge") is not None:
                    issued["market_alpha"] = econ.get("market_edge")
                    issued["market_alpha_source"] = "market_edge"
        if "funded" not in issued:
            value = _num(issued.get("policy_value"))
            if value is not None:
                issued["funded"] = value > 0.0
                if not issued["funded"] and not issued.get("reject_reason"):
                    issued["reject_reason"] = "NO_EXECUTABLE_NET_EDGE"
        if not issued.get("reject_reason") and issued.get("policy_value") is None:
            reason = _decision_reason(conn, str(row["symbol"]), float(row["decided_at"]))
            if reason:
                issued["reject_reason"] = reason
                issued["funded"] = False
        snap["issued_recovered"] = True
        conn.execute(
            "UPDATE policy_episodes SET snapshot_json=?, recovered=1, funded=?, reject_reason=? WHERE id=? AND recovered=0",
            (
                json.dumps(snap, separators=(",", ":"), default=str),
                1 if issued.get("funded") else 0,
                str(issued.get("reject_reason") or ""),
                int(row["id"]),
            ),
        )


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
        _recover_issued(conn)
        rows = conn.execute("SELECT * FROM policy_episodes WHERE status=? ORDER BY decided_at, id LIMIT ?", (OPEN, int(limit))).fetchall()
        for row in rows:
            mark = conn.execute("SELECT filled, realized_net FROM adaptive_candidate_markouts WHERE id=?", (int(row["candidate_id"]),)).fetchone()
            filled = mark is not None and int(mark["filled"] or 0)
            if filled and mark is not None and mark["realized_net"] is not None:
                _close(conn, row, kind=REAL, net=float(mark["realized_net"]), moment=moment, bid=None, reason="REAL_FILL", gross=None, hold=moment - float(row["decided_at"]))
                stats["real"] += 1
                conn.commit()
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
            if action == "exit" and not filled:
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
        _score_ready_groups(conn, moment)
        try:
            from backend.services.policy_state_learner import record_due_horizons

            record_due_horizons(conn, bid_at, moment)
        except Exception:
            logger.debug("POLICY_HORIZON_MARK_FAILED", exc_info=True)
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
    econ = adaptive.get("economic") if isinstance(adaptive.get("economic"), dict) else None
    cost = float((econ or {}).get("expected_cost") or 0.0)
    if cost <= 0.0:
        from backend.config.trading_economics import canonical_roundtrip_cost_pct

        cost = float(canonical_roundtrip_cost_pct())
    moment = float(cand.get("as_of") or time.time())
    value = _num((econ or {}).get("policy_value"))
    funded = value is not None and value > 0.0
    peers = []
    for alt in cand.get("alternatives") or []:
        peer = _num(alt.get("expected_net")) if isinstance(alt, dict) else None
        if peer is not None:
            peers.append(peer)
    return open_episode(
        db_path,
        candidate_id=int(candidate_id),
        engine=DAY_ENGINE,
        symbol=str(cand.get("symbol") or ""),
        setup=str(cand.get("learned_setup") or getattr(cand.get("signal"), "setup", "") or ""),
        regime=str(cand.get("regime_tag") or ""),
        entry_ask=float(cand.get("ask_price") or 0.0),
        roundtrip_cost=cost,
        economics=econ,
        rank_position=(cand.get("rank") or {}).get("position"),
        size_mult=adaptive.get("size_mult"),
        now=moment,
        decision_group_id=f"{DAY_ENGINE}:{moment:.6f}",
        funded=funded,
        reject_reason="" if funded else "NO_EXECUTABLE_NET_EDGE",
        features=cand.get("state_features") if isinstance(cand.get("state_features"), dict) else None,
        peers=peers,
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
    rank_position: Any = None,
    size_mult: Any = None,
    funded: bool = False,
    reject_reason: str = "",
    decision_group_id: str | None = None,
    features: dict[str, Any] | None = None,
    peers: list[float] | None = None,
) -> int | None:
    if not candidate_id:
        return None
    moment = float(now if now is not None else time.time())
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
        rank_position=rank_position,
        size_mult=size_mult,
        now=moment,
        decision_group_id=decision_group_id or f"{SCALP_ENGINE}:{moment:.6f}",
        funded=funded,
        reject_reason=reject_reason,
        features=features,
        peers=peers,
    )


def rebuild_direct_policy(source_db: str, dest_db: str) -> int:
    """Fold closed episode nets, in exit order, into ``dest_db``. Does not touch live entry."""
    src = _connect(source_db)
    try:
        ensure_schema(src)
        rows = src.execute(
            """
            SELECT engine_id, symbol, setup, regime, net, exit_at
            FROM policy_episodes
            WHERE status=? AND learned=1 AND net IS NOT NULL
            ORDER BY exit_at, id
            """,
            (CLOSED,),
        ).fetchall()
    finally:
        src.close()
    dest = _connect(dest_db)
    try:
        folded = 0
        for row in rows:
            ok = _fold_observation(
                dest,
                engine=str(row["engine_id"]),
                symbol=str(row["symbol"]),
                setup=str(row["setup"]),
                regime=str(row["regime"]),
                metric="direct_policy_net",
                value=float(row["net"]),
                strategy_version=current_strategy_version(str(row["engine_id"])),
                now=float(row["exit_at"]),
            )
            if not ok:
                raise RuntimeError("policy episode rebuild rejected")
            folded += 1
        dest.commit()
        return folded
    finally:
        dest.close()
