"""Exact forward policy outcomes for every executable DAY candidate.

Refused candidates are not orders. They do not spend cash, slots, or sleeve
capital, and their path does not update the live trade_net posterior. A real
fill stays the authoritative outcome. Historical candidates from before this
capture are not relabeled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, TAKER_FEE
from backend.services.adaptive_learning import continuation_learner, executable_bid_net
from backend.services.day_v2.day_policy_decision import decide_day_continuation
from backend.services.policy_episode import (
    CHECK_DDL,
    COUNTERFACTUAL,
    EPISODE_DDL,
    OPEN,
    PARITY_DDL,
    REAL,
    UNIVERSE,
    _connect,
    canon_symbol,
)

logger = logging.getLogger(__name__)

CAPTURE = "EXACT_POLICY_V1"
# Live exit monitor is on the order of half a minute. A gap past this is downtime,
# not a missed tick. The missing checks are not invented.
DOWNTIME_GAP_SEC = 180.0
ADVANCE_LIMIT = 120

_EPISODE_COLUMNS = (
    ("capture_version", "TEXT"),
    ("trade_id", "TEXT"),
    ("label_available_at", "REAL"),
    ("parity_status", "TEXT"),
    ("non_parity_reason", "TEXT"),
    ("entry_bid", "REAL"),
    ("entry_spread", "REAL"),
    ("continuation_version", "TEXT"),
    ("high_water", "REAL"),
    ("prev_net", "REAL"),
    ("last_check_at", "REAL"),
    ("train_eligible", "INTEGER NOT NULL DEFAULT 0"),
    ("train_applied", "INTEGER NOT NULL DEFAULT 0"),
)
_CHECK_COLUMNS = (
    ("bid", "REAL"),
    ("ask", "REAL"),
    ("spread", "REAL"),
    ("mfe", "REAL"),
    ("mae", "REAL"),
    ("giveback", "REAL"),
    ("dist_high", "REAL"),
    ("slope", "REAL"),
    ("age_min", "REAL"),
    ("reason", "TEXT"),
    ("continuation_version", "TEXT"),
    ("state_id", "TEXT"),
    ("trace_json", "TEXT"),
)
_PARITY_COLUMNS = (
    ("policy_match", "INTEGER"),
    ("fill_delta_bps", "REAL"),
    ("exit_reason_real", "TEXT"),
    ("exit_reason_audit", "TEXT"),
    ("trigger_bid", "REAL"),
    ("fill_price", "REAL"),
    ("slippage_note", "TEXT"),
)


def ensure_capture_schema(conn: sqlite3.Connection) -> None:
    conn.execute(EPISODE_DDL)
    conn.execute(CHECK_DDL)
    conn.execute(PARITY_DDL)
    _add_columns(conn, "policy_episodes", _EPISODE_COLUMNS)
    _add_columns(conn, "policy_episode_checks", _CHECK_COLUMNS)
    _add_columns(conn, "policy_fill_parity", _PARITY_COLUMNS)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_episodes_capture ON policy_episodes(capture_version, status, symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_episodes_candidate ON policy_episodes(candidate_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_episodes_trade ON policy_episodes(trade_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_episodes_label ON policy_episodes(label_available_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_policy_checks_episode ON policy_episode_checks(episode_id, checked_at)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exact_policy_learner_state (
            state_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )


def _add_columns(conn: sqlite3.Connection, table: str, columns: tuple[tuple[str, str], ...]) -> None:
    present = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns:
        if name not in present:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def learner_state_id(db_path: str) -> str:
    """Identity of the continuation state read at this moment.

    The first time an identity is seen, the ridge payload is stored once.
    Later checks store only the identity.
    """
    learner = continuation_learner(db_path, "DAY_V2")
    model_at = ""
    payload = ""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT payload, updated_at FROM adaptive_linear_model WHERE engine_id='DAY_V2' AND model LIKE 'continuation_surface%' ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row:
            payload = str(row[0] or "")
            model_at = str(row[1] or "")
    except sqlite3.Error:
        model_at = ""
        payload = ""
    raw = "|".join(
        (
            str(learner.get("economic_version") or ""),
            str(learner.get("learning_version") or ""),
            str(learner.get("authority") or ""),
            str(learner.get("aggregator") or ""),
            str(learner.get("installed_at") or ""),
            model_at,
            payload,
        )
    )
    state_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
    body = {
        "economic_version": learner.get("economic_version"),
        "learning_version": learner.get("learning_version"),
        "authority": learner.get("authority"),
        "aggregator": learner.get("aggregator"),
        "installed_at": learner.get("installed_at"),
        "model_updated_at": model_at,
        "ridge": payload,
    }
    try:
        conn = _connect(db_path)
        try:
            ensure_capture_schema(conn)
            conn.execute(
                "INSERT OR IGNORE INTO exact_policy_learner_state (state_id, created_at, payload_json) VALUES (?,?,?)",
                (state_id, time.time(), json.dumps(body, separators=(",", ":"), default=str)),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        logger.debug("EXACT_POLICY_STATE_FREEZE_FAILED", exc_info=True)
    return state_id


def _book(symbol: str) -> tuple[float | None, float | None]:
    try:
        from backend.config.day_entry_execution import BOOK_STALE_SEC
        from backend.config.redis_config import get_redis_client
        from backend.services.spread_book_telemetry import read_market_book

        book = read_market_book(get_redis_client(), symbol)
    except Exception:
        return None, None
    if not book:
        return None, None
    try:
        age = float(book.get("freshness_sec") or book.get("age_sec") or 0.0)
        bid = float(book.get("bid") or 0.0)
        ask = float(book.get("ask") or 0.0)
    except (TypeError, ValueError):
        return None, None
    if age > float(BOOK_STALE_SEC) or bid <= 0.0:
        return None, None
    return bid, ask if ask > 0 else None


def open_exact_episode(db_path: str, cand: dict[str, Any], candidate_id: int | None) -> int | None:
    """Persist one immutable entry snapshot. No order, no learner update, no model zoo."""
    if not candidate_id:
        return None
    symbol = canon_symbol(str(cand.get("symbol") or ""))
    if symbol not in UNIVERSE:
        return None
    ask = float(cand.get("ask_price") or 0.0)
    if ask <= 0.0:
        return None
    adaptive = cand.get("adaptive") if isinstance(cand.get("adaptive"), dict) else {}
    economic = adaptive.get("economic") if isinstance(adaptive.get("economic"), dict) else {}
    signal = cand.get("signal")
    bid, book_ask = _book(symbol)
    spread = None
    if bid and book_ask and bid > 0:
        spread = (book_ask - bid) / ((book_ask + bid) / 2.0)
    features = cand.get("state_features") if isinstance(cand.get("state_features"), dict) else {}
    rank = cand.get("rank") if isinstance(cand.get("rank"), dict) else {}
    moment = float(cand.get("as_of") or time.time())
    version = ""
    try:
        from backend.services.adaptive_learning import current_economic_version

        version = current_economic_version("DAY_V2")
    except Exception:
        version = ""
    state_id = learner_state_id(db_path)
    snap = {
        "capture": CAPTURE,
        "candidate_id": int(candidate_id),
        "decision_timestamp": moment,
        "symbol": symbol,
        "engine_id": "DAY_V2",
        "policy_contract_version": version,
        "entry_policy_version": version,
        "continuation_version": state_id,
        "learning_state_id": state_id,
        "features": {str(k): features[k] for k in features},
        "setup": str(cand.get("learned_setup") or getattr(signal, "setup", "") or ""),
        "geometry_setup": str(getattr(signal, "setup", "") or ""),
        "regime": str(cand.get("regime_tag") or ""),
        "expected_net": adaptive.get("expected_net"),
        "uncalibrated_policy_value": economic.get("uncalibrated_policy_value"),
        "trade_net_calibration": economic.get("policy_calibration"),
        "rank_position": rank.get("position"),
        "size_mult": adaptive.get("size_mult"),
        "entry_ask": ask,
        "entry_bid": bid,
        "spread": spread,
        "roundtrip_cost": float(economic.get("expected_cost") or ESTIMATED_ROUNDTRIP_COST),
        "taker_fee": float(TAKER_FEE),
        "atr_15m": float(getattr(signal, "atr", 0.0) or 0.0),
        "atr_1h": float(getattr(signal, "atr_1h", 0.0) or 0.0),
        "structural_anchor": float(getattr(signal, "structural_anchor", 0.0) or 0.0),
        "target_price": float(getattr(signal, "target_price", 0.0) or 0.0),
        "objective_structural": float(getattr(signal, "objective_structural", 0.0) or 0.0),
        "objective_atr_mult": adaptive.get("objective_atr_mult"),
        "structural_emphasis": adaptive.get("structural_emphasis"),
        "runner_activation_mult": adaptive.get("runner_activation_mult"),
        "runner_trail_mult": adaptive.get("runner_trail_mult"),
        "runner_tighten_mult": adaptive.get("runner_tighten_mult"),
    }
    conn = _connect(db_path)
    try:
        ensure_capture_schema(conn)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO policy_episodes (
                candidate_id, kind, engine_id, symbol, setup, regime, decided_at, entry_ask,
                roundtrip_cost, snapshot_json, status, funded, reject_reason, recovered,
                capture_version, entry_bid, entry_spread, continuation_version, high_water, parity_status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                int(candidate_id),
                COUNTERFACTUAL,
                "DAY_V2",
                symbol,
                snap["setup"],
                snap["regime"],
                moment,
                ask,
                float(snap["roundtrip_cost"]),
                json.dumps(snap, separators=(",", ":"), default=str),
                OPEN,
                None,
                "",
                0,
                CAPTURE,
                bid,
                spread,
                state_id,
                ask,
                "OPEN",
            ),
        )
        conn.commit()
        if cur.rowcount != 1:
            row = conn.execute("SELECT id FROM policy_episodes WHERE candidate_id=?", (int(candidate_id),)).fetchone()
            return None if row is None else int(row[0])
        return int(cur.lastrowid)
    except sqlite3.Error:
        logger.debug("EXACT_POLICY_OPEN_FAILED", exc_info=True)
        return None
    finally:
        conn.close()


def note_disposition(db_path: str, candidate_id: int | None, *, funded: bool, reason: str, trade_id: str = "") -> None:
    """Record fund or refuse once. Does not rewrite the entry snapshot."""
    if not candidate_id:
        return
    conn = _connect(db_path)
    try:
        ensure_capture_schema(conn)
        conn.execute(
            """
            UPDATE policy_episodes
            SET funded=?, reject_reason=CASE WHEN ?<>'' THEN ? ELSE reject_reason END,
                trade_id=CASE WHEN ?<>'' THEN ? ELSE trade_id END
            WHERE candidate_id=? AND capture_version=? AND status=?
            """,
            (1 if funded else 0, reason, reason, trade_id, trade_id, int(candidate_id), CAPTURE, OPEN),
        )
        conn.commit()
    except sqlite3.Error:
        logger.debug("EXACT_POLICY_DISPOSITION_FAILED", exc_info=True)
    finally:
        conn.close()


def _snap(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def record_live_check(
    db_path: str,
    *,
    candidate_id: int | None,
    trade_id: str,
    symbol: str,
    decision: dict[str, Any],
    bid: float | None,
    ask: float | None,
) -> None:
    """Persist one real-position continuation check. Failure must not block the exit."""
    if not candidate_id and not trade_id:
        return
    conn = _connect(db_path)
    try:
        ensure_capture_schema(conn)
        row = None
        if candidate_id:
            row = conn.execute(
                "SELECT id FROM policy_episodes WHERE candidate_id=? AND capture_version=?",
                (int(candidate_id), CAPTURE),
            ).fetchone()
        if row is None:
            return
        episode_id = int(row[0])
        if bid is None or ask is None:
            book_bid, book_ask = _book(symbol)
            bid = book_bid if bid is None else bid
            ask = book_ask if ask is None else ask
        if not decision.get("state_id"):
            try:
                decision["state_id"] = learner_state_id(db_path)
            except Exception:
                logger.debug("EXACT_POLICY_STATE_ID_FAILED", exc_info=True)
        _insert_check(conn, episode_id, symbol, decision, bid, ask, candidate_id)
        conn.commit()
    except sqlite3.Error:
        logger.debug("EXACT_POLICY_LIVE_CHECK_FAILED", exc_info=True)
    finally:
        conn.close()


def _insert_check(
    conn: sqlite3.Connection,
    episode_id: int | None,
    symbol: str,
    decision: dict[str, Any],
    bid: float | None,
    ask: float | None,
    candidate_id: int | None,
) -> None:
    features = decision.get("features") if isinstance(decision.get("features"), dict) else {}
    spread = None
    if bid and ask and bid > 0 and ask > 0:
        spread = (ask - bid) / ((ask + bid) / 2.0)
    state_id = str(decision.get("state_id") or "")
    conn.execute(
        """
        INSERT OR IGNORE INTO policy_episode_checks (
            episode_id, checked_at, unrealized, advantage, terminal, action,
            bid, ask, spread, mfe, mae, giveback, dist_high, slope, age_min, reason,
            continuation_version, state_id, trace_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(episode_id or 0),
            float(decision.get("at") or time.time()),
            _float(decision.get("net")),
            decision.get("advantage"),
            decision.get("terminal"),
            str(decision.get("action") or "hold"),
            bid,
            ask,
            spread,
            features.get("mfe"),
            features.get("mae"),
            features.get("giveback"),
            features.get("dist_high"),
            features.get("slope"),
            features.get("age_min"),
            str(decision.get("reason") or ""),
            str(decision.get("continuation_version") or state_id),
            state_id,
            json.dumps(decision.get("trace"), separators=(",", ":"), default=str) if decision.get("trace") else None,
        ),
    )


def advance_exact_episodes(db_path: str, *, now: float | None = None, limit: int = ADVANCE_LIMIT) -> dict[str, int]:
    """One live pass over refused executable episodes. Missing bids are skipped."""
    moment = float(now if now is not None else time.time())
    stats = {"checked": 0, "exited": 0, "real": 0, "skipped": 0, "downtime": 0}
    conn = _connect(db_path)
    try:
        ensure_capture_schema(conn)
        funded = conn.execute(
            """
            SELECT * FROM policy_episodes
            WHERE capture_version=? AND status=? AND engine_id='DAY_V2' AND funded=1
            ORDER BY decided_at, id LIMIT ?
            """,
            (CAPTURE, OPEN, int(limit)),
        ).fetchall()
        for row in funded:
            if _close_real_if_resolved(conn, row, moment):
                stats["real"] += 1
        rows = conn.execute(
            """
            SELECT * FROM policy_episodes
            WHERE capture_version=? AND status=? AND engine_id='DAY_V2' AND funded=0
            ORDER BY COALESCE(last_check_at, 0), id LIMIT ?
            """,
            (CAPTURE, OPEN, int(limit)),
        ).fetchall()
        books: dict[str, tuple[float | None, float | None]] = {}
        shared_state = ""
        try:
            shared_state = learner_state_id(db_path)
        except Exception:
            logger.debug("EXACT_POLICY_STATE_ID_FAILED", exc_info=True)
        for row in rows:
            symbol = str(row["symbol"])
            if symbol not in books:
                books[symbol] = _book(symbol)
            bid, ask = books[symbol]
            if bid is None:
                stats["skipped"] += 1
                continue
            last = row["last_check_at"]
            downtime = last is not None and moment - float(last) > DOWNTIME_GAP_SEC
            if downtime:
                conn.execute(
                    "UPDATE policy_episodes SET parity_status=?, non_parity_reason=? WHERE id=? AND status=?",
                    ("NON_PARITY", "DOWNTIME_GAP", int(row["id"]), OPEN),
                )
                stats["downtime"] += 1
            snap = _snap(str(row["snapshot_json"] or ""))
            entry = float(row["entry_ask"])
            high = max(float(row["high_water"] or entry), bid, entry)
            low = entry * (1.0 - float(row["mae"] or 0.0))
            low = min(low, bid)
            prev = None if row["prev_net"] is None else float(row["prev_net"])
            decision = decide_day_continuation(
                db_path,
                symbol=str(row["symbol"]),
                setup=str(row["setup"] or ""),
                exit_setup=str(snap.get("geometry_setup") or row["setup"] or ""),
                regime=str(row["regime"] or ""),
                entry_price=entry,
                mark=bid,
                low=low,
                high=high,
                prev_net=prev,
                age_sec=max(0.0, moment - float(row["decided_at"])),
                atr_at_entry=_float(snap.get("atr_15m")),
                structural_anchor=_float(snap.get("structural_anchor")),
                target_price=_float(snap.get("target_price")),
                entry_time=float(row["decided_at"]),
                roundtrip_cost=float(ESTIMATED_ROUNDTRIP_COST),
                atr_1h_at_entry=_float(snap.get("atr_1h")),
                objective_structural=_float(snap.get("objective_structural")),
                objective_atr_mult=_float(snap.get("objective_atr_mult"), 1.0),
                structural_emphasis=_float(snap.get("structural_emphasis"), 1.0),
                runner_activation_mult=_float(snap.get("runner_activation_mult"), 1.0),
                runner_trail_mult=_float(snap.get("runner_trail_mult"), 1.0),
                runner_tighten_mult=_float(snap.get("runner_tighten_mult"), 1.0),
                now=moment,
            )
            decision["state_id"] = shared_state
            features = decision["features"]
            favorable = max(float(row["mfe"] or 0.0), _float(features.get("mfe")))
            adverse = max(float(row["mae"] or 0.0), _float(features.get("mae")))
            _insert_check(conn, int(row["id"]), str(row["symbol"]), decision, bid, ask, int(row["candidate_id"]))
            conn.execute(
                """
                UPDATE policy_episodes
                SET mfe=?, mae=?, high_water=?, prev_net=?, last_check_at=?
                WHERE id=?
                """,
                (favorable, adverse, high, float(decision["net"]), moment, int(row["id"])),
            )
            stats["checked"] += 1
            if decision["action"] == "exit":
                gross = (bid - entry) / entry
                net = executable_bid_net(entry, bid, float(row["roundtrip_cost"] or ESTIMATED_ROUNDTRIP_COST))
                reason = str(decision["reason"] or "LEARNED_CONTINUATION_EXIT")
                conn.execute(
                    """
                    UPDATE policy_episodes
                    SET status='CLOSED', kind=?, exit_bid=?, exit_at=?, exit_reason=?, gross=?, net=?,
                        hold_sec=?, label_available_at=?, train_eligible=?, learned=1
                    WHERE id=? AND status=?
                    """,
                    (
                        COUNTERFACTUAL,
                        bid,
                        moment,
                        reason,
                        gross,
                        net,
                        moment - float(row["decided_at"]),
                        moment,
                        0 if downtime or str(row["parity_status"] or "") == "NON_PARITY" else 1,
                        int(row["id"]),
                        OPEN,
                    ),
                )
                stats["exited"] += 1
            conn.commit()
    except sqlite3.Error:
        logger.debug("EXACT_POLICY_ADVANCE_FAILED", exc_info=True)
    finally:
        conn.close()
    return stats


def _close_real_if_resolved(conn: sqlite3.Connection, row: sqlite3.Row, moment: float) -> bool:
    mark = conn.execute(
        "SELECT filled, realized_net FROM adaptive_candidate_markouts WHERE id=?",
        (int(row["candidate_id"]),),
    ).fetchone()
    if mark is None or not int(mark["filled"] or 0) or mark["realized_net"] is None:
        return False
    net = float(mark["realized_net"])
    last = conn.execute(
        "SELECT checked_at, action, reason, bid FROM policy_episode_checks WHERE episode_id=? ORDER BY checked_at DESC LIMIT 1",
        (int(row["id"]),),
    ).fetchone()
    conn.execute(
        """
        UPDATE policy_episodes
        SET status='CLOSED', kind=?, net=?, exit_at=?, label_available_at=?, learned=1, train_eligible=0, train_applied=0
        WHERE id=? AND status=?
        """,
        (REAL, net, moment, moment, int(row["id"]), OPEN),
    )
    audit_reason = "" if last is None else str(last["reason"] or "")
    trigger_bid = None if last is None else last["bid"]
    existing = conn.execute(
        "SELECT fill_price, fill_delta_bps, exit_reason_real FROM policy_fill_parity WHERE episode_id=?",
        (int(row["id"]),),
    ).fetchone()
    real_reason = "" if existing is None or existing["exit_reason_real"] is None else str(existing["exit_reason_real"])
    if audit_reason and real_reason:
        policy_match = 1 if audit_reason == real_reason else 0
    elif last is not None and str(last["action"] or "") == "exit":
        policy_match = 1
    else:
        policy_match = None
    conn.execute(
        """
        INSERT INTO policy_fill_parity (
            episode_id, exit_at, accounting_net, last_check_at, last_check_action, net_delta,
            policy_match, exit_reason_audit, exit_reason_real, trigger_bid, fill_price, fill_delta_bps, slippage_note
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(episode_id) DO UPDATE SET
            exit_at=excluded.exit_at,
            accounting_net=excluded.accounting_net,
            last_check_at=excluded.last_check_at,
            last_check_action=excluded.last_check_action,
            policy_match=excluded.policy_match,
            exit_reason_audit=excluded.exit_reason_audit,
            trigger_bid=COALESCE(policy_fill_parity.trigger_bid, excluded.trigger_bid),
            fill_price=COALESCE(policy_fill_parity.fill_price, excluded.fill_price),
            fill_delta_bps=COALESCE(policy_fill_parity.fill_delta_bps, excluded.fill_delta_bps),
            exit_reason_real=COALESCE(policy_fill_parity.exit_reason_real, excluded.exit_reason_real),
            slippage_note=excluded.slippage_note
        """,
        (
            int(row["id"]),
            moment,
            net,
            None if last is None else float(last["checked_at"]),
            None if last is None else str(last["action"] or ""),
            None,
            policy_match,
            audit_reason,
            None if existing is None else existing["exit_reason_real"],
            trigger_bid,
            None if existing is None else existing["fill_price"],
            None if existing is None else existing["fill_delta_bps"],
            "accounting net is authoritative; fill slippage is separate from the policy decision",
        ),
    )
    conn.commit()
    return True


def note_real_fill_price(db_path: str, candidate_id: int | None, *, fill_price: float, exit_reason: str) -> None:
    """Separate venue fill slippage from the policy decision. Does not replace the accounting net."""
    if not candidate_id or fill_price <= 0:
        return
    conn = _connect(db_path)
    try:
        ensure_capture_schema(conn)
        row = conn.execute(
            "SELECT id, exit_bid FROM policy_episodes WHERE candidate_id=? AND capture_version=?",
            (int(candidate_id), CAPTURE),
        ).fetchone()
        if row is None:
            return
        trigger = row["exit_bid"]
        if not trigger:
            last_bid = conn.execute(
                "SELECT bid FROM policy_episode_checks WHERE episode_id=? AND bid IS NOT NULL ORDER BY checked_at DESC LIMIT 1",
                (int(row["id"]),),
            ).fetchone()
            trigger = None if last_bid is None else last_bid[0]
        delta = None
        if trigger:
            delta = (float(fill_price) - float(trigger)) / float(trigger) * 10000.0
        conn.execute(
            """
            INSERT INTO policy_fill_parity (
                episode_id, fill_price, fill_delta_bps, exit_reason_real, trigger_bid, slippage_note
            ) VALUES (?,?,?,?,?,?)
            ON CONFLICT(episode_id) DO UPDATE SET
                fill_price=excluded.fill_price,
                fill_delta_bps=excluded.fill_delta_bps,
                exit_reason_real=excluded.exit_reason_real,
                trigger_bid=COALESCE(policy_fill_parity.trigger_bid, excluded.trigger_bid),
                policy_match=CASE
                    WHEN COALESCE(policy_fill_parity.exit_reason_audit, '')='' THEN policy_fill_parity.policy_match
                    WHEN policy_fill_parity.exit_reason_audit=excluded.exit_reason_real THEN 1
                    ELSE 0
                END,
                slippage_note=excluded.slippage_note
            """,
            (
                int(row["id"]),
                float(fill_price),
                delta,
                str(exit_reason or ""),
                trigger,
                "venue fill versus the executable bid at the policy trigger",
            ),
        )
        conn.commit()
    except sqlite3.Error:
        logger.debug("EXACT_POLICY_FILL_PARITY_FAILED", exc_info=True)
    finally:
        conn.close()


def live_day_decision(db_path: str, position: Any, current_price: float, roundtrip_cost: float, *, now: float | None = None) -> dict[str, Any]:
    """The live DAY exit decision. Same function a refused episode uses, with this position's state."""
    moment = float(now if now is not None else time.time())
    entry = float(getattr(position, "entry_price", 0.0) or 0.0)
    mark = float(current_price)
    high = float(getattr(position, "highest_price", 0.0) or entry)
    low = float(getattr(position, "lowest_price", 0.0) or mark)
    entry_time = float(getattr(position, "entry_time", 0.0) or 0.0)
    adapt = getattr(position, "adaptive_decision", None) or {}
    if not isinstance(adapt, dict):
        adapt = {}
    learned = str(adapt.get("setup") or getattr(position, "entry_thesis", "") or "")
    prev = _heartbeat_prev_net(db_path, str(getattr(position, "trade_id", "") or ""))
    decision = decide_day_continuation(
        db_path,
        symbol=str(getattr(position, "symbol", "") or ""),
        setup=learned,
        exit_setup=str(getattr(position, "entry_thesis", "") or learned),
        regime=str(adapt.get("regime") or ""),
        entry_price=entry,
        mark=mark,
        low=low if low > 0 else mark,
        high=high if high > 0 else entry,
        prev_net=prev,
        age_sec=max(0.0, moment - entry_time) if entry_time else 0.0,
        atr_at_entry=_float(getattr(position, "atr_at_entry", 0.0)),
        structural_anchor=_float(getattr(position, "thesis_invalid_level", 0.0)),
        target_price=_float(getattr(position, "thesis_target_level", 0.0)),
        entry_time=entry_time,
        roundtrip_cost=float(roundtrip_cost),
        atr_1h_at_entry=_float(getattr(position, "day_atr_1h_at_entry", 0.0)),
        objective_structural=_float(getattr(position, "day_objective_structural", 0.0)),
        objective_atr_mult=_float(adapt.get("objective_atr_mult"), 1.0),
        structural_emphasis=_float(adapt.get("structural_emphasis"), 1.0),
        runner_activation_mult=_float(adapt.get("runner_activation_mult"), 1.0),
        runner_trail_mult=_float(adapt.get("runner_trail_mult"), 1.0),
        runner_tighten_mult=_float(adapt.get("runner_tighten_mult"), 1.0),
        now=moment,
    )
    try:
        decision["state_id"] = learner_state_id(db_path)
    except Exception:
        logger.debug("EXACT_POLICY_STATE_ID_FAILED", exc_info=True)
        decision["state_id"] = ""
    return decision


def _heartbeat_prev_net(db_path: str, trade_id: str) -> float | None:
    if not db_path or not trade_id:
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT net_unrealized_pct FROM ai_position_heartbeats WHERE trade_id=? ORDER BY epoch_ms DESC LIMIT 1 OFFSET 1",
            (trade_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row or row[0] is None:
        return None
    try:
        value = float(row[0])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def capture_counts(db_path: str) -> dict[str, int]:
    conn = _connect(db_path)
    try:
        ensure_capture_schema(conn)
        row = conn.execute(
            """
            SELECT
              SUM(CASE WHEN status='OPEN' THEN 1 ELSE 0 END),
              SUM(CASE WHEN status='CLOSED' AND kind=? THEN 1 ELSE 0 END),
              SUM(CASE WHEN status='CLOSED' AND kind=? THEN 1 ELSE 0 END),
              SUM(CASE WHEN parity_status='NON_PARITY' THEN 1 ELSE 0 END)
            FROM policy_episodes WHERE capture_version=?
            """,
            (REAL, COUNTERFACTUAL, CAPTURE),
        ).fetchone()
    except sqlite3.Error:
        return {"open": 0, "real": 0, "counterfactual": 0, "non_parity": 0}
    finally:
        conn.close()
    return {
        "open": int(row[0] or 0),
        "real": int(row[1] or 0),
        "counterfactual": int(row[2] or 0),
        "non_parity": int(row[3] or 0),
    }
