"""ENTRY_POLICY_VALUE_V1: expected deployed-policy net of a new DAY entry.

Continuation value answers whether an already-open position should be held.
This contract answers a different question: the expected net of entering a
candidate and then running the deployed exit policy.

The label on resolved exact-policy episode i is ``planning_net``:

    planning_net_i = executable_bid_net(entry_ask_i, exit_bid_i, canonical_roundtrip_cost)

``executable_bid_net`` subtracts the canonical planning cost once. The exit
half-spread is already inside the bid and is not subtracted again. The
estimate does not subtract the cost a second time.

The estimate is the hierarchical decayed mean of those labels (engine, setup,
related, exact key), prior 0. A cold key is 0. That does not clear the
existing executable-net funding check. Evidence count changes shrinkage and
uncertainty only. It is not a permission.

Several setup rows on one symbol-bar share one price path. Each keeps its own
label on its own key. Each observation's weight is 1 / n_setups on that bar,
so the shared path is one sample in the parent pools.

State is stored under economic version ``ENTRY_POLICY_VALUE_V1`` and metric
``entry_policy_net_v1``. Hold-advantage rows are not read and are not overwritten.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from typing import Any

from backend.services.adaptive_learning import (
    DAY_ENGINE,
    LATTICE_LEVELS,
    PRIOR_STRENGTH,
    _connect,
    _data_clock,
    _decay_factor,
    _is_forensic_key,
    _lattice,
    _norm_symbol,
)
from backend.services.policy_episode import canon_symbol

logger = logging.getLogger(__name__)

ENTRY_POLICY_VALUE_V1 = "ENTRY_POLICY_VALUE_V1"
ENTRY_POLICY_NET_V1 = "entry_policy_net_v1"
ENTRY_POLICY_PRIOR = 0.0
_CAPTURE = "EXACT_POLICY_V1"
_INELIGIBLE_REASONS = frozenset({"DISPOSITION_MISSING"})
_CACHE: dict[tuple, tuple[float, list]] = {}


def entry_funding_authorized(entry_expected_policy_net: float) -> bool:
    """Existing executable-net check. The argument is the policy estimate only."""
    try:
        value = float(entry_expected_policy_net)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def _stamp(moment: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(moment)))


def _cold() -> dict[str, Any]:
    zeros = dict.fromkeys(LATTICE_LEVELS, 0.0)
    return {
        "mean": ENTRY_POLICY_PRIOR,
        "prior": ENTRY_POLICY_PRIOR,
        "parent": ENTRY_POLICY_PRIOR,
        "levels": dict(zeros),
        "level_weights": dict(zeros),
        "key_weight": 0.0,
        "pooled_weight": 0.0,
        "n": 0.0,
        "confidence": 0.0,
        "sd": 0.0,
        "entry_policy_version": ENTRY_POLICY_VALUE_V1,
        "metric": ENTRY_POLICY_NET_V1,
    }


def _moments(prev_n: float, prev_mean: float, prev_m2: float, updated_at: str, value: float, weight: float, now_epoch: float) -> tuple[float, float, float]:
    """Decayed running mean. Same update the adaptive state table uses for means."""
    w = float(weight)
    if prev_n <= 0.0:
        return w, float(value), 0.0
    decay = _decay_factor(updated_at, float(now_epoch), DAY_ENGINE)
    n = float(prev_n) * decay + w
    alpha = w / n if n > 0.0 else 1.0
    delta = float(value) - float(prev_mean)
    mean = float(prev_mean) + alpha * delta
    m2 = max(0.0, float(prev_m2)) * decay + w * delta * (float(value) - mean)
    return n, mean, max(0.0, m2)


def _invalidate() -> None:
    _CACHE.clear()


def _rows(db_path: str) -> list[sqlite3.Row]:
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        return []
    try:
        return list(
            conn.execute(
                """
                SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at
                FROM adaptive_metric_state
                WHERE engine_id=? AND economic_version=? AND metric=?
                """,
                (DAY_ENGINE, ENTRY_POLICY_VALUE_V1, ENTRY_POLICY_NET_V1),
            )
        )
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def entry_policy_value(db_path: str, symbol: str, setup: str, regime: str, *, now: float | None = None) -> dict[str, Any]:
    """Shrunk mean of causal planning-net labels. Prior 0. No sample-count veto."""
    rows = _rows(db_path)
    if not rows:
        return _cold()
    moment = float(now) if now is not None else _data_clock(rows)
    view = _lattice(
        rows,
        engine_id=DAY_ENGINE,
        symbol=symbol,
        setup=setup,
        regime=regime,
        weights={ENTRY_POLICY_NET_V1: 1.0},
        prior=ENTRY_POLICY_PRIOR,
        now=moment,
        include_engine=True,
    )
    view["entry_policy_version"] = ENTRY_POLICY_VALUE_V1
    view["metric"] = ENTRY_POLICY_NET_V1
    return view


def record_entry_policy_observation(
    db_path: str,
    *,
    symbol: str,
    setup: str,
    regime: str,
    value: float,
    weight: float = 1.0,
    now: float | None = None,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """Fold one planning-net label into ENTRY_POLICY_VALUE_V1. Does not touch hold_adv."""
    if value is None or not math.isfinite(float(value)):
        return False
    w = float(weight)
    if not (math.isfinite(w) and w > 0.0):
        return False
    engine_symbol = _norm_symbol(symbol)
    engine_setup = str(setup or "").upper()
    engine_regime = str(regime or "").lower()
    if _is_forensic_key(DAY_ENGINE, engine_symbol, engine_setup, engine_regime):
        return False
    moment = float(now if now is not None else time.time())
    own = conn is None
    db = conn if conn is not None else _connect(db_path)
    try:
        key = (DAY_ENGINE, ENTRY_POLICY_VALUE_V1, engine_symbol, engine_setup, engine_regime, ENTRY_POLICY_NET_V1)
        row = db.execute(
            """
            SELECT n, ewma, m2, updated_at FROM adaptive_metric_state
            WHERE engine_id=? AND economic_version=? AND symbol=? AND setup=? AND regime=? AND metric=?
            """,
            key,
        ).fetchone()
        if row is None or float(row["n"] or 0.0) <= 0.0:
            n, mean, m2 = _moments(0.0, 0.0, 0.0, "", float(value), w, moment)
        else:
            n, mean, m2 = _moments(float(row["n"]), float(row["ewma"]), float(row["m2"] or 0.0), str(row["updated_at"] or ""), float(value), w, moment)
        db.execute(
            """
            INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_id, economic_version, symbol, setup, regime, metric) DO UPDATE SET
                n=excluded.n, ewma=excluded.ewma, m2=excluded.m2, updated_at=excluded.updated_at
            """,
            (*key, n, mean, m2, _stamp(moment)),
        )
        if own:
            db.commit()
        _invalidate()
        return True
    except sqlite3.Error:
        logger.warning("ENTRY_POLICY_FOLD_FAILED", exc_info=True)
        return False
    finally:
        if own:
            db.close()


def ensure_entry_policy_column(conn: sqlite3.Connection) -> None:
    present = {str(row[1]) for row in conn.execute("PRAGMA table_info(policy_episodes)")}
    if "entry_policy_learned" not in present:
        conn.execute("ALTER TABLE policy_episodes ADD COLUMN entry_policy_learned INTEGER NOT NULL DEFAULT 0")


def _opportunity_counts(conn: sqlite3.Connection) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for symbol, decided in conn.execute(
        "SELECT symbol, decided_at FROM policy_episodes WHERE capture_version=? AND engine_id=?",
        (_CAPTURE, DAY_ENGINE),
    ):
        try:
            bar = int(float(decided) // 900.0) * 900
        except (TypeError, ValueError):
            continue
        key = (canon_symbol(str(symbol)), bar)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _planning_value(row: sqlite3.Row) -> float | None:
    stored = row["planning_net"]
    if stored is not None:
        try:
            value = float(stored)
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value):
            return value
    entry = row["entry_ask"]
    bid = row["exit_bid"]
    if entry is None or bid is None:
        return None
    try:
        entry_f = float(entry)
        bid_f = float(bid)
    except (TypeError, ValueError):
        return None
    if entry_f <= 0.0 or not math.isfinite(bid_f):
        return None
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
    from backend.services.adaptive_learning import executable_bid_net

    cost = row["roundtrip_cost"] if row["roundtrip_cost"] is not None else ESTIMATED_ROUNDTRIP_COST
    return float(executable_bid_net(entry_f, bid_f, float(cost)))


def _eligible(row: sqlite3.Row) -> bool:
    if str(row["status"] or "") != "CLOSED":
        return False
    if str(row["capture_version"] or "") != _CAPTURE or str(row["engine_id"] or "") != DAY_ENGINE:
        return False
    if row["label_available_at"] is None:
        return False
    reason = str(row["exit_reason"] or "")
    if reason in _INELIGIBLE_REASONS:
        return False
    if str(row["parity_status"] or "") == "NON_PARITY":
        return False
    kind = str(row["kind"] or "")
    if kind == "REAL_POLICY_OUTCOME":
        return True
    return int(row["train_eligible"] or 0) == 1


def apply_resolved_entry_policy_labels(db_path: str, *, now: float | None = None, available_before: float | None = None) -> int:
    """Fold resolved exact-policy labels in label-time order. Already-folded rows are skipped.

    ``available_before`` keeps a label out of state until that timestamp. A
    historical replay passes the decision time. Live learning passes none, so
    every label whose availability is at or before ``now`` is eligible.
    """
    moment = float(now if now is not None else time.time())
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        return 0
    applied = 0
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "policy_episodes" not in tables:
            return 0
        ensure_entry_policy_column(conn)
        counts = _opportunity_counts(conn)
        rows = conn.execute(
            """
            SELECT id, symbol, setup, regime, decided_at, entry_ask, exit_bid, planning_net,
                   roundtrip_cost, label_available_at, kind, train_eligible, parity_status,
                   exit_reason, status, capture_version, engine_id
            FROM policy_episodes
            WHERE capture_version=? AND engine_id=? AND COALESCE(entry_policy_learned, 0)=0
            ORDER BY label_available_at, id
            """,
            (_CAPTURE, DAY_ENGINE),
        ).fetchall()
        for row in rows:
            available = row["label_available_at"]
            try:
                available_f = float(available)
            except (TypeError, ValueError):
                continue
            if available_f > moment:
                continue
            if available_before is not None and not (available_f < float(available_before)):
                continue
            if not _eligible(row):
                continue
            value = _planning_value(row)
            if value is None:
                continue
            try:
                bar = int(float(row["decided_at"]) // 900.0) * 900
            except (TypeError, ValueError):
                continue
            siblings = counts.get((canon_symbol(str(row["symbol"])), bar), 1)
            weight = 1.0 / float(max(1, siblings))
            conn.execute("SAVEPOINT mystic_entry_policy")
            try:
                folded = record_entry_policy_observation(
                    db_path,
                    symbol=str(row["symbol"]),
                    setup=str(row["setup"]),
                    regime=str(row["regime"]),
                    value=value,
                    weight=weight,
                    now=available_f,
                    conn=conn,
                )
                if not folded:
                    conn.execute("ROLLBACK TO mystic_entry_policy")
                    conn.execute("RELEASE mystic_entry_policy")
                    continue
                marked = conn.execute(
                    "UPDATE policy_episodes SET entry_policy_learned=1 WHERE id=? AND COALESCE(entry_policy_learned, 0)=0",
                    (int(row["id"]),),
                )
                if marked.rowcount != 1:
                    conn.execute("ROLLBACK TO mystic_entry_policy")
                    conn.execute("RELEASE mystic_entry_policy")
                    continue
                conn.execute("RELEASE mystic_entry_policy")
                applied += 1
            except Exception:
                conn.execute("ROLLBACK TO mystic_entry_policy")
                conn.execute("RELEASE mystic_entry_policy")
                logger.warning("ENTRY_POLICY_LABEL_FAILED id=%s", row["id"], exc_info=True)
        conn.commit()
    except sqlite3.Error:
        logger.warning("ENTRY_POLICY_APPLY_FAILED", exc_info=True)
        return 0
    finally:
        conn.close()
    if applied:
        logger.info("ENTRY_POLICY_LABELS_APPLIED n=%s version=%s", applied, ENTRY_POLICY_VALUE_V1)
    return applied


def _fold_memory(state: dict[tuple[str, str, str], dict[str, Any]], episode: dict[str, Any]) -> None:
    symbol = _norm_symbol(str(episode.get("symbol") or ""))
    setup = str(episode.get("setup") or "").upper()
    regime = str(episode.get("regime") or "").lower()
    key = (symbol, setup, regime)
    prev = state.get(key)
    moment = float(episode["label_available_at"])
    value = float(episode["planning_net"])
    weight = float(episode.get("weight") or 1.0)
    if prev is None:
        n, mean, m2 = _moments(0.0, 0.0, 0.0, "", value, weight, moment)
    else:
        n, mean, m2 = _moments(float(prev["n"]), float(prev["ewma"]), float(prev["m2"]), str(prev["updated_at"]), value, weight, moment)
    state[key] = {"n": n, "ewma": mean, "m2": m2, "updated_at": _stamp(moment)}


def _posterior_memory(state: dict[tuple[str, str, str], dict[str, Any]], symbol: str, setup: str, regime: str, now: float) -> dict[str, Any]:
    if not state:
        return _cold()
    rows = []
    for (sym, stp, reg), rec in state.items():
        rows.append(
            {
                "symbol": sym,
                "setup": stp,
                "regime": reg,
                "metric": ENTRY_POLICY_NET_V1,
                "n": rec["n"],
                "ewma": rec["ewma"],
                "m2": rec["m2"],
                "updated_at": rec["updated_at"],
            }
        )
    view = _lattice(
        rows,
        engine_id=DAY_ENGINE,
        symbol=symbol,
        setup=setup,
        regime=regime,
        weights={ENTRY_POLICY_NET_V1: 1.0},
        prior=ENTRY_POLICY_PRIOR,
        now=float(now),
        include_engine=True,
    )
    view["entry_policy_version"] = ENTRY_POLICY_VALUE_V1
    return view


def causal_replay(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every episode at its decision time from labels available strictly earlier.

    ``episodes`` carry symbol, setup, regime, decided_at, and, when resolved,
    label_available_at, planning_net, weight, and eligible. The returned
    prediction is what ENTRY_POLICY_VALUE_V1 would have said at decided_at.
    """
    labels = []
    for episode in episodes:
        if not episode.get("eligible"):
            continue
        if episode.get("label_available_at") is None or episode.get("planning_net") is None:
            continue
        try:
            float(episode["label_available_at"])
            float(episode["planning_net"])
        except (TypeError, ValueError):
            continue
        labels.append(episode)
    labels.sort(key=lambda item: (float(item["label_available_at"]), int(item.get("id") or 0)))
    decisions = sorted(episodes, key=lambda item: (float(item.get("decided_at") or 0.0), int(item.get("id") or 0)))
    state: dict[tuple[str, str, str], dict[str, Any]] = {}
    cursor = 0
    scored: list[dict[str, Any]] = []
    for episode in decisions:
        moment = float(episode.get("decided_at") or 0.0)
        while cursor < len(labels) and float(labels[cursor]["label_available_at"]) < moment:
            _fold_memory(state, labels[cursor])
            cursor += 1
        view = _posterior_memory(state, str(episode.get("symbol") or ""), str(episode.get("setup") or ""), str(episode.get("regime") or ""), moment)
        scored.append(
            {
                "id": episode.get("id"),
                "symbol": episode.get("symbol"),
                "setup": episode.get("setup"),
                "regime": episode.get("regime"),
                "decided_at": moment,
                "planning_net": episode.get("planning_net"),
                "prediction": float(view["mean"]),
                "n": float(view["n"]),
                "sd": float(view["sd"]),
                "parent": float(view["parent"]),
            }
        )
    return scored


def shrinkage_strength() -> float:
    """Prior sample count. Exposed so tests can show low counts shrink rather than reject."""
    return float(PRIOR_STRENGTH)
