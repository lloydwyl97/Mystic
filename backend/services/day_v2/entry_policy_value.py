"""ENTRY_POLICY_VALUE_V2: expected deployed-policy net of a new DAY entry.

Continuation value answers whether an already-open position should be held.
This contract answers a different question: the expected net of entering a
candidate and then running the deployed exit policy.

Target provenance is explicit:
  - ``REAL_FUNDED_EXECUTION`` on ``REAL_POLICY_OUTCOME`` episodes uses the
    authoritative confirmed fill-to-fill realized accounting net (including
    actual exchange fees). Both ``planning_net`` and realized ``net`` remain
    on ``policy_episodes`` for audit.
  - ``COUNTERFACTUAL_POLICY`` on ``COUNTERFACTUAL_POLICY_OUTCOME`` episodes uses
    ``planning_net = executable_bid_net(entry_ask, exit_bid, roundtrip_cost)``,
    which subtracts canonical planning cost once without fabricating slippage.

Observation weights are level-specific and causally non-anticipative:
  - Engine level (``entry_policy_engine_net_v2``): sibling setups admitted on the
    same symbol-bar normalize by ``1 / n_setups_on_symbol_bar`` so one bar does
    not contribute multiple independent units to the cross-setup engine prior.
  - Specific levels — setup, related, and exact key (``entry_policy_net_v2``):
    an independent observation of that setup receives weight ``1.0`` and is not
    diluted by sibling setups on the admission bar.
  - Overlapping unfunded counterfactual episodes on the same ``(symbol, setup)``
    trajectory form one causally connected occupancy group whose total specific
    weight across all resolving members is bounded at ``<= 1.0``.
  - A genuinely funded real DAY trade receives specific weight ``1.0`` and is
    never double-counted with any shadow counterfactual copy or unfunded episode
    admitted while that real position occupied the symbol.

Historical ``ENTRY_POLICY_VALUE_V1`` (``entry_policy_net_v1``) rows and the V1
``entry_policy_learned`` flag are preserved untouched for audit/reproducibility.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from collections.abc import Mapping
from typing import Any

from backend.services.adaptive_learning import (
    _LEVEL_RELEVANCE,
    ADAPTIVE_STATE_VERSION,
    DAY_ENGINE,
    LATTICE_LEVELS,
    PRIOR_STRENGTH,
    _connect,
    _data_clock,
    _decay_factor,
    _is_forensic_key,
    _lattice,
    _level_of,
    _norm_symbol,
)
from backend.services.day_v2.entry_policy_value_v3 import (
    ENTRY_POLICY_NET_V3,
    ENTRY_POLICY_V3_FEATURE_VERSION,
    ENTRY_POLICY_VALUE_V3,
    POPULATION_DETECTOR_FIRED,
    POPULATION_MARKET_FALLBACK,
    apply_resolved_entry_policy_v3_labels,
    causal_replay_v3,
    entry_policy_value_v3,
    invalidate_v3_cache,
    rebuild_entry_policy_value_v3,
    record_entry_policy_v3_observation,
    v3_feature_contract,
    v3_prior_net,
)
from backend.services.policy_episode import canon_symbol

logger = logging.getLogger(__name__)

ENTRY_POLICY_VALUE_V1 = "ENTRY_POLICY_VALUE_V1"
ENTRY_POLICY_NET_V1 = "entry_policy_net_v1"

ENTRY_POLICY_VALUE_V2 = "ENTRY_POLICY_VALUE_V2"
ENTRY_POLICY_NET_V2 = "entry_policy_net_v2"
ENTRY_POLICY_ENGINE_NET_V2 = "entry_policy_engine_net_v2"

LABEL_REAL_FUNDED_EXECUTION = "REAL_FUNDED_EXECUTION"
LABEL_COUNTERFACTUAL_POLICY = "COUNTERFACTUAL_POLICY"

ENTRY_POLICY_PRIOR = 0.0
_CAPTURE = "EXACT_POLICY_V1"
_INELIGIBLE_REASONS = frozenset({"DISPOSITION_MISSING"})
_CACHE: dict[tuple, tuple[float, list]] = {}

_V2_EPISODE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("entry_policy_learned", "INTEGER NOT NULL DEFAULT 0"),
    ("entry_policy_v2_learned", "INTEGER NOT NULL DEFAULT 0"),
    ("entry_policy_v2_target", "REAL"),
    ("entry_policy_v2_target_type", "TEXT"),
    ("entry_policy_v2_specific_weight", "REAL"),
    ("entry_policy_v2_engine_weight", "REAL"),
    ("entry_policy_v2_overlap_group", "TEXT"),
)


def entry_funding_authorized(entry_expected_policy_net: float) -> bool:
    """Existing executable-net check. The argument is the policy estimate only."""
    try:
        value = float(entry_expected_policy_net)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def _stamp(moment: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(moment)))


def _cold(version: str = ENTRY_POLICY_VALUE_V2) -> dict[str, Any]:
    zeros = dict.fromkeys(LATTICE_LEVELS, 0.0)
    metric = ENTRY_POLICY_NET_V1 if version == ENTRY_POLICY_VALUE_V1 else ENTRY_POLICY_NET_V2
    return {
        "mean": ENTRY_POLICY_PRIOR,
        "prior": ENTRY_POLICY_PRIOR,
        "parent": ENTRY_POLICY_PRIOR,
        "levels": dict(zeros),
        "level_weights": dict(zeros),
        "key_weight": 0.0,
        "engine_weight": 0.0,
        "pooled_weight": 0.0,
        "n": 0.0,
        "confidence": 0.0,
        "sd": 0.0,
        "engine_sd": 0.0,
        "setup_sd": 0.0,
        "related_sd": 0.0,
        "key_sd": 0.0,
        "entry_policy_version": version,
        "metric": metric,
    }


def _moments(
    prev_n: float,
    prev_mean: float,
    prev_m2: float,
    updated_at: str,
    value: float,
    weight: float,
    now_epoch: float,
) -> tuple[float, float, float]:
    """Decayed running mean and second-moment accumulator."""
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
    invalidate_v3_cache()


def _rows_v1(db_path: str) -> list[sqlite3.Row]:
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


def _rows_v2(db_path: str) -> list[sqlite3.Row]:
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
                WHERE engine_id=? AND economic_version=? AND metric IN (?, ?)
                """,
                (DAY_ENGINE, ENTRY_POLICY_VALUE_V2, ENTRY_POLICY_NET_V2, ENTRY_POLICY_ENGINE_NET_V2),
            )
        )
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _empirical_sd(weight: float, sum_wx: float, sum_wx2: float) -> float:
    if weight <= 0.0:
        return 0.0
    mean = sum_wx / weight
    var = max(0.0, sum_wx2 / weight - mean * mean)
    return math.sqrt(var)


def _lattice_v2(
    rows: list[Mapping[str, Any] | sqlite3.Row],
    *,
    symbol: str,
    setup: str,
    regime: str,
    prior: float,
    now: float | None = None,
) -> dict[str, Any]:
    """Per-level hierarchical posterior for ENTRY_POLICY_VALUE_V2.

    The engine level aggregates ``entry_policy_engine_net_v2`` (sibling- and
    trajectory-normalized). Setup, related, and exact-key levels aggregate
    ``entry_policy_net_v2`` (specific trajectory-normalized, never diluted by
    sibling setups on the same admission bar).
    """
    if not rows:
        return _cold(ENTRY_POLICY_VALUE_V2)
    sym = _norm_symbol(symbol)
    stp = str(setup or "").upper()
    reg = str(regime or "").lower()
    moment = float(now) if now is not None else _data_clock(rows)  # type: ignore[arg-type]

    eng_acc = [0.0, 0.0, 0.0]
    spec_acc = {lvl: [0.0, 0.0, 0.0] for lvl in LATTICE_LEVELS}
    has_eng_rows = False

    for row in rows:
        metric = str(row["metric"] or "")
        raw_n = float(row["n"] or 0.0)
        if raw_n <= 0.0 or _is_forensic_key(DAY_ENGINE, row["symbol"], row["setup"], row["regime"]):
            continue
        w = raw_n * _decay_factor(str(row["updated_at"] or ""), moment, DAY_ENGINE)
        if w <= 0.0:
            continue
        mean = float(row["ewma"])
        var = max(0.0, float(row["m2"] or 0.0)) / raw_n
        second = w * (var + mean * mean)
        if metric == ENTRY_POLICY_ENGINE_NET_V2:
            has_eng_rows = True
            eng_acc[0] += w
            eng_acc[1] += w * mean
            eng_acc[2] += second
        elif metric == ENTRY_POLICY_NET_V2:
            lvl = _level_of(row, sym, stp, reg)  # type: ignore[arg-type]
            b = spec_acc[lvl]
            b[0] += w
            b[1] += w * mean
            b[2] += second

    if not has_eng_rows:
        for b in spec_acc.values():
            eng_acc[0] += b[0]
            eng_acc[1] += b[1]
            eng_acc[2] += b[2]

    ps = float(PRIOR_STRENGTH)
    w_engine, s_engine, q_engine = eng_acc
    mu_engine = (ps * prior + s_engine) / (ps + w_engine)

    w_setup = sum(spec_acc[lvl][0] for lvl in ("setup", "related", "key"))
    s_setup = sum(spec_acc[lvl][1] for lvl in ("setup", "related", "key"))
    q_setup = sum(spec_acc[lvl][2] for lvl in ("setup", "related", "key"))
    mu_setup = (ps * mu_engine + s_setup) / (ps + w_setup)

    w_related = sum(spec_acc[lvl][0] for lvl in ("related", "key"))
    s_related = sum(spec_acc[lvl][1] for lvl in ("related", "key"))
    q_related = sum(spec_acc[lvl][2] for lvl in ("related", "key"))
    mu_related = (ps * mu_setup + s_related) / (ps + w_related)

    w_key, s_key, q_key = spec_acc["key"]
    mu_key = (ps * mu_related + s_key) / (ps + w_key)

    level_means = {
        "engine": mu_engine,
        "setup": mu_setup,
        "related": mu_related,
        "key": mu_key,
    }
    nested_w = {
        "engine": w_engine,
        "setup": w_setup,
        "related": w_related,
        "key": w_key,
    }

    engine_sd = _empirical_sd(w_engine, s_engine, q_engine)
    setup_sd = _empirical_sd(w_setup, s_setup, q_setup)
    related_sd = _empirical_sd(w_related, s_related, q_related)
    key_sd = _empirical_sd(w_key, s_key, q_key)

    relevant = sum(spec_acc[lvl][0] * _LEVEL_RELEVANCE[lvl] for lvl in LATTICE_LEVELS)
    sigma2 = engine_sd * engine_sd
    return {
        "mean": mu_key,
        "prior": prior,
        "parent": mu_related,
        "levels": level_means,
        "level_weights": nested_w,
        "key_weight": w_key,
        "engine_weight": w_engine,
        "pooled_weight": w_setup - w_key,
        "key_n": {ENTRY_POLICY_NET_V2: w_key},
        "n": w_key,
        "confidence": relevant / (ps + relevant),
        "sd": math.sqrt(sigma2 / (ps + relevant)) if sigma2 > 0.0 else 0.0,
        "engine_sd": engine_sd,
        "setup_sd": setup_sd,
        "related_sd": related_sd,
        "key_sd": key_sd,
        "version": ADAPTIVE_STATE_VERSION,
        "entry_policy_version": ENTRY_POLICY_VALUE_V2,
        "metric": ENTRY_POLICY_NET_V2,
    }


def entry_policy_value(
    db_path: str,
    symbol: str,
    setup: str,
    regime: str,
    *,
    features: Mapping[str, Any] | None = None,
    now: float | None = None,
    version: str = ENTRY_POLICY_VALUE_V2,
) -> dict[str, Any]:
    """Shrunk causal policy estimator (V1, V2, or V3). No sample-count veto."""
    if version == ENTRY_POLICY_VALUE_V3:
        return entry_policy_value_v3(
            db_path,
            symbol,
            setup,
            regime,
            features=features,
            now=now,
        )
    if version == ENTRY_POLICY_VALUE_V1:
        rows_v1 = _rows_v1(db_path)
        if not rows_v1:
            return _cold(ENTRY_POLICY_VALUE_V1)
        moment = float(now) if now is not None else _data_clock(rows_v1)
        view = _lattice(
            rows_v1,
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
        view.setdefault("engine_weight", float(view["level_weights"]["engine"]))
        view.setdefault("engine_sd", 0.0)
        view.setdefault("setup_sd", 0.0)
        view.setdefault("related_sd", 0.0)
        view.setdefault("key_sd", 0.0)
        return view

    rows_v2 = _rows_v2(db_path)
    if not rows_v2:
        return _cold(ENTRY_POLICY_VALUE_V2)
    return _lattice_v2(
        rows_v2,
        symbol=symbol,
        setup=setup,
        regime=regime,
        prior=ENTRY_POLICY_PRIOR,
        now=now,
    )


def _upsert_metric_moment(
    db: sqlite3.Connection,
    *,
    economic_version: str,
    symbol: str,
    setup: str,
    regime: str,
    metric: str,
    value: float,
    weight: float,
    moment: float,
) -> None:
    key = (DAY_ENGINE, economic_version, symbol, setup, regime, metric)
    row = db.execute(
        """
        SELECT n, ewma, m2, updated_at FROM adaptive_metric_state
        WHERE engine_id=? AND economic_version=? AND symbol=? AND setup=? AND regime=? AND metric=?
        """,
        key,
    ).fetchone()
    if row is None or float(row["n"] or 0.0) <= 0.0:
        n, mean, m2 = _moments(0.0, 0.0, 0.0, "", float(value), float(weight), moment)
    else:
        n, mean, m2 = _moments(
            float(row["n"]),
            float(row["ewma"]),
            float(row["m2"] or 0.0),
            str(row["updated_at"] or ""),
            float(value),
            float(weight),
            moment,
        )
    db.execute(
        """
        INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(engine_id, economic_version, symbol, setup, regime, metric) DO UPDATE SET
            n=excluded.n, ewma=excluded.ewma, m2=excluded.m2, updated_at=excluded.updated_at
        """,
        (*key, n, mean, m2, _stamp(moment)),
    )


def record_entry_policy_observation(
    db_path: str,
    *,
    symbol: str,
    setup: str,
    regime: str,
    value: float,
    weight: float = 1.0,
    specific_weight: float | None = None,
    engine_weight: float | None = None,
    version: str = ENTRY_POLICY_VALUE_V2,
    now: float | None = None,
    decided_at: float | None = None,
    features: Mapping[str, Any] | None = None,
    cluster_id: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> bool:
    """Fold one causal policy label into ENTRY_POLICY_VALUE_V2 (or V1/V3 if requested). Does not touch hold_adv."""
    if version == ENTRY_POLICY_VALUE_V3:
        if value is None or not math.isfinite(float(value)):
            return False
        w_v3 = float(specific_weight if specific_weight is not None else weight)
        if not (math.isfinite(w_v3) and w_v3 > 0.0):
            return False
        return record_entry_policy_v3_observation(
            db_path,
            symbol=symbol,
            setup=setup,
            regime=regime,
            value=float(value),
            weight=w_v3,
            now=now,
            decided_at=decided_at,
            features=features,
            cluster_id=cluster_id,
        )
    if value is None or not math.isfinite(float(value)):
        return False
    w_spec = float(specific_weight if specific_weight is not None else weight)
    w_eng = float(engine_weight if engine_weight is not None else weight)
    if version == ENTRY_POLICY_VALUE_V1:
        w_v1 = float(weight)
        if not (math.isfinite(w_v1) and w_v1 > 0.0):
            return False
    elif not (math.isfinite(w_spec) and math.isfinite(w_eng) and (w_spec > 0.0 or w_eng > 0.0)):
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
        if version == ENTRY_POLICY_VALUE_V1:
            _upsert_metric_moment(
                db,
                economic_version=ENTRY_POLICY_VALUE_V1,
                symbol=engine_symbol,
                setup=engine_setup,
                regime=engine_regime,
                metric=ENTRY_POLICY_NET_V1,
                value=float(value),
                weight=float(weight),
                moment=moment,
            )
        else:
            if w_spec > 0.0:
                _upsert_metric_moment(
                    db,
                    economic_version=ENTRY_POLICY_VALUE_V2,
                    symbol=engine_symbol,
                    setup=engine_setup,
                    regime=engine_regime,
                    metric=ENTRY_POLICY_NET_V2,
                    value=float(value),
                    weight=w_spec,
                    moment=moment,
                )
            if w_eng > 0.0:
                _upsert_metric_moment(
                    db,
                    economic_version=ENTRY_POLICY_VALUE_V2,
                    symbol=engine_symbol,
                    setup=engine_setup,
                    regime=engine_regime,
                    metric=ENTRY_POLICY_ENGINE_NET_V2,
                    value=float(value),
                    weight=w_eng,
                    moment=moment,
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
    for name, decl in _V2_EPISODE_COLUMNS:
        if name not in present:
            conn.execute(f"ALTER TABLE policy_episodes ADD COLUMN {name} {decl}")


def _col(row: Mapping[str, Any] | sqlite3.Row, name: str, default: Any = None) -> Any:
    try:
        return row[name]
    except (KeyError, IndexError):
        return default


def _has_col(row: Mapping[str, Any] | sqlite3.Row, name: str) -> bool:
    try:
        row[name]
        return True
    except (KeyError, IndexError):
        return False


def _planning_value(row: Mapping[str, Any] | sqlite3.Row) -> float | None:
    stored = _col(row, "planning_net")
    if stored is not None:
        try:
            value = float(stored)
        except (TypeError, ValueError):
            value = None
        if value is not None and math.isfinite(value):
            return value
    entry = _col(row, "entry_ask")
    bid = _col(row, "exit_bid")
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

    rc = _col(row, "roundtrip_cost")
    cost = rc if rc is not None else ESTIMATED_ROUNDTRIP_COST
    return float(executable_bid_net(entry_f, bid_f, float(cost)))


def resolve_episode_target(row: Mapping[str, Any] | sqlite3.Row) -> tuple[float | None, str | None]:
    """Return ``(target_net, target_type)`` according to episode provenance.

    ``REAL_POLICY_OUTCOME`` learns authoritative confirmed fill-to-fill
    realized accounting net (never stale pre-ranking ``planning_net``).
    ``COUNTERFACTUAL_POLICY_OUTCOME`` learns ``planning_net`` based on
    decision-time ask, live future exit bid, and canonical planning cost.
    """
    kind = str(_col(row, "kind") or "")
    if kind == "REAL_POLICY_OUTCOME":
        acm_filled = _col(row, "acm_filled")
        acm_net = _col(row, "acm_realized_net")
        if acm_net is not None and (acm_filled is None or int(acm_filled or 0) == 1):
            try:
                v = float(acm_net)
                if math.isfinite(v):
                    return v, LABEL_REAL_FUNDED_EXECUTION
            except (TypeError, ValueError):
                pass
        fp_net = _col(row, "fp_accounting_net")
        if fp_net is not None:
            try:
                v = float(fp_net)
                if math.isfinite(v):
                    return v, LABEL_REAL_FUNDED_EXECUTION
            except (TypeError, ValueError):
                pass
        raw_net = _col(row, "net")
        if raw_net is not None:
            try:
                v = float(raw_net)
                if math.isfinite(v):
                    return v, LABEL_REAL_FUNDED_EXECUTION
            except (TypeError, ValueError):
                pass
        return None, None

    val = _planning_value(row)
    if val is None or not math.isfinite(val):
        return None, None
    return float(val), LABEL_COUNTERFACTUAL_POLICY


def _eligible(row: Mapping[str, Any] | sqlite3.Row) -> bool:
    if str(_col(row, "status", "CLOSED") or "") != "CLOSED":
        return False
    if _has_col(row, "capture_version") and str(_col(row, "capture_version") or "") != _CAPTURE:
        return False
    if _has_col(row, "engine_id") and str(_col(row, "engine_id") or "") != DAY_ENGINE:
        return False
    if _col(row, "label_available_at") is None:
        return False
    reason = str(_col(row, "exit_reason") or "")
    if reason in _INELIGIBLE_REASONS:
        return False
    if str(_col(row, "parity_status") or "") == "NON_PARITY":
        return False
    kind = str(_col(row, "kind") or "")
    if kind == "REAL_POLICY_OUTCOME":
        return True
    elig = _col(row, "eligible")
    if elig is not None:
        return bool(elig)
    return int(_col(row, "train_eligible") or 0) == 1


def _bar_bucket(decided_at: float) -> int:
    return int(float(decided_at) // 900.0) * 900


def _is_funded_real(row: Mapping[str, Any] | sqlite3.Row) -> bool:
    if str(_col(row, "kind") or "") == "REAL_POLICY_OUTCOME":
        return True
    funded = _col(row, "funded")
    return funded is not None and int(funded or 0) == 1


def _episode_end(row: Mapping[str, Any] | sqlite3.Row) -> float:
    for col in ("exit_at", "label_available_at"):
        val = _col(row, col)
        if val is not None:
            try:
                f = float(val)
                if math.isfinite(f):
                    return f
            except (TypeError, ValueError):
                continue
    return math.inf


def plan_causal_v2_episodes(
    rows: list[Mapping[str, Any] | sqlite3.Row],
    *,
    cutoff: float,
) -> dict[int, dict[str, Any]]:
    """Assign causal overlap groups, per-level weights, and target provenance up to ``cutoff``.

    Grouping uses ONLY ``engine_id``, ``symbol``, ``setup``, ``funded``/``kind``,
    ``decided_at``, and whether previously admitted episodes had already closed
    at ``decided_at`` or prior to ``label_available_at``. No future outcomes or
    future arrivals after ``cutoff`` are read.
    """
    admitted: list[Mapping[str, Any] | sqlite3.Row] = []
    for row in rows:
        try:
            d = float(row["decided_at"])
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(d) and d <= cutoff:
            admitted.append(row)

    admitted.sort(key=lambda r: (float(r["decided_at"]), int(r["id"] or 0)))

    # 1. Identify funded real decisions and their live symbol-occupancy intervals
    funded_cids: set[int] = set()
    funded_tids: set[str] = set()
    funded_bar_setups: set[tuple[str, str, int]] = set()
    real_occupancy: dict[str, list[tuple[float, float, int]]] = {}

    for r in admitted:
        if not _is_funded_real(r):
            continue
        rid = int(r["id"] or 0)
        cid = int(_col(r, "candidate_id") or 0)
        if cid > 0:
            funded_cids.add(cid)
        tid = str(_col(r, "trade_id") or "").strip()
        if tid:
            funded_tids.add(tid)
        sym = canon_symbol(str(r["symbol"] or ""))
        stp = str(r["setup"] or "").upper()
        d = float(r["decided_at"])
        funded_bar_setups.add((sym, stp, _bar_bucket(d)))
        real_occupancy.setdefault(sym, []).append((d, _episode_end(r), rid))

    # 2. Classify shadow/occupied counterfactual rows vs genuine candidates, and count setups per (symbol, bar)
    shadow_ids: set[int] = set()
    bar_setups: dict[tuple[str, int], set[str]] = {}
    for r in admitted:
        rid = int(r["id"] or 0)
        sym = canon_symbol(str(r["symbol"] or ""))
        stp = str(r["setup"] or "").upper()
        d = float(r["decided_at"])
        bar = _bar_bucket(d)
        if _is_funded_real(r):
            bar_setups.setdefault((sym, bar), set()).add(stp)
            continue
        cid = int(_col(r, "candidate_id") or 0)
        tid = str(_col(r, "trade_id") or "").strip()
        is_shadow = (
            (cid > 0 and cid in funded_cids)
            or (bool(tid) and tid in funded_tids)
            or ((sym, stp, bar) in funded_bar_setups)
            or any(d_real < d < x_real for d_real, x_real, _ in real_occupancy.get(sym, ()))
        )
        if is_shadow:
            shadow_ids.add(rid)
        else:
            bar_setups.setdefault((sym, bar), set()).add(stp)

    # 3. Assign causal connected occupancy trajectories per (symbol, setup) for unfunded counterfactuals
    cf_groups: dict[str, list[Mapping[str, Any] | sqlite3.Row]] = {}
    row_group: dict[int, str] = {}
    active_state: dict[tuple[str, str], tuple[str, float]] = {}

    for r in admitted:
        rid = int(r["id"] or 0)
        if _is_funded_real(r) or rid in shadow_ids:
            continue
        sym = canon_symbol(str(r["symbol"] or ""))
        stp = str(r["setup"] or "").upper()
        d = float(r["decided_at"])
        x = _episode_end(r)
        prev = active_state.get((sym, stp))
        if prev is not None and d < prev[1]:
            gid = prev[0]
            new_end = max(prev[1], x)
        else:
            gid = f"{DAY_ENGINE}:{sym}:{stp}:{int(d)}"
            new_end = x
        active_state[(sym, stp)] = (gid, new_end)
        row_group[rid] = gid
        cf_groups.setdefault(gid, []).append(r)

    plan: dict[int, dict[str, Any]] = {}

    # 4. Plan shadow counterfactual rows (zero weight, never double-counted)
    for r in admitted:
        rid = int(r["id"] or 0)
        if rid not in shadow_ids:
            continue
        avail = _col(r, "label_available_at")
        if avail is None:
            continue
        try:
            avail_f = float(avail)
        except (TypeError, ValueError):
            continue
        if avail_f <= cutoff:
            plan[rid] = {
                "id": rid,
                "fold": False,
                "mark_learned": str(_col(r, "status") or "") == "CLOSED",
                "target": None,
                "target_type": LABEL_COUNTERFACTUAL_POLICY,
                "specific_weight": 0.0,
                "engine_weight": 0.0,
                "overlap_group": "SHADOW_OF_REAL",
                "label_available_at": avail_f,
            }

    # 5. Plan funded real rows (authoritative full specific weight = 1.0)
    for r in admitted:
        if not _is_funded_real(r):
            continue
        rid = int(r["id"] or 0)
        avail = _col(r, "label_available_at")
        if avail is None:
            continue
        try:
            avail_f = float(avail)
        except (TypeError, ValueError):
            continue
        if avail_f > cutoff or not _eligible(r):
            continue
        target_val, target_type = resolve_episode_target(r)
        if target_val is None or target_type is None:
            continue
        sym = canon_symbol(str(r["symbol"] or ""))
        bar = _bar_bucket(float(r["decided_at"]))
        n_sib = max(1, len(bar_setups.get((sym, bar), ())))
        tid = str(_col(r, "trade_id") or "").strip()
        plan[rid] = {
            "id": rid,
            "fold": True,
            "mark_learned": True,
            "target": float(target_val),
            "target_type": target_type,
            "specific_weight": 1.0,
            "engine_weight": 1.0 / float(n_sib),
            "overlap_group": f"REAL:{tid or rid}",
            "label_available_at": avail_f,
        }

    # 6. Plan counterfactual trajectory groups causally at each resolution timestamp <= cutoff
    for gid, members in cf_groups.items():
        by_tau: dict[float, list[Mapping[str, Any] | sqlite3.Row]] = {}
        for m in members:
            avail = _col(m, "label_available_at")
            if avail is None:
                continue
            try:
                tau = float(avail)
            except (TypeError, ValueError):
                continue
            if math.isfinite(tau) and tau <= cutoff:
                by_tau.setdefault(tau, []).append(m)

        if not by_tau:
            continue

        remaining = 1.0
        for tau in sorted(by_tau):
            resolving = sorted(by_tau[tau], key=lambda item: int(item["id"] or 0))
            # Causally known members of gid at time tau that did not already resolve strictly before tau
            unresolved_count = 0
            for m in members:
                if float(m["decided_at"]) > tau:
                    continue
                m_avail = _col(m, "label_available_at")
                if m_avail is not None:
                    try:
                        if float(m_avail) < tau:
                            continue
                    except (TypeError, ValueError):
                        pass
                unresolved_count += 1
            denom = max(len(resolving), unresolved_count)
            share = (remaining / float(denom)) if (denom > 0 and remaining > 0.0) else 0.0
            remaining = max(0.0, remaining - float(len(resolving)) * share)

            for m in resolving:
                mid = int(m["id"] or 0)
                explicit_w = _col(m, "weight")
                sym = canon_symbol(str(m["symbol"] or ""))
                bar = _bar_bucket(float(m["decided_at"]))
                n_sib = max(1, len(bar_setups.get((sym, bar), ())))
                w_spec = float(share) if explicit_w is None else float(share) * float(explicit_w)
                w_eng = w_spec / float(n_sib)
                if not _eligible(m):
                    plan[mid] = {
                        "id": mid,
                        "fold": False,
                        "mark_learned": False,
                        "target": None,
                        "target_type": LABEL_COUNTERFACTUAL_POLICY,
                        "specific_weight": 0.0,
                        "engine_weight": 0.0,
                        "overlap_group": gid,
                        "label_available_at": tau,
                    }
                    continue
                target_val, target_type = resolve_episode_target(m)
                if target_val is None or target_type is None or w_spec <= 0.0:
                    continue
                plan[mid] = {
                    "id": mid,
                    "fold": True,
                    "mark_learned": True,
                    "target": float(target_val),
                    "target_type": target_type,
                    "specific_weight": w_spec,
                    "engine_weight": w_eng,
                    "overlap_group": gid,
                    "label_available_at": tau,
                }

    return plan


def _fetch_episode_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    pe_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(policy_episodes)")}
    funded_expr = "pe.funded" if "funded" in pe_cols else "NULL AS funded"
    exit_at_expr = "pe.exit_at" if "exit_at" in pe_cols else "pe.label_available_at AS exit_at"
    net_expr = "pe.net" if "net" in pe_cols else "NULL AS net"
    tid_expr = "pe.trade_id" if "trade_id" in pe_cols else "NULL AS trade_id"
    cid_expr = "pe.candidate_id" if "candidate_id" in pe_cols else "0 AS candidate_id"

    acm_select = "NULL AS acm_filled, NULL AS acm_realized_net"
    acm_join = ""
    if "adaptive_candidate_markouts" in tables and "candidate_id" in pe_cols:
        acm_select = "acm.filled AS acm_filled, acm.realized_net AS acm_realized_net"
        acm_join = "LEFT JOIN adaptive_candidate_markouts acm ON acm.id = pe.candidate_id"

    fp_select = "NULL AS fp_accounting_net"
    fp_join = ""
    if "policy_fill_parity" in tables:
        fp_select = "fp.accounting_net AS fp_accounting_net"
        fp_join = "LEFT JOIN policy_fill_parity fp ON fp.episode_id = pe.id"

    sql = f"""
        SELECT pe.id, {cid_expr}, {tid_expr}, pe.symbol, pe.setup, pe.regime,
               pe.decided_at, {exit_at_expr}, pe.entry_ask, pe.exit_bid,
               {net_expr}, pe.planning_net, pe.roundtrip_cost, pe.label_available_at,
               pe.kind, {funded_expr}, pe.train_eligible, pe.parity_status,
               pe.exit_reason, pe.status, pe.capture_version, pe.engine_id,
               COALESCE(pe.entry_policy_v2_learned, 0) AS entry_policy_v2_learned,
               {acm_select}, {fp_select}
        FROM policy_episodes pe
        {acm_join}
        {fp_join}
        WHERE pe.capture_version=? AND pe.engine_id=?
        ORDER BY pe.label_available_at ASC, pe.id ASC
    """
    return list(conn.execute(sql, (_CAPTURE, DAY_ENGINE)).fetchall())


def _apply_resolved_v2_on_conn(
    conn: sqlite3.Connection,
    db_path: str,
    *,
    moment: float,
    available_before: float | None = None,
) -> int:
    ensure_entry_policy_column(conn)
    cutoff = moment if available_before is None else min(moment, math.nextafter(float(available_before), -math.inf))

    has_pending = conn.execute(
        """
        SELECT 1 FROM policy_episodes
        WHERE capture_version=? AND engine_id=? AND status='CLOSED'
          AND COALESCE(entry_policy_v2_learned, 0)=0
          AND label_available_at IS NOT NULL AND label_available_at <= ?
          AND COALESCE(parity_status, '') != 'NON_PARITY'
          AND COALESCE(exit_reason, '') != 'DISPOSITION_MISSING'
          AND (kind='REAL_POLICY_OUTCOME' OR COALESCE(train_eligible, 0)=1)
        LIMIT 1
        """,
        (_CAPTURE, DAY_ENGINE, cutoff),
    ).fetchone()
    if has_pending is None:
        return 0

    all_rows = _fetch_episode_rows(conn)
    plan = plan_causal_v2_episodes(all_rows, cutoff=cutoff)

    pending_rows = [r for r in all_rows if int(r["entry_policy_v2_learned"] or 0) == 0 and int(r["id"] or 0) in plan]
    pending_rows.sort(key=lambda r: (float(r["label_available_at"]), int(r["id"] or 0)))

    applied = 0
    for row in pending_rows:
        rid = int(row["id"] or 0)
        item = plan[rid]
        if not item.get("mark_learned"):
            continue
        conn.execute("SAVEPOINT mystic_entry_policy_v2")
        try:
            if item.get("fold"):
                folded = record_entry_policy_observation(
                    db_path,
                    symbol=str(row["symbol"]),
                    setup=str(row["setup"]),
                    regime=str(row["regime"]),
                    value=float(item["target"]),
                    specific_weight=float(item["specific_weight"]),
                    engine_weight=float(item["engine_weight"]),
                    version=ENTRY_POLICY_VALUE_V2,
                    now=float(item["label_available_at"]),
                    conn=conn,
                )
                if not folded:
                    conn.execute("ROLLBACK TO mystic_entry_policy_v2")
                    conn.execute("RELEASE mystic_entry_policy_v2")
                    continue
            marked = conn.execute(
                """
                UPDATE policy_episodes
                SET entry_policy_v2_learned=1,
                    entry_policy_v2_target=?,
                    entry_policy_v2_target_type=?,
                    entry_policy_v2_specific_weight=?,
                    entry_policy_v2_engine_weight=?,
                    entry_policy_v2_overlap_group=?
                WHERE id=? AND COALESCE(entry_policy_v2_learned, 0)=0
                """,
                (
                    item["target"],
                    item["target_type"],
                    float(item["specific_weight"]),
                    float(item["engine_weight"]),
                    str(item["overlap_group"]),
                    rid,
                ),
            )
            if marked.rowcount != 1:
                conn.execute("ROLLBACK TO mystic_entry_policy_v2")
                conn.execute("RELEASE mystic_entry_policy_v2")
                continue
            conn.execute("RELEASE mystic_entry_policy_v2")
            if item.get("fold"):
                applied += 1
        except Exception:
            conn.execute("ROLLBACK TO mystic_entry_policy_v2")
            conn.execute("RELEASE mystic_entry_policy_v2")
            logger.warning("ENTRY_POLICY_V2_LABEL_FAILED id=%s", row["id"], exc_info=True)
    return applied


def apply_resolved_entry_policy_labels(
    db_path: str,
    *,
    now: float | None = None,
    available_before: float | None = None,
) -> int:
    """Fold resolved exact-policy labels into ENTRY_POLICY_VALUE_V2 in causal label-time order.

    ``ENTRY_POLICY_VALUE_V1`` rows and ``entry_policy_learned`` are never modified.
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
        applied = _apply_resolved_v2_on_conn(conn, db_path, moment=moment, available_before=available_before)
        conn.commit()
        _invalidate()
    except sqlite3.Error:
        logger.warning("ENTRY_POLICY_APPLY_FAILED", exc_info=True)
        return 0
    finally:
        conn.close()
    v3_cutoff = float(available_before) - 1e-6 if available_before is not None else moment
    with sqlite3.connect(db_path, timeout=30.0) as _chk_conn:
        _has_pe = _chk_conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='policy_episodes'").fetchone()
    if _has_pe is not None and available_before is None:
        try:
            apply_resolved_entry_policy_v3_labels(db_path, now=v3_cutoff)
        except sqlite3.Error:
            logger.warning("ENTRY_POLICY_V3_APPLY_FAILED", exc_info=True)
    if applied:
        logger.info("ENTRY_POLICY_LABELS_APPLIED n=%s version=%s", applied, ENTRY_POLICY_VALUE_V2)
    return applied


def rebuild_entry_policy_value_v2(
    db_path: str,
    *,
    now: float | None = None,
    available_before: float | None = None,
) -> dict[str, Any]:
    """Deterministically rebuild ENTRY_POLICY_VALUE_V2 from all provable retained episodes.

    Executes atomically in one transaction and preserves ENTRY_POLICY_VALUE_V1
    rows and ``entry_policy_learned`` completely untouched.
    """
    moment = float(now if now is not None else time.time())
    conn = _connect(db_path)
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "policy_episodes" not in tables:
            return {"applied": 0, "version": ENTRY_POLICY_VALUE_V2, "v1_rows": 0, "v2_rows": 0}
        ensure_entry_policy_column(conn)
        v1_before = int(
            conn.execute(
                "SELECT COUNT(*) FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?",
                (DAY_ENGINE, ENTRY_POLICY_VALUE_V1),
            ).fetchone()[0]
        )
        conn.execute(
            "DELETE FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?",
            (DAY_ENGINE, ENTRY_POLICY_VALUE_V2),
        )
        conn.execute(
            """
            UPDATE policy_episodes
            SET entry_policy_v2_learned=0,
                entry_policy_v2_target=NULL,
                entry_policy_v2_target_type=NULL,
                entry_policy_v2_specific_weight=NULL,
                entry_policy_v2_engine_weight=NULL,
                entry_policy_v2_overlap_group=NULL
            WHERE capture_version=? AND engine_id=?
            """,
            (_CAPTURE, DAY_ENGINE),
        )
        applied = _apply_resolved_v2_on_conn(conn, db_path, moment=moment, available_before=available_before)
        v1_after = int(
            conn.execute(
                "SELECT COUNT(*) FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?",
                (DAY_ENGINE, ENTRY_POLICY_VALUE_V1),
            ).fetchone()[0]
        )
        v2_rows = int(
            conn.execute(
                "SELECT COUNT(*) FROM adaptive_metric_state WHERE engine_id=? AND economic_version=?",
                (DAY_ENGINE, ENTRY_POLICY_VALUE_V2),
            ).fetchone()[0]
        )
        assert v1_before == v1_after
        conn.commit()
        _invalidate()
        return {
            "applied": applied,
            "version": ENTRY_POLICY_VALUE_V2,
            "v1_rows": v1_after,
            "v2_rows": v2_rows,
        }
    finally:
        conn.close()


def _fold_memory_v1(state: dict[tuple[str, str, str], dict[str, Any]], episode: Mapping[str, Any]) -> None:
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


def _posterior_memory_v1(
    state: dict[tuple[str, str, str], dict[str, Any]],
    symbol: str,
    setup: str,
    regime: str,
    now: float,
) -> dict[str, Any]:
    if not state:
        return _cold(ENTRY_POLICY_VALUE_V1)
    rows = [
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
        for (sym, stp, reg), rec in state.items()
    ]
    view = _lattice(
        rows,  # type: ignore[arg-type]
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


def _fold_memory_v2(
    spec_state: dict[tuple[str, str, str], dict[str, Any]],
    eng_state: dict[tuple[str, str, str], dict[str, Any]],
    *,
    symbol: str,
    setup: str,
    regime: str,
    value: float,
    specific_weight: float,
    engine_weight: float,
    moment: float,
) -> None:
    key = (_norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower())
    if _is_forensic_key(DAY_ENGINE, *key):
        return
    if specific_weight > 0.0:
        prev_s = spec_state.get(key)
        if prev_s is None:
            n, mean, m2 = _moments(0.0, 0.0, 0.0, "", value, specific_weight, moment)
        else:
            n, mean, m2 = _moments(float(prev_s["n"]), float(prev_s["ewma"]), float(prev_s["m2"]), str(prev_s["updated_at"]), value, specific_weight, moment)
        spec_state[key] = {"n": n, "ewma": mean, "m2": m2, "updated_at": _stamp(moment)}
    if engine_weight > 0.0:
        prev_e = eng_state.get(key)
        if prev_e is None:
            n, mean, m2 = _moments(0.0, 0.0, 0.0, "", value, engine_weight, moment)
        else:
            n, mean, m2 = _moments(float(prev_e["n"]), float(prev_e["ewma"]), float(prev_e["m2"]), str(prev_e["updated_at"]), value, engine_weight, moment)
        eng_state[key] = {"n": n, "ewma": mean, "m2": m2, "updated_at": _stamp(moment)}


def _posterior_memory_v2(
    spec_state: dict[tuple[str, str, str], dict[str, Any]],
    eng_state: dict[tuple[str, str, str], dict[str, Any]],
    symbol: str,
    setup: str,
    regime: str,
    now: float,
) -> dict[str, Any]:
    if not spec_state and not eng_state:
        return _cold(ENTRY_POLICY_VALUE_V2)
    rows: list[dict[str, Any]] = []
    for (sym, stp, reg), rec in spec_state.items():
        rows.append(
            {
                "symbol": sym,
                "setup": stp,
                "regime": reg,
                "metric": ENTRY_POLICY_NET_V2,
                "n": rec["n"],
                "ewma": rec["ewma"],
                "m2": rec["m2"],
                "updated_at": rec["updated_at"],
            }
        )
    for (sym, stp, reg), rec in eng_state.items():
        rows.append(
            {
                "symbol": sym,
                "setup": stp,
                "regime": reg,
                "metric": ENTRY_POLICY_ENGINE_NET_V2,
                "n": rec["n"],
                "ewma": rec["ewma"],
                "m2": rec["m2"],
                "updated_at": rec["updated_at"],
            }
        )
    return _lattice_v2(rows, symbol=symbol, setup=setup, regime=regime, prior=ENTRY_POLICY_PRIOR, now=float(now))


def causal_replay(
    episodes: list[dict[str, Any]],
    *,
    version: str = ENTRY_POLICY_VALUE_V2,
    ohlcv_index: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Score every episode at its decision time from labels available strictly earlier."""
    if version == ENTRY_POLICY_VALUE_V3:
        return causal_replay_v3(episodes, ohlcv_index=ohlcv_index)
    if version == ENTRY_POLICY_VALUE_V1:
        labels_v1 = []
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
            labels_v1.append(episode)
        labels_v1.sort(key=lambda item: (float(item["label_available_at"]), int(item.get("id") or 0)))
        decisions_v1 = sorted(episodes, key=lambda item: (float(item.get("decided_at") or 0.0), int(item.get("id") or 0)))
        state_v1: dict[tuple[str, str, str], dict[str, Any]] = {}
        cursor_v1 = 0
        scored_v1: list[dict[str, Any]] = []
        for episode in decisions_v1:
            moment = float(episode.get("decided_at") or 0.0)
            while cursor_v1 < len(labels_v1) and float(labels_v1[cursor_v1]["label_available_at"]) < moment:
                _fold_memory_v1(state_v1, labels_v1[cursor_v1])
                cursor_v1 += 1
            view = _posterior_memory_v1(state_v1, str(episode.get("symbol") or ""), str(episode.get("setup") or ""), str(episode.get("regime") or ""), moment)
            scored_v1.append(
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
                    "version": ENTRY_POLICY_VALUE_V1,
                }
            )
        return scored_v1

    normalized: list[dict[str, Any]] = []
    for idx, ep in enumerate(episodes, start=1):
        copy_ep = dict(ep)
        copy_ep.setdefault("id", idx)
        copy_ep.setdefault("status", "CLOSED" if copy_ep.get("label_available_at") is not None else "OPEN")
        copy_ep.setdefault("capture_version", _CAPTURE)
        copy_ep.setdefault("engine_id", DAY_ENGINE)
        if "kind" not in copy_ep:
            copy_ep["kind"] = "REAL_POLICY_OUTCOME" if int(copy_ep.get("funded") or 0) == 1 else "COUNTERFACTUAL_POLICY_OUTCOME"
        if "train_eligible" not in copy_ep:
            copy_ep["train_eligible"] = 1 if copy_ep.get("eligible", True) else 0
        normalized.append(copy_ep)

    max_ts = max([0.0] + [float(e.get("decided_at") or 0.0) for e in normalized] + [float(e["label_available_at"]) for e in normalized if e.get("label_available_at") is not None])
    # Because plan_causal_v2_episodes is strictly invariant at every resolution tau <= max_ts,
    # planning at max_ts yields identical causal weights at each episode's label_available_at.
    plan = plan_causal_v2_episodes(normalized, cutoff=max_ts)

    resolvable: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    by_id = {int(e["id"]): e for e in normalized}
    for rid, item in plan.items():
        if item.get("fold"):
            resolvable.append((float(item["label_available_at"]), rid, by_id[rid], item))
    resolvable.sort(key=lambda t: (t[0], t[1]))

    decisions = sorted(normalized, key=lambda item: (float(item.get("decided_at") or 0.0), int(item["id"])))
    spec_state: dict[tuple[str, str, str], dict[str, Any]] = {}
    eng_state: dict[tuple[str, str, str], dict[str, Any]] = {}
    cursor = 0
    scored: list[dict[str, Any]] = []
    for episode in decisions:
        eval_ts = float(episode.get("eval_at") if episode.get("eval_at") is not None else (episode.get("decided_at") or 0.0))
        while cursor < len(resolvable) and resolvable[cursor][0] < eval_ts:
            tau, _rid, src_ep, planned = resolvable[cursor]
            _fold_memory_v2(
                spec_state,
                eng_state,
                symbol=str(src_ep.get("symbol") or ""),
                setup=str(src_ep.get("setup") or ""),
                regime=str(src_ep.get("regime") or ""),
                value=float(planned["target"]),
                specific_weight=float(planned["specific_weight"]),
                engine_weight=float(planned["engine_weight"]),
                moment=tau,
            )
            cursor += 1
        view = _posterior_memory_v2(
            spec_state,
            eng_state,
            str(episode.get("symbol") or ""),
            str(episode.get("setup") or ""),
            str(episode.get("regime") or ""),
            eval_ts,
        )
        ep_plan = plan.get(int(episode["id"]), {})
        scored.append(
            {
                "id": episode.get("id"),
                "symbol": episode.get("symbol"),
                "setup": episode.get("setup"),
                "regime": episode.get("regime"),
                "decided_at": float(episode.get("decided_at") or 0.0),
                "eval_at": eval_ts,
                "planning_net": episode.get("planning_net"),
                "target_net": ep_plan.get("target"),
                "target_type": ep_plan.get("target_type"),
                "specific_weight": ep_plan.get("specific_weight"),
                "engine_weight": ep_plan.get("engine_weight"),
                "overlap_group": ep_plan.get("overlap_group"),
                "prediction": float(view["mean"]),
                "n": float(view["n"]),
                "sd": float(view["sd"]),
                "engine_sd": float(view["engine_sd"]),
                "related_sd": float(view["related_sd"]),
                "key_sd": float(view["key_sd"]),
                "parent": float(view["parent"]),
                "levels": dict(view["levels"]),
                "level_weights": dict(view["level_weights"]),
                "labels_folded": cursor,
                "version": ENTRY_POLICY_VALUE_V2,
            }
        )
    return scored


def shrinkage_strength() -> float:
    """Prior sample count. Exposed so tests can show low counts shrink rather than reject."""
    return float(PRIOR_STRENGTH)
