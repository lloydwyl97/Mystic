"""Online adaptive state for DAY_V2 and SCALP_V2.

One framework, two engines. A row is keyed by engine, symbol, setup and regime,
so DAY observations never update SCALP and SCALP observations never update DAY.
Each estimate is a shrunk mean: ``(prior_strength * prior + n * ewma) / (prior_strength + n)``.
With no observations the estimate is the prior and confidence is 0. There is no
minimum-trade gate and no profit-factor gate. Legacy strategy versions are
refused at write time.

Realized closes update ``trade_*`` metrics. Candidate markouts update
``markout_*`` metrics. Decisions blend the two. Both are consumed by live
ranking, sizing and exit calibration.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable
from typing import Any

from backend.services.strategy_version import ADAPTIVE_STATE_VERSION, engine_versions

PRIOR_STRENGTH = 8.0
EWMA_ALPHA = 0.25
MARKOUT_WEIGHT = 0.35

DAY_ENGINE = "DAY_V2"
SCALP_ENGINE = "SCALP_V2"

DAY_HORIZONS_MIN = (15, 30, 60, 120, 240, 360)
SCALP_HORIZONS_SEC = (30, 60, 120, 300, 600, 1200)

_PRIORS: dict[str, dict[str, float]] = {
    DAY_ENGINE: {
        "trade_mfe": 0.012,
        "trade_mae": 0.006,
        "trade_time_to_mfe_min": 90.0,
        "trade_continuation": 0.45,
        "markout_forward": 0.012,
        "markout_mae": 0.006,
    },
    SCALP_ENGINE: {
        "trade_mfe": 0.0025,
        "trade_mae": 0.0015,
        "trade_net": 0.0015,
        "trade_time_to_mfe_min": 8.0,
        "markout_forward": 0.0015,
        "markout_mae": 0.0015,
    },
}

SIZE_BOUNDS = {"DAY_V2": (0.55, 1.35), "SCALP_V2": (0.50, 1.25)}
OBJECTIVE_ATR_BOUNDS = (0.75, 1.35)
STRUCTURAL_EMPHASIS_BOUNDS = (0.85, 1.25)
ACTIVATION_BOUNDS = (0.80, 1.25)
TRAIL_BOUNDS = (0.80, 1.20)
TIGHTEN_BOUNDS = (0.75, 1.15)
SCALP_TARGET_BOUNDS = (0.0015, 0.006)
SCALP_HOLD_FLOOR_MIN = 4.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _prior(engine: str, metric: str) -> float:
    return float(_PRIORS.get(engine, {}).get(metric, 0.0))


def current_strategy_version(engine: str) -> str:
    versions = engine_versions(engine)
    return str(versions["strategy_version"]) if versions else ""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adaptive_metric_state (
            engine_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            setup TEXT NOT NULL,
            regime TEXT NOT NULL,
            metric TEXT NOT NULL,
            n REAL NOT NULL,
            ewma REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (engine_id, symbol, setup, regime, metric)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adaptive_candidate_markouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            engine_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            setup TEXT NOT NULL,
            regime TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            signaled INTEGER NOT NULL,
            ref_price REAL NOT NULL,
            roundtrip_cost REAL NOT NULL,
            evaluated_at REAL NOT NULL,
            markouts_json TEXT NOT NULL DEFAULT '{}',
            learned INTEGER NOT NULL DEFAULT 0,
            resolved INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return conn


def observe(
    db_path: str,
    *,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    metric: str,
    value: float,
    strategy_version: str,
) -> bool:
    """Fold one current-version observation into the key. Returns False for any other version."""
    engine_id = str(engine or "").upper()
    if engine_id not in _PRIORS or metric not in _PRIORS[engine_id]:
        return False
    if str(strategy_version or "") != current_strategy_version(engine_id):
        return False
    if value is None:
        return False
    key = (engine_id, str(symbol or "").upper(), str(setup or "").upper(), str(regime or "").lower(), metric)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT n, ewma FROM adaptive_metric_state WHERE engine_id=? AND symbol=? AND setup=? AND regime=? AND metric=?",
            key,
        ).fetchone()
        if row is None or float(row["n"]) <= 0:
            n, ewma = 1.0, float(value)
        else:
            n = float(row["n"]) + 1.0
            ewma = (1.0 - EWMA_ALPHA) * float(row["ewma"]) + EWMA_ALPHA * float(value)
        conn.execute(
            """
            INSERT INTO adaptive_metric_state (engine_id, symbol, setup, regime, metric, n, ewma, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_id, symbol, setup, regime, metric) DO UPDATE SET
                n=excluded.n, ewma=excluded.ewma, updated_at=excluded.updated_at
            """,
            (*key, n, ewma, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        conn.commit()
    return True


def estimate(db_path: str, engine: str, symbol: str, setup: str, regime: str, metric: str) -> dict[str, Any]:
    """Shrunk mean for one key. Parent is the same engine+setup across symbols, then the prior."""
    engine_id = str(engine or "").upper()
    prior = _prior(engine_id, metric)
    empty = {
        "mean": prior,
        "n": 0.0,
        "prior": prior,
        "confidence": 0.0,
        "version": ADAPTIVE_STATE_VERSION,
    }
    if engine_id not in _PRIORS:
        return empty
    try:
        with _connect(db_path) as conn:
            specific = conn.execute(
                "SELECT n, ewma FROM adaptive_metric_state WHERE engine_id=? AND symbol=? AND setup=? AND regime=? AND metric=?",
                (engine_id, str(symbol or "").upper(), str(setup or "").upper(), str(regime or "").lower(), metric),
            ).fetchone()
            pooled = conn.execute(
                "SELECT COALESCE(SUM(n), 0), COALESCE(SUM(n * ewma), 0) FROM adaptive_metric_state WHERE engine_id=? AND setup=? AND metric=?",
                (engine_id, str(setup or "").upper(), metric),
            ).fetchone()
    except sqlite3.Error:
        return empty
    spec_n = float(specific["n"]) if specific else 0.0
    spec_mean = float(specific["ewma"]) if specific else prior
    pool_n = float(pooled[0] or 0.0)
    pool_sum = float(pooled[1] or 0.0)
    sib_n = max(0.0, pool_n - spec_n)
    sib_sum = pool_sum - spec_n * spec_mean
    parent = (PRIOR_STRENGTH * prior + sib_sum) / (PRIOR_STRENGTH + sib_n)
    mean = (PRIOR_STRENGTH * parent + spec_n * spec_mean) / (PRIOR_STRENGTH + spec_n)
    return {
        "mean": mean,
        "n": spec_n,
        "prior": prior,
        "parent": parent,
        "confidence": spec_n / (PRIOR_STRENGTH + spec_n),
        "version": ADAPTIVE_STATE_VERSION,
    }


def _blend(parts: list[tuple[float, float]], prior: float) -> tuple[float, float]:
    weighted = [(mean, weight) for mean, weight in parts if weight > 0]
    if not weighted:
        return prior, 0.0
    weight = sum(item[1] for item in weighted)
    mean = (PRIOR_STRENGTH * prior + sum(m * w for m, w in weighted)) / (PRIOR_STRENGTH + weight)
    return mean, weight


def _tilt(mean: float, prior: float) -> float:
    if prior == 0:
        return 0.0
    return math.tanh((mean - prior) / abs(prior))


def day_decision(db_path: str, symbol: str, setup: str, regime: str) -> dict[str, Any]:
    """What the next DAY candidate reads. Ranking, size, objective and runner only."""
    mfe = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_mfe")
    mae = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_mae")
    timing = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_time_to_mfe_min")
    continuation = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_continuation")
    forward = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "markout_forward")
    expected, n_eff = _blend(
        [(mfe["mean"], mfe["n"]), (forward["mean"], forward["n"] * MARKOUT_WEIGHT)],
        mfe["prior"],
    )
    lo, hi = SIZE_BOUNDS[DAY_ENGINE]
    confidence = n_eff / (PRIOR_STRENGTH + n_eff)
    return {
        "adaptive_state_version": ADAPTIVE_STATE_VERSION,
        "engine_id": DAY_ENGINE,
        "symbol": str(symbol or "").upper(),
        "setup": str(setup or "").upper(),
        "regime": str(regime or "").lower(),
        "expected_move": expected,
        "expected_move_prior": mfe["prior"],
        "confidence": confidence,
        "uncertainty": mfe["prior"] * (1.0 - confidence),
        "mfe": mfe["mean"],
        "mae": mae["mean"],
        "time_to_mfe_min": timing["mean"],
        "continuation": continuation["mean"],
        "size_mult": _clamp(1.0 + 0.30 * _tilt(expected, mfe["prior"]), lo, hi),
        "objective_atr_mult": _clamp(1.0 + 0.35 * _tilt(mfe["mean"], mfe["prior"]), *OBJECTIVE_ATR_BOUNDS),
        "structural_emphasis": _clamp(1.0 + 0.15 * _tilt(expected, mfe["prior"]), *STRUCTURAL_EMPHASIS_BOUNDS),
        "runner_activation_mult": _clamp(timing["mean"] / timing["prior"], *ACTIVATION_BOUNDS),
        "runner_trail_mult": _clamp(mae["mean"] / mae["prior"], *TRAIL_BOUNDS),
        "runner_tighten_mult": _clamp(0.75 + 0.40 * (continuation["mean"] / continuation["prior"]), *TIGHTEN_BOUNDS),
        "risk_estimate": mae["mean"],
        "n_mfe": mfe["n"],
        "n_forward": forward["n"],
    }


def scalp_decision(db_path: str, symbol: str, setup: str, regime: str) -> dict[str, Any]:
    """What the next SCALP candidate reads. Ranking, size, target and hold only."""
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_TIME_STOP_MIN

    net = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_net")
    mfe = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_mfe")
    mae = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_mae")
    timing = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_time_to_mfe_min")
    forward = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "markout_forward")
    edge, n_eff = _blend([(net["mean"], net["n"]), (forward["mean"], forward["n"] * MARKOUT_WEIGHT)], net["prior"])
    learned_mfe, _ = _blend([(mfe["mean"], mfe["n"]), (max(0.0, forward["mean"]), forward["n"] * MARKOUT_WEIGHT)], mfe["prior"])
    lo, hi = SIZE_BOUNDS[SCALP_ENGINE]
    hard_hold = float(SCALP_V2_TIME_STOP_MIN)
    confidence = n_eff / (PRIOR_STRENGTH + n_eff)
    return {
        "adaptive_state_version": ADAPTIVE_STATE_VERSION,
        "engine_id": SCALP_ENGINE,
        "symbol": str(symbol or "").upper(),
        "setup": str(setup or "").upper(),
        "regime": str(regime or "").lower(),
        "expected_edge": edge,
        "edge_prior": net["prior"],
        "confidence": confidence,
        "uncertainty": abs(net["prior"]) * (1.0 - confidence),
        "mfe": learned_mfe,
        "mae": mae["mean"],
        "target_pct": _clamp(learned_mfe, *SCALP_TARGET_BOUNDS),
        "hold_min": _clamp(timing["mean"], SCALP_HOLD_FLOOR_MIN, hard_hold),
        "hold_hard_max_min": hard_hold,
        "size_mult": _clamp(1.0 + 0.40 * _tilt(edge, net["prior"]), lo, hi),
        "risk_estimate": mae["mean"],
        "time_to_mfe_min": timing["mean"],
        "n_net": net["n"],
        "n_forward": forward["n"],
    }


def learn_from_close(
    db_path: str,
    *,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    strategy_version: str,
    net_pct: float | None,
    mfe_pct: float | None,
    mae_pct: float | None,
    hold_min: float | None,
    continuation: float | None,
    version_current: bool,
    is_dust: bool,
) -> bool:
    """Realized-trade update. Dust and non-current versions do not move state."""
    if is_dust or not version_current:
        return False
    engine_id = str(engine or "").upper()
    wrote = False
    if engine_id == DAY_ENGINE:
        pairs = (
            ("trade_mfe", mfe_pct),
            ("trade_mae", abs(mae_pct) if mae_pct is not None else None),
            ("trade_time_to_mfe_min", hold_min),
            ("trade_continuation", continuation),
        )
    elif engine_id == SCALP_ENGINE:
        pairs = (
            ("trade_net", net_pct),
            ("trade_mfe", mfe_pct),
            ("trade_mae", abs(mae_pct) if mae_pct is not None else None),
            ("trade_time_to_mfe_min", hold_min),
        )
    else:
        return False
    for metric, value in pairs:
        if value is None:
            continue
        wrote = observe(db_path, engine=engine_id, symbol=symbol, setup=setup, regime=regime, metric=metric, value=float(value), strategy_version=strategy_version) or wrote
    return wrote


def record_candidate(
    db_path: str,
    *,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    ref_price: float,
    roundtrip_cost: float,
    signaled: bool,
    evaluated_at: float | None = None,
) -> None:
    engine_id = str(engine or "").upper()
    version = current_strategy_version(engine_id)
    if not version or float(ref_price or 0) <= 0 or not str(setup or "").strip():
        return
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO adaptive_candidate_markouts (
                engine_id, symbol, setup, regime, strategy_version, signaled,
                ref_price, roundtrip_cost, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                engine_id,
                str(symbol or "").upper(),
                str(setup or "").upper(),
                str(regime or "").lower(),
                version,
                1 if signaled else 0,
                float(ref_price),
                float(roundtrip_cost or 0),
                float(evaluated_at or time.time()),
            ),
        )
        conn.commit()


def resolve_markouts(db_path: str, quote: Callable[[str, float], float | None], *, now: float | None = None) -> int:
    """Fill due forward marks and fold them into markout metrics. Returns rows newly learned."""
    moment = float(now if now is not None else time.time())
    learned = 0
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        return 0
    try:
        rows = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE resolved=0 ORDER BY id ASC LIMIT 200").fetchall()
        for row in rows:
            engine_id = str(row["engine_id"])
            horizons = DAY_HORIZONS_MIN if engine_id == DAY_ENGINE else SCALP_HORIZONS_SEC
            unit = 60.0 if engine_id == DAY_ENGINE else 1.0
            grace = 900.0 if engine_id == DAY_ENGINE else 60.0
            marks = json.loads(row["markouts_json"] or "{}")
            done = True
            for horizon in horizons:
                key = str(horizon)
                if key in marks:
                    continue
                due = float(row["evaluated_at"]) + horizon * unit
                if moment < due:
                    done = False
                    continue
                price = quote(str(row["symbol"]), due)
                if price is None and moment < due + grace:
                    done = False
                    continue
                if price is None or float(row["ref_price"]) <= 0:
                    marks[key] = None
                else:
                    marks[key] = (float(price) - float(row["ref_price"])) / float(row["ref_price"]) - float(row["roundtrip_cost"] or 0)
            values = [float(v) for v in marks.values() if v is not None]
            row_learned = int(row["learned"] or 0)
            if done and values and not row_learned and str(row["strategy_version"]) == current_strategy_version(engine_id):
                observe(
                    db_path,
                    engine=engine_id,
                    symbol=row["symbol"],
                    setup=row["setup"],
                    regime=row["regime"],
                    metric="markout_forward",
                    value=max(values),
                    strategy_version=str(row["strategy_version"]),
                )
                observe(
                    db_path,
                    engine=engine_id,
                    symbol=row["symbol"],
                    setup=row["setup"],
                    regime=row["regime"],
                    metric="markout_mae",
                    value=max(0.0, -min(values)),
                    strategy_version=str(row["strategy_version"]),
                )
                row_learned = 1
                learned += 1
            conn.execute(
                "UPDATE adaptive_candidate_markouts SET markouts_json=?, learned=?, resolved=? WHERE id=?",
                (json.dumps(marks), row_learned, 1 if done else 0, row["id"]),
            )
        conn.commit()
    finally:
        conn.close()
    return learned


def ohlcv_quote(db_path: str, symbol: str, epoch: float) -> float | None:
    """Close of the feature_ohlcv bar that contains ``epoch``. 1m first, then 15m."""
    from backend.services.candle_contract import _parse_ts

    raw = str(symbol or "").upper().replace("-", "").replace("/", "")
    variants = [raw, raw.replace("USDT", "-USDT"), raw.replace("USDT", "/USDT")]
    try:
        conn = sqlite3.connect(db_path, timeout=5)
    except sqlite3.Error:
        return None
    try:
        for interval, sec in (("1m", 60), ("15m", 900)):
            for variant in variants:
                try:
                    rows = conn.execute(
                        "SELECT ts, close FROM feature_ohlcv WHERE symbol=? AND interval=? ORDER BY ts DESC LIMIT 400",
                        (variant, interval),
                    ).fetchall()
                except sqlite3.Error:
                    return None
                for ts, close in rows:
                    opened = _parse_ts(ts)
                    if opened is None or close is None:
                        continue
                    if opened <= epoch < opened + sec + 2:
                        return float(close)
    finally:
        conn.close()
    return None


def persist_trade_adaptive(conn: sqlite3.Connection, trade_id: str, decision: dict[str, Any] | None) -> None:
    if not trade_id or not decision:
        return
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    if "adaptive_decision_json" not in cols:
        conn.execute("ALTER TABLE paper_trades ADD COLUMN adaptive_decision_json TEXT DEFAULT ''")
    conn.execute(
        "UPDATE paper_trades SET adaptive_decision_json=? WHERE trade_id=?",
        (json.dumps(decision, separators=(",", ":"), default=str), trade_id),
    )


__all__ = [
    "ADAPTIVE_STATE_VERSION",
    "DAY_HORIZONS_MIN",
    "SCALP_HORIZONS_SEC",
    "day_decision",
    "estimate",
    "learn_from_close",
    "observe",
    "ohlcv_quote",
    "persist_trade_adaptive",
    "record_candidate",
    "resolve_markouts",
    "scalp_decision",
]
