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

import contextlib
import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from backend.services.scalp_v2.raw_move_source import is_directional, normalize_raw_move_source
from backend.services.strategy_version import ADAPTIVE_STATE_VERSION, engine_versions

PRIOR_STRENGTH = 8.0
EWMA_ALPHA = 0.25
MARKOUT_WEIGHT = 0.35

DAY_ENGINE = "DAY_V2"
SCALP_ENGINE = "SCALP_V2"

# Pre-stamp history. These keys stay in the table for forensics and are never
# copied onto current setup names. The current adaptive API does not list them
# as live learning.
FORENSIC_ADAPTIVE_KEYS = frozenset(
    {
        (SCALP_ENGINE, "XRPUSDT", "RANGE", ""),
        (SCALP_ENGINE, "XRPUSDT", "VWAP", ""),
    }
)

DAY_HORIZONS_MIN = (15, 30, 60, 120, 240, 360)
SCALP_HORIZONS_SEC = (30, 60, 120, 300, 600, 1200)

_PRIORS: dict[str, dict[str, float]] = {
    DAY_ENGINE: {
        "trade_mfe": 0.012,
        "trade_mae": 0.006,
        "trade_time_to_mfe_min": 90.0,
        "trade_continuation": 0.45,
        # Net quantities (realized trade net after costs, cost-adjusted forward
        # markout) start neutral: no edge is assumed before evidence. The MFE
        # prior is a path maximum, not a net return, and must not seed them.
        "trade_net": 0.0,
        "markout_forward": 0.0,
        "markout_mae": 0.006,
    },
    SCALP_ENGINE: {
        "trade_mfe": 0.0025,
        "trade_mae": 0.0015,
        "trade_net": 0.0015,
        "trade_time_to_mfe_min": 8.0,
        "markout_forward": 0.0015,
        # Gross adverse price excursion of the candidate path over its committed
        # horizon (SCALP). This is the live risk estimate.
        "markout_mae": 0.0015,
        # Realized net markout minus the decision-time base executable edge
        # (raw expected move - cost), kept per raw-move source because an ATR
        # estimate and a strategy structural claim carry different biases.
        # Neutral cold prior: no evidence, no adjustment.
        "edge_residual": 0.0,
        "edge_residual_strategy": 0.0,
        # Decayed moments of (claim base edge, residual) for the claim capture slope.
        "claim_base_edge": 0.0,
        "claim_base_edge_sq": 0.0,
        "claim_residual": 0.0,
        "claim_base_edge_x_residual": 0.0,
    },
}

CLAIM_MOMENT_METRICS = ("claim_base_edge", "claim_base_edge_sq", "claim_residual", "claim_base_edge_x_residual")

# DAY candidate markout states. NEAR_QUALIFIED rows (one entry condition short)
# are resolved for evidence and never folded into decision state.
CANDIDATE_QUALIFIED = "QUALIFIED"
CANDIDATE_NEAR_QUALIFIED = "NEAR_QUALIFIED"

# Bound on the learned residual added to a SCALP candidate's base executable
# edge. Equal to the raw expected-move cap, so evidence can cancel a full claim.
SCALP_RESIDUAL_MAX = 0.006


def residual_metric(raw_move_source: str | None) -> str:
    """Residual key per raw-move source. Only the strategy-claim residual is live;
    ``edge_residual`` (ATR-estimate rows) is forensic and never read by eligibility."""
    return "edge_residual_strategy" if is_directional(raw_move_source) else "edge_residual"


# Metrics folded as a decayed running mean (weight 1/n) instead of the fast EWMA.
# Causal calibration on current-version SCALP markouts: a 0.25 EWMA residual
# tracks the last few labels and produced 4x more positive predictions with no
# better realization; the running mean converges to the key's actual bias.
MEAN_FORM_METRICS = frozenset({"edge_residual", "edge_residual_strategy", *CLAIM_MOMENT_METRICS})

SIZE_BOUNDS = {"DAY_V2": (0.55, 1.35), "SCALP_V2": (0.50, 1.25)}
OBJECTIVE_ATR_BOUNDS = (0.75, 1.35)
STRUCTURAL_EMPHASIS_BOUNDS = (0.85, 1.25)
ACTIVATION_BOUNDS = (0.80, 1.25)
TRAIL_BOUNDS = (0.80, 1.20)
TIGHTEN_BOUNDS = (0.75, 1.15)
SCALP_TARGET_BOUNDS = (0.0015, 0.006)
SCALP_HOLD_FLOOR_MIN = 4.0

# Half-life for observation weight. Older evidence loses effective sample count
# so the estimator tracks current market behaviour instead of averaging over a
# stale regime forever. It never zeroes a key (the new observation always counts
# for 1), so there is no min-trade gate and no forgetting to a hard stop.
ADAPTIVE_HALF_LIFE_DAYS = float(os.getenv("ADAPTIVE_HALF_LIFE_DAYS", "14") or "14")

# SCALP microstructure edge model. A single inspectable online linear model
# (normalised LMS) that learns how the current book/flow shifts the candidate's
# edge residual (realized net markout minus base executable edge). Its output is
# a bounded, zero-centred residual on the base edge; it can lower a candidate
# but never lift one whose base edge plus learned residual is not positive.
SCALP_MICRO_FEATURES = (
    "microprice_pressure",
    "obi_l5",
    "ofi_5s",
    "agg_flow_imbalance_5s",
    "adverse_selection_score",
    "spread_pct",
)
MICRO_MODEL_LR = 0.02  # learning rate for the online weight update
MICRO_MODEL_W_MAX = 0.5  # per-feature weight clamp (keeps any one feature bounded)
MICRO_MODEL_TILT_MAX = 0.0015  # max absolute edge tilt the model can add (15 bps)
MICRO_MODEL_STD_ALPHA = 0.05  # EWMA rate for feature standardisation stats
MICRO_MODEL_CONF_K = 20.0  # shrink the tilt by n/(n+K) so a cold model barely moves

# Learned negative-expectancy flag. Telemetry on both engines, never a live veto:
# SCALP's negative-edge gate is the canonical executable edge; DAY carries its
# learned net expectancy into bounded size and rank. There is no sample-count
# floor. The shrunk posterior is the uncertainty weighting: thin evidence moves
# it a little, accumulating evidence moves it more, and it recovers the same way.
# Kill-switch: ADAPTIVE_ABSTENTION_ENABLED=false clears the flag.
ABSTAIN_ENABLED = (os.getenv("ADAPTIVE_ABSTENTION_ENABLED", "true") or "true").strip().lower() in ("1", "true", "yes", "on")
ABSTAIN_EDGE_MARGIN = float(os.getenv("ADAPTIVE_ABSTENTION_EDGE_MARGIN", "0.0005") or "0.0005")


def _abstain(net_edge: float) -> tuple[bool, str]:
    """Flag a learned net expectancy at or below the margin. Returns (flag, reason)."""
    if not ABSTAIN_ENABLED:
        return False, ""
    if net_edge <= -ABSTAIN_EDGE_MARGIN:
        return True, f"LEARNED_NEGATIVE_EDGE net={net_edge:.5f}"
    return False, ""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _parse_iso(ts: str) -> float | None:
    try:
        return datetime.strptime(str(ts), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        return None


def _decay_factor(updated_at: str, now_epoch: float) -> float:
    """Weight retained for a key's prior sample count, by age. 1.0 if age unknown."""
    t0 = _parse_iso(updated_at)
    half_life = ADAPTIVE_HALF_LIFE_DAYS * 86400.0
    if t0 is None or half_life <= 0:
        return 1.0
    elapsed = max(0.0, float(now_epoch) - t0)
    return float(0.5 ** (elapsed / half_life))


def _is_forensic_key(engine_id: str, symbol: str, setup: str, regime: str) -> bool:
    return (str(engine_id), _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower()) in FORENSIC_ADAPTIVE_KEYS


def _norm_symbol(symbol: str) -> str:
    """Single symbol key form. Closes pass 'XRP/USDT'; ranking passes 'XRPUSDT'.

    Both must land on the same adaptive row, so the separator is always stripped
    before the key is built. Without this the realized-trade write and the next
    candidate's read never meet.
    """
    return str(symbol or "").upper().replace("-", "").replace("/", "")


def _prior(engine: str, metric: str) -> float:
    return float(_PRIORS.get(engine, {}).get(metric, 0.0))


def current_strategy_version(engine: str) -> str:
    versions = engine_versions(engine)
    return str(versions["strategy_version"]) if versions else ""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
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
            features_json TEXT NOT NULL DEFAULT '{}',
            learned INTEGER NOT NULL DEFAULT 0,
            resolved INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # Migration: add features_json to pre-existing markout tables.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(adaptive_candidate_markouts)").fetchall()}
    if "features_json" not in cols:
        conn.execute("ALTER TABLE adaptive_candidate_markouts ADD COLUMN features_json TEXT NOT NULL DEFAULT '{}'")
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(adaptive_candidate_markouts)").fetchall()}
    if "label_horizon" not in cols:
        conn.execute("ALTER TABLE adaptive_candidate_markouts ADD COLUMN label_horizon REAL NOT NULL DEFAULT 0")
    if "raw_expected_move" not in cols:
        conn.execute("ALTER TABLE adaptive_candidate_markouts ADD COLUMN raw_expected_move REAL")
    if "raw_move_source" not in cols:
        conn.execute("ALTER TABLE adaptive_candidate_markouts ADD COLUMN raw_move_source TEXT")
    if "candidate_state" not in cols:
        conn.execute("ALTER TABLE adaptive_candidate_markouts ADD COLUMN candidate_state TEXT NOT NULL DEFAULT ''")
    # resolve_markouts runs every SCALP cycle; without these, its key repair and
    # unresolved scan are full table scans that grow with the markout history.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_key ON adaptive_candidate_markouts(engine_id, symbol, setup, regime, learned)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_unresolved ON adaptive_candidate_markouts(resolved, id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS adaptive_linear_model (
            engine_id TEXT NOT NULL,
            model TEXT NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (engine_id, model)
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
    now: float | None = None,
) -> bool:
    """Fold one current-version observation into the key. Returns False for any other version.

    ``now`` is the observation time (defaults to wall clock); a chronological
    rebuild passes each label's own time so decay matches live learning.
    """
    engine_id = str(engine or "").upper()
    if engine_id not in _PRIORS or metric not in _PRIORS[engine_id]:
        return False
    if str(strategy_version or "") != current_strategy_version(engine_id):
        return False
    if value is None or not math.isfinite(float(value)):
        return False
    key = (engine_id, _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower(), metric)
    moment = float(now if now is not None else time.time())
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT n, ewma, updated_at FROM adaptive_metric_state WHERE engine_id=? AND symbol=? AND setup=? AND regime=? AND metric=?",
            key,
        ).fetchone()
        if row is None or float(row["n"]) <= 0:
            n, ewma = 1.0, float(value)
        else:
            # Age out the prior sample count so stale evidence stops dominating,
            # then fold in the new observation. The new point always counts for 1.
            decay = _decay_factor(str(row["updated_at"] or ""), moment)
            n = float(row["n"]) * decay + 1.0
            alpha = 1.0 / n if metric in MEAN_FORM_METRICS else EWMA_ALPHA
            ewma = (1.0 - alpha) * float(row["ewma"]) + alpha * float(value)
        conn.execute(
            """
            INSERT INTO adaptive_metric_state (engine_id, symbol, setup, regime, metric, n, ewma, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_id, symbol, setup, regime, metric) DO UPDATE SET
                n=excluded.n, ewma=excluded.ewma, updated_at=excluded.updated_at
            """,
            (*key, n, ewma, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))),
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
                (engine_id, _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower(), metric),
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


# DAY net expectancy reads two separately learned net quantities: realized trade
# net (production lifecycle, costs once) and the counterfactual forward markout
# of qualified candidates, down-weighted by MARKOUT_WEIGHT. They stay separate
# metrics; only this read combines them.
DAY_NET_PARTS: tuple[tuple[str, float], ...] = (("trade_net", 1.0), ("markout_forward", MARKOUT_WEIGHT))


def day_net_expectancy(db_path: str, symbol: str, setup: str, regime: str) -> dict[str, Any]:
    """Learned DAY net expectancy for one key: the key shrunk toward the same
    setup's other keys, which shrink toward the neutral prior (as ``estimate``).
    No sample-count floor: one observation moves the posterior by its weight."""
    sym = _norm_symbol(symbol)
    stp = str(setup or "").upper()
    reg = str(regime or "").lower()
    weights = dict(DAY_NET_PARTS)
    prior = _prior(DAY_ENGINE, "trade_net")
    spec_n = spec_sum = sib_n = sib_sum = 0.0
    n_metric = dict.fromkeys(weights, 0.0)
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT symbol, regime, metric, n, ewma FROM adaptive_metric_state WHERE engine_id=? AND setup=? AND metric IN ({','.join('?' * len(weights))})",
                (DAY_ENGINE, stp, *weights),
            ).fetchall()
    except sqlite3.Error:
        rows = []
    for row in rows:
        n = float(row["n"] or 0.0)
        w = weights[row["metric"]] * n
        if _norm_symbol(row["symbol"]) == sym and str(row["regime"] or "").lower() == reg:
            spec_n += w
            spec_sum += w * float(row["ewma"])
            n_metric[row["metric"]] += n
        else:
            sib_n += w
            sib_sum += w * float(row["ewma"])
    parent = (PRIOR_STRENGTH * prior + sib_sum) / (PRIOR_STRENGTH + sib_n)
    return {
        "mean": (PRIOR_STRENGTH * parent + spec_sum) / (PRIOR_STRENGTH + spec_n),
        "parent": parent,
        "prior": prior,
        "n_trade": n_metric["trade_net"],
        "n_forward": n_metric["markout_forward"],
        "weight": spec_n,
        "pooled_weight": sib_n,
        "confidence": spec_n / (PRIOR_STRENGTH + spec_n),
    }


def day_decision(db_path: str, symbol: str, setup: str, regime: str) -> dict[str, Any]:
    """What the next DAY candidate reads. Ranking, size, objective and runner only.

    Learned net expectancy sets the bounded size tilt and joins the rank score;
    learned move potential (MFE) sets the objective. Neither removes a candidate.
    """
    from backend.config.trading_economics import canonical_roundtrip_cost_pct

    mfe = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_mfe")
    mae = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_mae")
    timing = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_time_to_mfe_min")
    continuation = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_continuation")
    forward = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "markout_forward")
    net = day_net_expectancy(db_path, symbol, setup, regime)
    n_eff = mfe["n"] + forward["n"] * MARKOUT_WEIGHT
    lo, hi = SIZE_BOUNDS[DAY_ENGINE]
    confidence = n_eff / (PRIOR_STRENGTH + n_eff)
    # Expectancy per unit of adverse risk, as SCALP sizes its final edge.
    risk = max(0.0, mae["mean"]) + canonical_roundtrip_cost_pct()
    net_tilt = math.tanh(net["mean"] / risk) if risk > 0 else 0.0
    abstain_flag, abstain_reason = _abstain(net["mean"])
    return {
        "adaptive_state_version": ADAPTIVE_STATE_VERSION,
        "engine_id": DAY_ENGINE,
        "symbol": str(symbol or "").upper(),
        "setup": str(setup or "").upper(),
        "regime": str(regime or "").lower(),
        "expected_move": mfe["mean"],
        "expected_move_prior": mfe["prior"],
        "expected_net": net["mean"],
        "expected_net_parent": net["parent"],
        "net_confidence": net["confidence"],
        "confidence": confidence,
        "abstain": abstain_flag,
        "abstain_reason": abstain_reason,
        "abstain_net_edge": net["mean"],
        "abstain_confidence": net["confidence"],
        "abstain_live_veto": False,
        "uncertainty": mfe["prior"] * (1.0 - confidence),
        "mfe": mfe["mean"],
        "mae": mae["mean"],
        "time_to_mfe_min": timing["mean"],
        "continuation": continuation["mean"],
        "size_mult": _clamp(1.0 + 0.30 * net_tilt, lo, hi),
        "objective_atr_mult": _clamp(1.0 + 0.35 * _tilt(mfe["mean"], mfe["prior"]), *OBJECTIVE_ATR_BOUNDS),
        "structural_emphasis": _clamp(1.0 + 0.15 * _tilt(mfe["mean"], mfe["prior"]), *STRUCTURAL_EMPHASIS_BOUNDS),
        "runner_activation_mult": _clamp(timing["mean"] / timing["prior"], *ACTIVATION_BOUNDS),
        "runner_trail_mult": _clamp(mae["mean"] / mae["prior"], *TRAIL_BOUNDS),
        "runner_tighten_mult": _clamp(0.75 + 0.40 * (continuation["mean"] / continuation["prior"]), *TIGHTEN_BOUNDS),
        "risk_estimate": mae["mean"],
        "n_mfe": mfe["n"],
        "n_forward": forward["n"],
        "n_trade_net": net["n_trade"],
    }


def _micro_features(raw: dict | None) -> dict[str, float]:
    """Pull the fixed SCALP microstructure vector out of a raw feature dict."""
    raw = raw if isinstance(raw, dict) else {}
    out: dict[str, float] = {}
    for f in SCALP_MICRO_FEATURES:
        try:
            out[f] = float(raw.get(f) or 0.0)
        except (TypeError, ValueError):
            out[f] = 0.0
    return out


def _load_linear(conn: sqlite3.Connection, engine_id: str, model: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT payload FROM adaptive_linear_model WHERE engine_id=? AND model=?",
        (engine_id, model),
    ).fetchone()
    if row and row["payload"]:
        try:
            st = json.loads(row["payload"])
            if isinstance(st, dict):
                st.setdefault("n", 0)
                st.setdefault("bias", 0.0)
                st.setdefault("w", {})
                st.setdefault("mu", {})
                st.setdefault("s2", {})
                return st
        except (ValueError, TypeError):
            pass
    return {"n": 0, "bias": 0.0, "w": {}, "mu": {}, "s2": {}}


def _standardize(st: dict[str, Any], feats: dict[str, float]) -> dict[str, float]:
    z: dict[str, float] = {}
    for f, x in feats.items():
        mu = float(st["mu"].get(f, 0.0))
        s2 = float(st["s2"].get(f, 1.0))
        z[f] = (x - mu) / math.sqrt(s2 + 1e-9)
    return z


def update_linear_model(db_path: str, engine: str, model: str, features: dict | None, target: float) -> None:
    """One online (normalised-LMS) step. Standardisation stats adapt via EWMA;
    weights are clamped. Trained on the same cost-adjusted markout target."""
    raw = features if isinstance(features, dict) else {}
    try:
        age = float(raw.get("data_age_sec") or 0.0)
    except (TypeError, ValueError):
        age = 0.0
    if age > 10.0:
        return
    feats = _micro_features(features)
    if not any(math.isfinite(v) and v != 0.0 for v in feats.values()) or not math.isfinite(float(target)):
        return
    if any(not math.isfinite(v) for v in feats.values()):
        return
    engine_id = str(engine or "").upper()
    with _connect(db_path) as conn:
        st = _load_linear(conn, engine_id, model)
        a = MICRO_MODEL_STD_ALPHA
        for f, x in feats.items():
            mu = float(st["mu"].get(f, x))
            new_mu = (1.0 - a) * mu + a * x
            s2 = float(st["s2"].get(f, 1.0))
            new_s2 = (1.0 - a) * s2 + a * (x - new_mu) ** 2
            st["mu"][f] = new_mu
            st["s2"][f] = max(new_s2, 1e-9)
        z = _standardize(st, feats)
        pred = float(st["bias"]) + sum(float(st["w"].get(f, 0.0)) * z[f] for f in feats)
        err = float(target) - pred
        st["bias"] = float(st["bias"]) + MICRO_MODEL_LR * err
        for f in feats:
            w = float(st["w"].get(f, 0.0)) + MICRO_MODEL_LR * err * z[f]
            st["w"][f] = _clamp(w, -MICRO_MODEL_W_MAX, MICRO_MODEL_W_MAX)
        st["n"] = int(st.get("n", 0)) + 1
        conn.execute(
            """
            INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(engine_id, model) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (engine_id, model, json.dumps(st, separators=(",", ":")), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        conn.commit()


def micro_edge_tilt(db_path: str, engine: str, features: dict | None) -> tuple[float, int]:
    """Bounded, zero-centred microstructure edge tilt (excludes the bias/mean).
    Returns (tilt, n). Empty features or a cold model return (0.0, 0)."""
    feats = _micro_features(features)
    if not any(v != 0.0 for v in feats.values()):
        return 0.0, 0
    engine_id = str(engine or "").upper()
    try:
        with _connect(db_path) as conn:
            st = _load_linear(conn, engine_id, "micro_edge")
    except sqlite3.Error:
        return 0.0, 0
    n = int(st.get("n", 0))
    if n <= 0:
        return 0.0, 0
    z = _standardize(st, feats)
    tilt = sum(float(st["w"].get(f, 0.0)) * z[f] for f in feats)
    return _clamp(tilt, -MICRO_MODEL_TILT_MAX, MICRO_MODEL_TILT_MAX), n


def learn_claim_label(
    db_path: str,
    *,
    symbol: str,
    setup: str,
    regime: str,
    strategy_version: str,
    base_edge: float,
    residual: float,
    now: float | None = None,
) -> bool:
    """Fold one resolved strategy claim into the key's (base edge, residual) moments.

    ``base_edge`` is the decision-time raw claim minus cost; ``residual`` is the
    realized label minus that base edge.
    """
    b = float(base_edge)
    r = float(residual)
    wrote = False
    for metric, value in zip(CLAIM_MOMENT_METRICS, (b, b * b, r, b * r), strict=True):
        wrote = observe(db_path, engine=SCALP_ENGINE, symbol=symbol, setup=setup, regime=regime, metric=metric, value=value, strategy_version=strategy_version, now=now) or wrote
    return wrote


def scalp_claim_calibration(db_path: str, symbol: str, setup: str, regime: str) -> dict[str, Any]:
    """How much of a strategy claim's base edge shows up in realized net.

    The key residual alone assumes realized net moves 1:1 with the claimed base
    edge. ``claim_capture`` = 1 + the pooled within-key slope of residual on base
    edge across the setup, shrunk toward 1 by its within-key degrees of freedom
    and kept in [0, 1]. Cold, it is exactly 1 (no change to the edge).
    """
    sym = _norm_symbol(symbol)
    reg = str(regime or "").lower()
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT symbol, regime, metric, n, ewma FROM adaptive_metric_state WHERE engine_id=? AND setup=? AND metric IN ({','.join('?' * len(CLAIM_MOMENT_METRICS))})",
                (SCALP_ENGINE, str(setup or "").upper(), *CLAIM_MOMENT_METRICS),
            ).fetchall()
    except sqlite3.Error:
        rows = []
    keys: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        rec = keys.setdefault((_norm_symbol(row["symbol"]), str(row["regime"] or "").lower()), {})
        rec[row["metric"]] = float(row["ewma"])
        rec["n"] = min(rec.get("n", math.inf), float(row["n"] or 0.0))
    cov = var = dof = total_n = total_base = 0.0
    for rec in keys.values():
        if any(m not in rec for m in CLAIM_MOMENT_METRICS):
            continue
        n = rec["n"]
        b = rec["claim_base_edge"]
        cov += n * (rec["claim_base_edge_x_residual"] - b * rec["claim_residual"])
        var += n * max(0.0, rec["claim_base_edge_sq"] - b * b)
        dof += max(0.0, n - 1.0)
        total_n += n
        total_base += n * b
    slope = cov / var if var > 0 else 0.0
    own = keys.get((sym, reg)) or {}
    if own.get("n", 0.0) > 0 and "claim_base_edge" in own:
        base_mean = own["claim_base_edge"]
    else:
        base_mean = total_base / total_n if total_n > 0 else 0.0
    return {
        "claim_capture": _clamp(1.0 + slope * dof / (dof + PRIOR_STRENGTH), 0.0, 1.0),
        "claim_slope": slope,
        "claim_base_mean": base_mean,
        "claim_dof": dof,
        "n_claim": total_n,
    }


def scalp_decision(db_path: str, symbol: str, setup: str, regime: str, features: dict | None = None) -> dict[str, Any]:
    """Learned adjustments for the next SCALP candidate. Never an edge by itself.

    The candidate's own base executable edge (raw expected move - live cost) is
    built in scalp_v2.executable_edge. This view supplies the bounded learned
    residual and microstructure residual added to it, plus confidence, risk,
    target and hold. A cold key contributes a residual of exactly 0.
    """
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_TIME_STOP_MIN

    residual = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "edge_residual")
    residual_strategy = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "edge_residual_strategy")
    net = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_net")
    mfe = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_mfe")
    path_mae = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "markout_mae")
    timing = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_time_to_mfe_min")
    forward = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "markout_forward")
    learned_mfe, _ = _blend([(mfe["mean"], mfe["n"]), (max(0.0, forward["mean"]), forward["n"] * MARKOUT_WEIGHT)], mfe["prior"])
    hard_hold = float(SCALP_V2_TIME_STOP_MIN)
    confidence = float(residual["confidence"])
    adaptive_residual = _clamp(residual["mean"], -SCALP_RESIDUAL_MAX, SCALP_RESIDUAL_MAX)
    # Bounded microstructure residual: shrunk by the model's own sample count so
    # a cold model barely moves the edge. Zero-centred (excludes the model bias).
    micro_tilt, micro_n = micro_edge_tilt(db_path, SCALP_ENGINE, features)
    micro_conf = micro_n / (micro_n + MICRO_MODEL_CONF_K) if micro_n > 0 else 0.0
    micro_residual = micro_conf * micro_tilt
    claim = scalp_claim_calibration(db_path, symbol, setup, regime)
    # Telemetry only: the canonical executable edge is the single live negative-edge gate.
    learned = forward if forward["n"] > 0 else net
    learned_net = learned["mean"]
    abstain, abstain_reason = _abstain(learned_net)
    return {
        "adaptive_state_version": ADAPTIVE_STATE_VERSION,
        "engine_id": SCALP_ENGINE,
        "symbol": str(symbol or "").upper(),
        "setup": str(setup or "").upper(),
        "regime": str(regime or "").lower(),
        "adaptive_residual": adaptive_residual,
        "adaptive_residual_raw": residual["mean"],
        "n_residual": residual["n"],
        "adaptive_residual_strategy": _clamp(residual_strategy["mean"], -SCALP_RESIDUAL_MAX, SCALP_RESIDUAL_MAX),
        "n_residual_strategy": residual_strategy["n"],
        "confidence_strategy": float(residual_strategy["confidence"]),
        "micro_residual": micro_residual,
        "micro_tilt": round(micro_residual, 6),
        "micro_tilt_raw": round(micro_tilt, 6),
        "micro_model_n": micro_n,
        "claim_capture": claim["claim_capture"],
        "claim_slope": claim["claim_slope"],
        "claim_base_mean": claim["claim_base_mean"],
        "claim_dof": claim["claim_dof"],
        "n_claim": claim["n_claim"],
        "confidence": confidence,
        "mfe": learned_mfe,
        "mae": path_mae["mean"],
        "target_pct": _clamp(learned_mfe, *SCALP_TARGET_BOUNDS),
        "hold_min": _clamp(timing["mean"], SCALP_HOLD_FLOOR_MIN, hard_hold),
        "hold_hard_max_min": hard_hold,
        "size_mult": 1.0,
        "risk_estimate": path_mae["mean"],
        "risk_source": "markout_mae_gross_path",
        "n_risk": path_mae["n"],
        "time_to_mfe_min": timing["mean"],
        "n_net": net["n"],
        "n_forward": forward["n"],
        "abstain": abstain,
        "abstain_reason": abstain_reason,
        "abstain_net_edge": learned_net,
        "abstain_confidence": learned["confidence"],
        "abstain_live_veto": False,
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
            ("trade_net", net_pct),
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


def seed_day_trade_net(db_path: str, entered_since: str, *, apply: bool = False) -> dict[str, Any]:
    """Replay current-version DAY closes entered at or after ``entered_since`` into ``trade_net``.

    ``trade_net`` is the realized-net channel; closes before it existed moved
    every other trade metric but not this one. Same value and keys as
    ``learn_from_close``: (exit - entry) / entry minus the estimated round-trip
    cost, under the setup/regime stamped on the entry decision, at close time.
    Refuses once any DAY ``trade_net`` state exists, so a close already learned
    live is never counted twice. Dry run unless ``apply``.
    """
    from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
    from backend.services.strategy_version import is_current_version

    out: dict[str, Any] = {"entered_since": entered_since, "applied": False, "refused": "", "trades": []}
    plan: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            if (
                conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='adaptive_metric_state'").fetchone()
                and conn.execute("SELECT 1 FROM adaptive_metric_state WHERE engine_id=? AND metric='trade_net' LIMIT 1", (DAY_ENGINE,)).fetchone()
            ):
                out["refused"] = "DAY_TRADE_NET_STATE_EXISTS"
                return out
            sells = conn.execute(
                "SELECT * FROM paper_trades WHERE UPPER(side)='SELL' AND UPPER(COALESCE(engine_id,''))=? AND COALESCE(entry_timestamp,'')>=? ORDER BY timestamp",
                (DAY_ENGINE, str(entered_since)),
            ).fetchall()
            for sell in sells:
                if not is_current_version(DAY_ENGINE, sell) or "DUST" in str(sell["exit_reason"] or "").upper():
                    continue
                entry, exit_px = float(sell["entry_price"] or 0.0), float(sell["price"] or 0.0)
                buy = conn.execute(
                    "SELECT adaptive_decision_json FROM paper_trades WHERE UPPER(side)='BUY' AND UPPER(COALESCE(engine_id,''))=? "
                    "AND ((COALESCE(decision_id,'')!='' AND decision_id=?) OR trade_id=?) ORDER BY rowid DESC LIMIT 1",
                    (DAY_ENGINE, sell["decision_id"], sell["trade_id"]),
                ).fetchone()
                try:
                    decision = json.loads((buy["adaptive_decision_json"] if buy else "") or "{}")
                except (TypeError, ValueError):
                    decision = {}
                closed = datetime.fromisoformat(str(sell["timestamp"]).replace("Z", "+00:00"))
                if closed.tzinfo is None:
                    closed = closed.replace(tzinfo=timezone.utc)
                if entry <= 0 or exit_px <= 0 or not isinstance(decision, dict) or not decision.get("setup"):
                    continue
                plan.append(
                    {
                        "trade_id": str(sell["trade_id"]),
                        "symbol": _norm_symbol(sell["symbol"]),
                        "setup": str(decision["setup"]),
                        "regime": str(decision.get("regime") or ""),
                        "strategy_version": str(sell["strategy_version"]),
                        "net_pct": (exit_px - entry) / entry - ESTIMATED_ROUNDTRIP_COST,
                        "closed_at": closed.timestamp(),
                    }
                )
        finally:
            conn.close()
    except (sqlite3.Error, ValueError) as exc:
        out["refused"] = f"READ_FAILED {type(exc).__name__}"
        return out
    out["trades"] = plan
    if apply:
        for trade in plan:
            observe(
                db_path,
                engine=DAY_ENGINE,
                symbol=trade["symbol"],
                setup=trade["setup"],
                regime=trade["regime"],
                metric="trade_net",
                value=trade["net_pct"],
                strategy_version=trade["strategy_version"],
                now=trade["closed_at"],
            )
        out["applied"] = True
    return out


def continuation_ratio(*, entry_price: float, highest_price: float, objective: float) -> float:
    """How far the trade traveled toward the entry-stamped objective.

    0 = no favorable expansion. 1 = reached the objective. Above 1 = continued
    past it. A small net-profitable exit that never approached the objective
    stays near 0. Not a function of whether the close was green.
    """
    entry = float(entry_price or 0.0)
    high = float(highest_price or 0.0)
    goal = float(objective or 0.0)
    if entry <= 0 or high <= 0 or goal <= entry:
        return 0.0
    return max(0.0, (high - entry) / (goal - entry))


def _snap_horizon(target: float, grid: tuple[float, ...]) -> float:
    """Closest grid horizon. Ties take the shorter one. Fixed before any future price."""
    return float(min(grid, key=lambda h: (abs(float(h) - float(target)), float(h))))


def _label_horizon_for(db_path: str, engine_id: str, symbol: str, setup: str, regime: str) -> float:
    """Decision-time horizon. DAY uses the learned time-to-objective; SCALP uses the hold."""
    if engine_id == SCALP_ENGINE:
        view = scalp_decision(db_path, symbol, setup, regime)
        return _snap_horizon(float(view["hold_min"]) * 60.0, SCALP_HORIZONS_SEC)
    view = day_decision(db_path, symbol, setup, regime)
    return _snap_horizon(float(view["time_to_mfe_min"]), DAY_HORIZONS_MIN)


def _horizon_key(horizon: float) -> str:
    h = float(horizon)
    return str(int(h)) if abs(h - int(h)) < 1e-9 else str(h)


def _forward_from_stored(row: sqlite3.Row) -> float | None:
    """Causal forward return already stored on a markout row. Not a P&L."""
    try:
        marks = json.loads(row["markouts_json"] or "{}")
    except (TypeError, ValueError):
        return None
    horizon = float(row["label_horizon"] or 0)
    if horizon <= 0:
        horizon = 60.0 if str(row["engine_id"]) == DAY_ENGINE else 600.0
    return _mark_at(marks if isinstance(marks, dict) else {}, horizon)


def _mark_at(marks: dict, horizon: float) -> float | None:
    raw = marks.get(_horizon_key(horizon))
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _stored_features(features: dict | None) -> str:
    if not isinstance(features, dict) or not features:
        return "{}"
    feats = _micro_features(features)
    if not any(v != 0.0 for v in feats.values()):
        return "{}"
    try:
        age = float(features.get("data_age_sec") or 0.0)
    except (TypeError, ValueError):
        age = 0.0
    if age > 0:
        feats["data_age_sec"] = age
    return json.dumps(feats, separators=(",", ":"))


def _repair_adaptive_keys(conn: sqlite3.Connection) -> None:
    """Fold slash-symbol duplicates onto the normalized key, and drop markout
    rows whose sample count was inflated by re-training one unresolved candidate."""
    rows = list(conn.execute("SELECT rowid, * FROM adaptive_metric_state"))
    for row in rows:
        norm = _norm_symbol(row["symbol"])
        if norm == row["symbol"]:
            continue
        clash = conn.execute(
            "SELECT rowid FROM adaptive_metric_state WHERE engine_id=? AND symbol=? AND setup=? AND regime=? AND metric=?",
            (row["engine_id"], norm, row["setup"], row["regime"], row["metric"]),
        ).fetchone()
        if clash is None:
            conn.execute("UPDATE adaptive_metric_state SET symbol=? WHERE rowid=?", (norm, row["rowid"]))
        else:
            conn.execute("DELETE FROM adaptive_metric_state WHERE rowid=?", (row["rowid"],))
    inflated = list(conn.execute("SELECT rowid, engine_id, symbol, setup, regime, metric, n FROM adaptive_metric_state WHERE metric LIKE 'markout_%'"))
    for row in inflated:
        learned_n = conn.execute(
            "SELECT COUNT(*) FROM adaptive_candidate_markouts WHERE learned=1 AND engine_id=? AND symbol=? AND setup=? AND regime=?",
            (row["engine_id"], row["symbol"], row["setup"], row["regime"]),
        ).fetchone()[0]
        if float(row["n"]) > float(learned_n) + 1.5:
            conn.execute("DELETE FROM adaptive_metric_state WHERE rowid=?", (row["rowid"],))
    conn.commit()


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
    features: dict | None = None,
    raw_expected_move: float | None = None,
    raw_move_source: str | None = None,
    candidate_state: str = "",
) -> int | None:
    """Store one decision-time candidate for causal forward markouts. Returns the row id."""
    engine_id = str(engine or "").upper()
    version = current_strategy_version(engine_id)
    if not version or float(ref_price or 0) <= 0 or not str(setup or "").strip():
        return None
    feats_json = _stored_features(features)
    horizon = _label_horizon_for(db_path, engine_id, symbol, setup, regime)
    raw_move: float | None = None
    with contextlib.suppress(TypeError, ValueError):
        raw_move = float(raw_expected_move) if raw_expected_move is not None and float(raw_expected_move) > 0 else None
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO adaptive_candidate_markouts (
                engine_id, symbol, setup, regime, strategy_version, signaled,
                ref_price, roundtrip_cost, evaluated_at, features_json, label_horizon,
                raw_expected_move, raw_move_source, candidate_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                engine_id,
                _norm_symbol(symbol),
                str(setup or "").upper(),
                str(regime or "").lower(),
                version,
                1 if signaled else 0,
                float(ref_price),
                float(roundtrip_cost or 0),
                float(evaluated_at or time.time()),
                feats_json,
                float(horizon),
                raw_move,
                normalize_raw_move_source(raw_move_source) if raw_move_source else None,
                str(candidate_state or "").upper(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid) if cur.lastrowid is not None else None


def scalp_edge_residual(*, forward_net: float, raw_expected_move: float, roundtrip_cost: float) -> float:
    """Realized net markout minus the decision-time base executable edge."""
    return float(forward_net) - (float(raw_expected_move) - float(roundtrip_cost))


def scalp_gross_path_mae(*, ref_price: float, roundtrip_cost: float, marks: list[float], path_low: float | None) -> float | None:
    """Gross adverse excursion from the entry reference over the committed horizon.

    ``marks`` are net forward marks (cost already subtracted); they are put back
    into gross price units before combining with the bar-low path.
    """
    if ref_price <= 0:
        return None
    prices = [float(ref_price) * (1.0 + float(m) + float(roundtrip_cost)) for m in marks]
    if path_low is not None and math.isfinite(float(path_low)) and float(path_low) > 0:
        prices.append(float(path_low))
    if not prices:
        return None
    return max(0.0, (float(ref_price) - min(prices)) / float(ref_price))


def resolve_markouts(
    db_path: str,
    quote: Callable[[str, float], float | None],
    *,
    now: float | None = None,
    path_low: Callable[[str, float, float], float | None] | None = None,
) -> int:
    """Fill due forward marks and fold the decision-time horizon into state.

    Every horizon is stored. The learned label is the return at the horizon
    stamped when the candidate was recorded — not the best later horizon.
    A row is claimed (learned=1) before the state write so a locked retry
    cannot train the same markout twice. Returns rows newly learned.

    SCALP rows also learn the edge residual (label minus the decision-time base
    executable edge) and the gross path MAE over the committed horizon, which
    is the live risk estimate. ``path_low(symbol, start, end)`` supplies the
    bar-low path when available.
    """
    moment = float(now if now is not None else time.time())
    learned = 0
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        return 0
    try:
        with contextlib.suppress(sqlite3.Error):
            _repair_adaptive_keys(conn)
        rows = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE resolved=0 ORDER BY id ASC LIMIT 200").fetchall()
        for row in rows:
            try:
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
                    try:
                        price = quote(str(row["symbol"]), due)
                    except Exception:
                        price = None
                    if price is None and moment < due + grace:
                        done = False
                        continue
                    if price is None or float(row["ref_price"]) <= 0 or not math.isfinite(float(price or 0)):
                        marks[key] = None
                    else:
                        marks[key] = (float(price) - float(row["ref_price"])) / float(row["ref_price"]) - float(row["roundtrip_cost"] or 0)
                label_h = float(row["label_horizon"] or 0)
                if label_h <= 0:
                    label_h = 60.0 if engine_id == DAY_ENGINE else 600.0
                forward = _mark_at(marks, label_h)
                path = [_mark_at(marks, h) for h in horizons if h <= label_h + 1e-9]
                path = [v for v in path if v is not None]
                mae = max(0.0, -min(path)) if path else None
                residual = None
                micro_target = forward
                cols = row.keys()
                state = str(row["candidate_state"] or "") if "candidate_state" in cols else ""
                learnable = not (engine_id == DAY_ENGINE and state == CANDIDATE_NEAR_QUALIFIED)
                raw_move = None
                raw_source = None
                if engine_id == SCALP_ENGINE:
                    low = None
                    if path_low is not None and forward is not None:
                        with contextlib.suppress(Exception):
                            low = path_low(str(row["symbol"]), float(row["evaluated_at"]), float(row["evaluated_at"]) + label_h)
                    mae = scalp_gross_path_mae(ref_price=float(row["ref_price"]), roundtrip_cost=float(row["roundtrip_cost"] or 0), marks=path, path_low=low)
                    raw_move = row["raw_expected_move"] if "raw_expected_move" in cols else None
                    raw_source = row["raw_move_source"] if "raw_move_source" in cols else None
                    if forward is not None and raw_move is not None and float(raw_move) > 0:
                        residual = scalp_edge_residual(forward_net=forward, raw_expected_move=float(raw_move), roundtrip_cost=float(row["roundtrip_cost"] or 0))
                    # Only a directional claim's residual trains the micro residual;
                    # forensic ATR-estimate residuals must not shape live edge.
                    micro_target = residual if is_directional(raw_source) else None
                row_learned = int(row["learned"] or 0)
                version_ok = str(row["strategy_version"]) == current_strategy_version(engine_id)
                if forward is not None and not row_learned and version_ok and learnable:
                    # Claim first. A failed state write must not re-train this row.
                    cur = conn.execute(
                        "UPDATE adaptive_candidate_markouts SET markouts_json=?, learned=1, resolved=? WHERE id=? AND learned=0",
                        (json.dumps(marks), 1 if done else 0, row["id"]),
                    )
                    conn.commit()
                    if cur.rowcount != 1:
                        continue
                    observe(
                        db_path,
                        engine=engine_id,
                        symbol=row["symbol"],
                        setup=row["setup"],
                        regime=row["regime"],
                        metric="markout_forward",
                        value=forward,
                        strategy_version=str(row["strategy_version"]),
                    )
                    if mae is not None:
                        observe(
                            db_path,
                            engine=engine_id,
                            symbol=row["symbol"],
                            setup=row["setup"],
                            regime=row["regime"],
                            metric="markout_mae",
                            value=mae,
                            strategy_version=str(row["strategy_version"]),
                        )
                    if residual is not None:
                        observe(
                            db_path,
                            engine=engine_id,
                            symbol=row["symbol"],
                            setup=row["setup"],
                            regime=row["regime"],
                            metric=residual_metric(raw_source),
                            value=residual,
                            strategy_version=str(row["strategy_version"]),
                        )
                        if is_directional(raw_source) and raw_move is not None:
                            learn_claim_label(
                                db_path,
                                symbol=row["symbol"],
                                setup=row["setup"],
                                regime=row["regime"],
                                strategy_version=str(row["strategy_version"]),
                                base_edge=float(raw_move) - float(row["roundtrip_cost"] or 0),
                                residual=residual,
                            )
                    if engine_id == SCALP_ENGINE and micro_target is not None:
                        with contextlib.suppress(Exception):
                            feats = json.loads(row["features_json"] or "{}")
                            if isinstance(feats, dict) and feats:
                                update_linear_model(db_path, SCALP_ENGINE, "micro_edge", feats, micro_target)
                    learned += 1
                else:
                    conn.execute(
                        "UPDATE adaptive_candidate_markouts SET markouts_json=?, learned=?, resolved=? WHERE id=?",
                        (json.dumps(marks), row_learned, 1 if done else 0, row["id"]),
                    )
                    conn.commit()
            except sqlite3.Error:
                logger.warning("MARKOUT_RESOLVE_ROW_FAILED id=%s", row["id"])
                continue
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


def ohlcv_low_between(db_path: str, symbol: str, start: float, end: float) -> float | None:
    """Lowest 1m low over bars opening in [start, end). The decision-minute bar is excluded."""
    raw = str(symbol or "").upper().replace("-", "").replace("/", "")
    lo_s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(math.ceil(float(start) / 60.0) * 60.0))
    hi_s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(end)))
    try:
        conn = sqlite3.connect(db_path, timeout=5)
    except sqlite3.Error:
        return None
    try:
        for variant in (raw, raw.replace("USDT", "-USDT"), raw.replace("USDT", "/USDT")):
            row = conn.execute(
                "SELECT MIN(low) FROM feature_ohlcv WHERE symbol=? AND interval='1m' AND ts>=? AND ts<?",
                (variant, lo_s, hi_s),
            ).fetchone()
            if row and row[0] is not None:
                return float(row[0])
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return None


def _ohlc_rows(conn: sqlite3.Connection, symbol: str, interval: str, limit: int) -> list[tuple[float, float, float, float]]:
    """Newest-last (ts asc) high/low/close for a symbol+interval, across name variants."""
    raw = str(symbol or "").upper().replace("-", "").replace("/", "")
    for variant in (raw, raw.replace("USDT", "-USDT"), raw.replace("USDT", "/USDT")):
        try:
            rows = conn.execute(
                "SELECT ts, high, low, close FROM feature_ohlcv WHERE symbol=? AND interval=? ORDER BY ts DESC LIMIT ?",
                (variant, interval, int(limit)),
            ).fetchall()
        except sqlite3.Error:
            return []
        if rows:
            out = []
            for _ts, hi, lo, cl in rows:
                if hi is None or lo is None or cl is None:
                    continue
                out.append((float(hi), float(lo), float(cl)))
            out.reverse()
            return out
    return []


def _trend_bucket(conn: sqlite3.Connection, symbol: str, interval: str, lookback: int) -> str:
    rows = _ohlc_rows(conn, symbol, interval, lookback)
    if len(rows) < 4:
        return ""
    first = rows[0][2]
    last = rows[-1][2]
    if first <= 0:
        return ""
    change = (last - first) / first
    if change >= 0.005:
        return "up"
    if change <= -0.005:
        return "down"
    return "flat"


def _vol_bucket(conn: sqlite3.Connection, symbol: str) -> str:
    rows = _ohlc_rows(conn, symbol, "15m", 30)
    if len(rows) < 12:
        return ""

    def _atr(window: list[tuple[float, float, float]]) -> float:
        trs = []
        for i in range(1, len(window)):
            hi, lo, _cl = window[i]
            prev_c = window[i - 1][2]
            trs.append(max(hi - lo, abs(hi - prev_c), abs(lo - prev_c)))
        return sum(trs) / len(trs) if trs else 0.0

    short = _atr(rows[-7:])
    long = _atr(rows[-25:]) if len(rows) >= 25 else _atr(rows)
    if long <= 0:
        return ""
    return "volhi" if short / long >= 1.15 else "vollo"


def market_regime_tag(db_path: str, symbol: str) -> str:
    """Coarse, inspectable regime key: BTC 1h trend + this symbol's 15m vol bucket.

    Returns e.g. 'btcup_volhi'. Empty string when data is missing, so callers
    fall back to whatever regime string they already had (never crashes, never
    gates). Read/write alignment is preserved because the tag is stamped on the
    entry decision and reused at close.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5)
    except sqlite3.Error:
        return ""
    try:
        btc = _trend_bucket(conn, "BTCUSDT", "1h", 24)
        vol = _vol_bucket(conn, symbol)
    finally:
        conn.close()
    if not btc or not vol:
        return ""
    return f"btc{btc}_{vol}"


def adaptive_state_report(db_path: str) -> dict[str, Any]:
    """Current learned state per engine, as the decision views the next candidate reads.

    Read-only. Groups the distinct (symbol, setup, regime) keys that carry any
    observations and returns the DAY/SCALP decision for each so a dashboard can
    show exactly what learning currently changes and why.
    """
    out: dict[str, Any] = {
        "adaptive_state_version": ADAPTIVE_STATE_VERSION,
        "half_life_days": ADAPTIVE_HALF_LIFE_DAYS,
        "engines": {},
        "forensic": [],
    }
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        return out
    try:
        for engine_id, decide in ((DAY_ENGINE, day_decision), (SCALP_ENGINE, scalp_decision)):
            keys = conn.execute(
                "SELECT DISTINCT symbol, setup, regime FROM adaptive_metric_state WHERE engine_id=? ORDER BY symbol, setup, regime",
                (engine_id,),
            ).fetchall()
            rows = []
            for k in keys:
                if _is_forensic_key(engine_id, k["symbol"], k["setup"], k["regime"]):
                    out["forensic"].append(
                        {
                            "engine_id": engine_id,
                            "symbol": _norm_symbol(k["symbol"]),
                            "setup": str(k["setup"] or "").upper(),
                            "regime": str(k["regime"] or "").lower(),
                            "classification": "legacy_pre_stamp",
                        }
                    )
                    continue
                view = decide(db_path, k["symbol"], k["setup"], k["regime"])
                if engine_id == DAY_ENGINE:
                    rows.append(
                        {
                            "symbol": k["symbol"],
                            "setup": k["setup"],
                            "regime": k["regime"],
                            "expected_move": round(view["expected_move"], 6),
                            "expected_net": round(view["expected_net"], 6),
                            "size_mult": round(view["size_mult"], 4),
                            "objective_atr_mult": round(view["objective_atr_mult"], 4),
                            "confidence": round(view["confidence"], 3),
                            "net_confidence": round(view["net_confidence"], 3),
                            "abstain": view["abstain"],
                            "n": view["n_mfe"],
                            "n_trade_net": view["n_trade_net"],
                            "n_forward": view["n_forward"],
                        }
                    )
                else:
                    rows.append(
                        {
                            "symbol": k["symbol"],
                            "setup": k["setup"],
                            "regime": k["regime"],
                            "adaptive_residual": round(view["adaptive_residual"], 6),
                            "adaptive_residual_strategy": round(view["adaptive_residual_strategy"], 6),
                            "claim_capture": round(view["claim_capture"], 4),
                            "claim_base_mean": round(view["claim_base_mean"], 6),
                            "risk_estimate": round(view["risk_estimate"], 6),
                            "target_pct": round(view["target_pct"], 5),
                            "hold_min": round(view["hold_min"], 2),
                            "confidence": round(view["confidence"], 3),
                            "abstain": view["abstain"],
                            "n": view["n_residual"],
                        }
                    )
            out["engines"][engine_id] = rows
        # Inspectable SCALP microstructure model: its learned weights and mean.
        micro = _load_linear(conn, SCALP_ENGINE, "micro_edge")
        out["scalp_micro_model"] = {
            "n": int(micro.get("n", 0)),
            "bias": round(float(micro.get("bias", 0.0)), 6),
            "tilt_max": MICRO_MODEL_TILT_MAX,
            "features": SCALP_MICRO_FEATURES,
            "weights": {f: round(float(micro.get("w", {}).get(f, 0.0)), 4) for f in SCALP_MICRO_FEATURES},
        }
    finally:
        conn.close()
    return out


def calibration_report(db_path: str) -> dict[str, Any]:
    """Do higher-confidence / larger-size entries realize better net? Read-only.

    Buckets current-version SELL rows by the entry-stamped confidence and size
    multiplier, and reports realized net %. Pure telemetry: it never gates a
    trade, it only exposes whether the learned controls are calibrated.
    """
    from backend.services.strategy_version import is_current_version

    out: dict[str, Any] = {"adaptive_state_version": ADAPTIVE_STATE_VERSION, "engines": {}}
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    except sqlite3.Error:
        return out
    if "adaptive_decision_json" not in cols or "pnl_pct_net" not in cols:
        conn.close()
        return out

    def _conf_bucket(c: float) -> str:
        if c < 0.10:
            return "conf<0.10"
        if c < 0.30:
            return "conf0.10-0.30"
        return "conf>=0.30"

    def _size_bucket(s: float) -> str:
        if s < 0.90:
            return "size<0.90"
        if s <= 1.10:
            return "size0.90-1.10"
        return "size>1.10"

    try:
        rows = conn.execute(
            "SELECT engine_id, strategy_version, entry_contract_version, exit_contract_version, "
            "pnl_pct_net, adaptive_decision_json FROM paper_trades "
            "WHERE UPPER(side)='SELL' AND LOWER(COALESCE(mode,''))='live' "
            "AND adaptive_decision_json IS NOT NULL AND adaptive_decision_json != '' "
            "AND pnl_pct_net IS NOT NULL"
        ).fetchall()
    except sqlite3.Error:
        conn.close()
        return out
    conn.close()
    agg: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        engine = str(row["engine_id"] or "").upper()
        if engine not in _PRIORS or not is_current_version(engine, row):
            continue
        try:
            dec = json.loads(row["adaptive_decision_json"] or "{}")
        except (ValueError, TypeError):
            continue
        conf = float(dec.get("confidence") or 0.0)
        size = float(dec.get("size_mult") or 1.0)
        net = float(row["pnl_pct_net"])
        eng = agg.setdefault(engine, {})
        eng.setdefault(_conf_bucket(conf), []).append(net)
        eng.setdefault(_size_bucket(size), []).append(net)
    for engine, buckets in agg.items():
        out["engines"][engine] = {b: {"n": len(v), "avg_net_pct": round(sum(v) / len(v), 6), "wins": sum(1 for x in v if x > 0)} for b, v in sorted(buckets.items())}
    return out


def abstention_report(db_path: str, window_days: float = 7.0) -> dict[str, Any]:
    """Which keys carry a learned negative net expectancy? Read-only telemetry.

    Per engine: keys whose learned net edge is at or below the margin versus
    the rest, plus any historical REJECTED:LEARNED_NEGATIVE_EDGE skips in the
    window and their counterfactual markouts. Neither engine vetoes on the flag:
    DAY moves size and rank, SCALP gates on its canonical executable edge.
    """
    since = time.time() - float(window_days) * 86400.0
    out: dict[str, Any] = {
        "adaptive_state_version": ADAPTIVE_STATE_VERSION,
        "enabled": ABSTAIN_ENABLED,
        "live_veto": False,
        "edge_margin": ABSTAIN_EDGE_MARGIN,
        "window_days": float(window_days),
        "engines": {},
    }
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        return out
    try:
        for engine_id, decide, table in (
            (DAY_ENGINE, day_decision, "day_v2_decisions"),
            (SCALP_ENGINE, scalp_decision, "scalp_v2_decisions"),
        ):
            keys = conn.execute(
                "SELECT DISTINCT symbol, setup, regime FROM adaptive_metric_state WHERE engine_id=? ORDER BY symbol, setup, regime",
                (engine_id,),
            ).fetchall()
            abstaining: list[dict[str, Any]] = []
            active: list[dict[str, Any]] = []
            for k in keys:
                if _is_forensic_key(engine_id, k["symbol"], k["setup"], k["regime"]):
                    continue
                v = decide(db_path, k["symbol"], k["setup"], k["regime"])
                rec = {
                    "symbol": k["symbol"],
                    "setup": k["setup"],
                    "regime": k["regime"],
                    "net_edge": round(float(v["abstain_net_edge"]), 6),
                    "confidence": round(float(v["abstain_confidence"]), 3),
                }
                (abstaining if v["abstain"] else active).append(rec)
            skips = 0
            with contextlib.suppress(sqlite3.Error):
                skips = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE cycle_ts>=? AND result=?",
                        (since, "REJECTED:LEARNED_NEGATIVE_EDGE"),
                    ).fetchone()[0]
                )
            avg_abstained = round(sum(r["net_edge"] for r in abstaining) / len(abstaining), 6) if abstaining else None
            avg_active = round(sum(r["net_edge"] for r in active) / len(active), 6) if active else None
            # Subsequent candidate markouts: what the skipped opportunity's
            # decision-time horizon actually did, versus candidates that were
            # not skipped. Counterfactual marks, not realized P&L.
            skip_events: list = []
            mark_rows: list = []
            with contextlib.suppress(sqlite3.Error):
                skip_events = conn.execute(
                    f"SELECT symbol, cycle_ts FROM {table} WHERE cycle_ts>=? AND result=?",
                    (since, "REJECTED:LEARNED_NEGATIVE_EDGE"),
                ).fetchall()
            with contextlib.suppress(sqlite3.Error):
                mark_rows = conn.execute(
                    "SELECT symbol, setup, regime, evaluated_at, markouts_json, label_horizon, engine_id FROM adaptive_candidate_markouts WHERE engine_id=? AND evaluated_at>=?",
                    (engine_id, since),
                ).fetchall()
            used: set[int] = set()
            skipped_fw: list[float] = []
            by_key: dict[tuple, list[float]] = {}
            for ev in skip_events:
                for i, m in enumerate(mark_rows):
                    if i in used:
                        continue
                    same = _norm_symbol(ev["symbol"]) == _norm_symbol(m["symbol"])
                    close_in_time = abs(float(ev["cycle_ts"]) - float(m["evaluated_at"])) <= 180.0
                    if not (same and close_in_time):
                        continue
                    used.add(i)
                    fw = _forward_from_stored(m)
                    if fw is None:
                        break
                    skipped_fw.append(fw)
                    by_key.setdefault((m["symbol"], m["setup"], m["regime"]), []).append(fw)
                    break
            kept_fw = [fw for i, m in enumerate(mark_rows) if i not in used and (fw := _forward_from_stored(m)) is not None]
            out["engines"][engine_id] = {
                "skips_in_window": skips,
                "abstaining_keys": len(abstaining),
                "active_keys": len(active),
                "avg_net_edge_abstained": avg_abstained,
                "avg_net_edge_active": avg_active,
                "bps_avoided_per_skip": round(-avg_abstained * 10000.0, 2) if (avg_abstained is not None and avg_abstained < 0) else 0.0,
                "abstaining": sorted(abstaining, key=lambda r: r["net_edge"])[:20],
                "skipped_markouts": len(skipped_fw),
                "skipped_avg_forward": round(sum(skipped_fw) / len(skipped_fw), 6) if skipped_fw else None,
                "kept_markouts": len(kept_fw),
                "kept_avg_forward": round(sum(kept_fw) / len(kept_fw), 6) if kept_fw else None,
                "skipped_by_key": [{"symbol": k[0], "setup": k[1], "regime": k[2], "n": len(v), "avg_forward": round(sum(v) / len(v), 6)} for k, v in sorted(by_key.items())],
            }
    finally:
        conn.close()
    return out


def day_candidate_markout_report(db_path: str, window_days: float = 7.0) -> dict[str, Any]:
    """Causal DAY candidate markouts by decision state and setup. Read-only.

    Current strategy version only. NEAR_QUALIFIED rows were one entry condition
    short (evidence, never learned). Qualified rows are SELECTED when the same
    bar's DAY decision was FILLED and REJECTED otherwise. Values are net of the
    stamped round-trip cost from the decision-time executable price.
    """
    since = time.time() - float(window_days) * 86400.0
    version = current_strategy_version(DAY_ENGINE)
    out: dict[str, Any] = {"strategy_version": version, "window_days": float(window_days), "horizons_min": list(DAY_HORIZONS_MIN), "states": {}}
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return out
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(adaptive_candidate_markouts)")}
        if not cols:
            return out
        state_col = "candidate_state" if "candidate_state" in cols else "''"
        rows = conn.execute(
            f"SELECT symbol, setup, evaluated_at, markouts_json, {state_col} AS state FROM adaptive_candidate_markouts WHERE engine_id=? AND strategy_version=? AND evaluated_at>=?",
            (DAY_ENGINE, version, since),
        ).fetchall()
        results: dict[tuple[str, float], str] = {}
        with contextlib.suppress(sqlite3.Error):
            for d in conn.execute("SELECT symbol, cycle_ts, result FROM day_v2_decisions WHERE cycle_ts>=?", (since,)):
                results[(_norm_symbol(d["symbol"]), float(d["cycle_ts"]))] = str(d["result"] or "")
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    agg: dict[str, dict[str, dict[str, list[float]]]] = {}
    for row in rows:
        state = str(row["state"] or "")
        if state != CANDIDATE_NEAR_QUALIFIED:
            result = results.get((_norm_symbol(row["symbol"]), float(row["evaluated_at"])), "")
            state = "SELECTED" if result == "FILLED" else "REJECTED"
        try:
            marks = json.loads(row["markouts_json"] or "{}")
        except (TypeError, ValueError):
            continue
        per_h = agg.setdefault(state, {}).setdefault(str(row["setup"] or ""), {})
        for h in DAY_HORIZONS_MIN:
            value = _mark_at(marks if isinstance(marks, dict) else {}, h)
            if value is not None:
                per_h.setdefault(str(h), []).append(value)
    for state, setups in agg.items():
        out["states"][state] = {
            setup: {
                h: {
                    "n": len(v),
                    "mean": round(sum(v) / len(v), 6),
                    "median": round(sorted(v)[len(v) // 2], 6),
                    "positive_rate": round(sum(1 for x in v if x > 0) / len(v), 3),
                }
                for h, v in sorted(per_h.items(), key=lambda kv: float(kv[0]))
            }
            for setup, per_h in sorted(setups.items())
        }
    return out


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
    "ADAPTIVE_HALF_LIFE_DAYS",
    "ADAPTIVE_STATE_VERSION",
    "DAY_HORIZONS_MIN",
    "SCALP_HORIZONS_SEC",
    "abstention_report",
    "adaptive_state_report",
    "calibration_report",
    "continuation_ratio",
    "day_candidate_markout_report",
    "day_decision",
    "day_net_expectancy",
    "estimate",
    "learn_claim_label",
    "learn_from_close",
    "market_regime_tag",
    "micro_edge_tilt",
    "observe",
    "ohlcv_quote",
    "persist_trade_adaptive",
    "record_candidate",
    "resolve_markouts",
    "scalp_claim_calibration",
    "scalp_decision",
    "seed_day_trade_net",
    "update_linear_model",
]
