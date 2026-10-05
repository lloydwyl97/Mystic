"""Online adaptive state for DAY_V2 and SCALP_V2.

One framework, two engines. A row is keyed by engine, economic version, symbol,
setup, regime and metric, so DAY observations never update SCALP, and state
learned under another contract, anchor or learner format is never read. The
economic version (``strategy_version.economic_version``) is stamped on every
state row and candidate row; evidence decided before the engine's economic
anchor never moves state.

Estimates are hierarchical. A key (symbol + setup + regime) shrinks toward the
related evidence of the same setup (same symbol or same regime), which shrinks
toward the setup, which shrinks toward the engine, which shrinks toward the
prior. Each observation is counted at exactly one level, and each level's
weight is its decayed sample count against ``PRIOR_STRENGTH``: thin specific
evidence leans on the broader levels, informative specific evidence dominates.
There is no minimum-trade gate and no profit-factor gate.

Realized closes update ``trade_*`` metrics. DAY candidates update
``lifecycle_net`` (the live exit contract replayed from the decision ask) and
the fixed-horizon ``markout_*`` diagnostics. SCALP claims update the claim
calibration. Decisions blend these continuously; DAY net expectancy sets
bounded size and rank, SCALP calibration sets the executable edge.
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
from backend.services.strategy_version import ADAPTIVE_STATE_VERSION, economic_anchor, economic_version, engine_versions

PRIOR_STRENGTH = 8.0
EWMA_ALPHA = 0.25
MARKOUT_WEIGHT = 0.35
# DAY lifecycle labels replay the live exit contract on 1m bars; against the
# realized post-anchor closes they correlate 0.94. They count slightly below a
# realized close so simulated evidence never outweighs the same amount of fills.
LIFECYCLE_WEIGHT = 0.75

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
        # Net quantities (realized trade net after costs, lifecycle net,
        # cost-adjusted forward markout) start neutral: no edge is assumed
        # before evidence. The MFE prior is a path maximum, not a net return,
        # and must not seed them.
        "trade_net": 0.0,
        "lifecycle_net": 0.0,
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
        # Decayed moments of (decision-time micro tilt, what the edge before micro
        # missed) for the learned micro weight.
        "micro_tilt_sq": 0.0,
        "micro_tilt_x_miss": 0.0,
    },
}

CLAIM_MOMENT_METRICS = ("claim_base_edge", "claim_base_edge_sq", "claim_residual", "claim_base_edge_x_residual")
MICRO_WEIGHT_METRICS = ("micro_tilt_sq", "micro_tilt_x_miss")
MICRO_WEIGHT_KEY = ("", "MICRO_MODEL", "")

# DAY candidate states. QUALIFIED: the setup fired and passed integrity checks.
# QUALIFIED_BLOCKED: it fired but a capacity rule (24h frequency cap) stopped
# it; same contract, so its lifecycle is learnable. NEAR_QUALIFIED: one entry
# condition short; resolved for evidence and never folded into decision state.
CANDIDATE_QUALIFIED = "QUALIFIED"
CANDIDATE_QUALIFIED_BLOCKED = "QUALIFIED_BLOCKED"
CANDIDATE_NEAR_QUALIFIED = "NEAR_QUALIFIED"
LEARNABLE_DAY_STATES = frozenset({"", CANDIDATE_QUALIFIED, CANDIDATE_QUALIFIED_BLOCKED})

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
MEAN_FORM_METRICS = frozenset({"edge_residual", "edge_residual_strategy", "lifecycle_net", "trade_net", *CLAIM_MOMENT_METRICS, *MICRO_WEIGHT_METRICS})

SIZE_BOUNDS = {"DAY_V2": (0.55, 1.35), "SCALP_V2": (0.50, 1.25)}
# Prior information on the SCALP claim slope, in claim-variance units: a key's
# within-key claim spread must reach PRIOR_STRENGTH claims of this variance to
# weigh as much as the parent slope.
CLAIM_SLOPE_PRIOR_VAR = 0.0008**2
# Prior information on the micro weight, in tilt-variance units (typical tilt ~2 bps).
MICRO_WEIGHT_PRIOR_VAR = 0.0002**2
OBJECTIVE_ATR_BOUNDS = (0.75, 1.35)
STRUCTURAL_EMPHASIS_BOUNDS = (0.85, 1.25)
ACTIVATION_BOUNDS = (0.80, 1.25)
TRAIL_BOUNDS = (0.80, 1.20)
TIGHTEN_BOUNDS = (0.75, 1.15)
SCALP_TARGET_BOUNDS = (0.0015, 0.006)
SCALP_HOLD_FLOOR_MIN = 4.0

# Half-life for observation weight, per engine. Older evidence loses effective
# sample count, at write time and at read time, so a key that stops receiving
# evidence relaxes toward its parent and a losing state can recover. SCALP
# resolves hundreds of claims a day, DAY about ten, so SCALP forgets faster.
# It never zeroes a key, so there is no min-trade gate and no hard stop.
ADAPTIVE_HALF_LIFE_DAYS = float(os.getenv("ADAPTIVE_HALF_LIFE_DAYS", "14") or "14")
SCALP_HALF_LIFE_DAYS = float(os.getenv("ADAPTIVE_SCALP_HALF_LIFE_DAYS", "3") or "3")

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


def half_life_days(engine: str) -> float:
    return SCALP_HALF_LIFE_DAYS if str(engine or "").upper() == SCALP_ENGINE else ADAPTIVE_HALF_LIFE_DAYS


def _decay_factor(updated_at: str, now_epoch: float, engine: str = DAY_ENGINE) -> float:
    """Weight retained for a key's prior sample count, by age. 1.0 if age unknown."""
    t0 = _parse_iso(updated_at)
    half_life = half_life_days(engine) * 86400.0
    if t0 is None or half_life <= 0:
        return 1.0
    elapsed = max(0.0, float(now_epoch) - t0)
    return float(0.5 ** (elapsed / half_life))


def _data_clock(rows: list[sqlite3.Row]) -> float:
    """Newest evidence time among ``rows``: reads decay older keys against the
    latest information, so identical state always reads identically."""
    stamps = [t for r in rows if (t := _parse_iso(str(r["updated_at"] or ""))) is not None]
    return max(stamps) if stamps else 0.0


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


def current_economic_version(engine: str) -> str:
    return economic_version(str(engine or "").upper())


def anchor_epoch(engine: str) -> float:
    anchor = economic_anchor(str(engine or "").upper())
    return float(anchor["epoch"]) if anchor else 0.0


_STATE_DDL = """
    CREATE TABLE IF NOT EXISTS adaptive_metric_state (
        engine_id TEXT NOT NULL,
        economic_version TEXT NOT NULL,
        symbol TEXT NOT NULL,
        setup TEXT NOT NULL,
        regime TEXT NOT NULL,
        metric TEXT NOT NULL,
        n REAL NOT NULL,
        ewma REAL NOT NULL,
        m2 REAL NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (engine_id, economic_version, symbol, setup, regime, metric)
    )
"""
LEGACY_STATE_TABLE = "adaptive_metric_state_legacy"


def _migrate_state_table(conn: sqlite3.Connection) -> None:
    """Move an unversioned state table aside (forensic) and create the versioned one.

    Runs once per database. The legacy rows mixed contract versions and are
    never read as economic state again; they stay in ``adaptive_metric_state_legacy``.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(adaptive_metric_state)").fetchall()}
    if cols and "economic_version" in cols:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(adaptive_metric_state)").fetchall()}
        if cols and "economic_version" not in cols:
            name = LEGACY_STATE_TABLE
            suffix = 1
            while conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone():
                suffix += 1
                name = f"{LEGACY_STATE_TABLE}_{suffix}"
            conn.execute(f"ALTER TABLE adaptive_metric_state RENAME TO {name}")
        conn.execute(_STATE_DDL)
        conn.execute("COMMIT")
    except sqlite3.Error:
        conn.execute("ROLLBACK")
        raise


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=15000")
    _migrate_state_table(conn)
    conn.execute(_STATE_DDL)
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
    # Economic version, DAY lifecycle inputs, decision-time economics, the
    # opportunity identity (one learnable label per opportunity) and fill state.
    for col, ddl in (
        ("economic_version", "TEXT NOT NULL DEFAULT ''"),
        ("lifecycle_json", "TEXT NOT NULL DEFAULT ''"),
        ("economic_json", "TEXT NOT NULL DEFAULT '{}'"),
        ("opportunity_id", "TEXT NOT NULL DEFAULT ''"),
        ("filled", "INTEGER NOT NULL DEFAULT 0"),
        ("lifecycle_learned", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE adaptive_candidate_markouts ADD COLUMN {col} {ddl}")
    # resolve_markouts runs every SCALP cycle; without these, its key repair and
    # unresolved scan are full table scans that grow with the markout history.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_key ON adaptive_candidate_markouts(engine_id, symbol, setup, regime, learned)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_unresolved ON adaptive_candidate_markouts(resolved, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_opportunity ON adaptive_candidate_markouts(engine_id, opportunity_id)")
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
    """Fold one current-version observation into the key under the current economic version.

    Returns False for another strategy version, an unknown metric or a value
    that is not finite. ``now`` is the observation time (defaults to wall clock);
    a chronological rebuild passes each label's own time so decay matches live
    learning. The decayed second moment ``m2`` feeds the reported uncertainty.
    """
    engine_id = str(engine or "").upper()
    if engine_id not in _PRIORS or metric not in _PRIORS[engine_id]:
        return False
    if str(strategy_version or "") != current_strategy_version(engine_id):
        return False
    if value is None or not math.isfinite(float(value)):
        return False
    value = float(value)
    key = (engine_id, current_economic_version(engine_id), _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower(), metric)
    moment = float(now if now is not None else time.time())
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT n, ewma, m2, updated_at FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND symbol=? AND setup=? AND regime=? AND metric=?",
            key,
        ).fetchone()
        if row is None or float(row["n"]) <= 0:
            n, mean, m2 = 1.0, value, 0.0
        else:
            # Age out the prior sample count so stale evidence stops dominating,
            # then fold in the new observation. The new point always counts for 1.
            decay = _decay_factor(str(row["updated_at"] or ""), moment, engine_id)
            n = float(row["n"]) * decay + 1.0
            alpha = 1.0 / n if metric in MEAN_FORM_METRICS else EWMA_ALPHA
            delta = value - float(row["ewma"])
            mean = float(row["ewma"]) + alpha * delta
            m2 = max(0.0, float(row["m2"] or 0.0)) * decay + delta * (value - mean)
        conn.execute(
            """
            INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_id, economic_version, symbol, setup, regime, metric) DO UPDATE SET
                n=excluded.n, ewma=excluded.ewma, m2=excluded.m2, updated_at=excluded.updated_at
            """,
            (*key, n, mean, max(0.0, m2), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))),
        )
        conn.commit()
    return True


# Hierarchy levels, broadest first, nested: ENGINE is every row of the engine,
# SETUP every row of the key's setup, RELATED the setup's rows sharing the
# key's symbol or regime, KEY the exact key. ``_level_of`` gives the innermost
# level a row belongs to. Estimates pool within the key's setup only: another
# setup's evidence never moves this setup's value or confidence. The ENGINE
# figure is reported for audit and feeds the outcome dispersion.
LATTICE_LEVELS = ("engine", "setup", "related", "key")
# Relevance of each level to the key, for the reported confidence/uncertainty only.
_LEVEL_RELEVANCE = {"engine": 0.0, "setup": 0.25, "related": 0.5, "key": 1.0}


def _level_of(row: sqlite3.Row, sym: str, stp: str, reg: str) -> str:
    if str(row["setup"] or "").upper() != stp:
        return "engine"
    same_sym = _norm_symbol(row["symbol"]) == sym
    same_reg = str(row["regime"] or "").lower() == reg
    if same_sym and same_reg:
        return "key"
    return "related" if (same_sym or same_reg) else "setup"


def _state_rows(db_path: str, engine_id: str, metrics: tuple[str, ...]) -> list[sqlite3.Row]:
    """Current-economic-version state rows of one engine for the given metrics."""
    try:
        with _connect(db_path) as conn:
            return conn.execute(
                f"SELECT symbol, setup, regime, metric, n, ewma, m2, updated_at FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND metric IN ({','.join('?' * len(metrics))})",
                (engine_id, current_economic_version(engine_id), *metrics),
            ).fetchall()
    except sqlite3.Error:
        return []


def _lattice(
    rows: list[sqlite3.Row],
    *,
    engine_id: str,
    symbol: str,
    setup: str,
    regime: str,
    weights: dict[str, float],
    prior: float,
    now: float | None = None,
) -> dict[str, Any]:
    """Hierarchical posterior mean of a (weighted) metric set at one key.

    Evidence is decayed sample count times the metric weight. Starting from the
    prior, each nested level from SETUP to KEY is the mean of all evidence
    inside it, shrunk toward the level above by ``PRIOR_STRENGTH / (PRIOR_STRENGTH + W)``.
    With no evidence at a level the parent passes through unchanged. There is no
    sample floor. The ENGINE level (all setups) is computed for audit only.
    """
    sym, stp, reg = _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower()
    moment = float(now) if now is not None else _data_clock(rows)
    acc = {lvl: [0.0, 0.0, 0.0] for lvl in LATTICE_LEVELS}
    key_n: dict[str, float] = dict.fromkeys(weights, 0.0)
    for row in rows:
        metric_w = float(weights.get(str(row["metric"]), 0.0))
        raw_n = float(row["n"] or 0.0)
        if metric_w <= 0 or raw_n <= 0 or _is_forensic_key(engine_id, row["symbol"], row["setup"], row["regime"]):
            continue
        n = raw_n * _decay_factor(str(row["updated_at"] or ""), moment, engine_id)
        mean = float(row["ewma"])
        var = max(0.0, float(row["m2"] or 0.0)) / raw_n
        level = _level_of(row, sym, stp, reg)
        w = metric_w * n
        bucket = acc[level]
        bucket[0] += w
        bucket[1] += w * mean
        bucket[2] += w * (var + mean * mean)
        if level == "key":
            key_n[str(row["metric"])] += n
    mu = prior
    level_means: dict[str, float] = {}
    nested_w: dict[str, float] = {}
    for i, level in enumerate(LATTICE_LEVELS):
        inner = LATTICE_LEVELS[i:]
        w = sum(acc[lvl][0] for lvl in inner)
        s = sum(acc[lvl][1] for lvl in inner)
        nested_w[level] = w
        if level == "engine":
            level_means[level] = (PRIOR_STRENGTH * prior + s) / (PRIOR_STRENGTH + w)
            continue
        mu = (PRIOR_STRENGTH * mu + s) / (PRIOR_STRENGTH + w)
        level_means[level] = mu
    total_w = nested_w["engine"]
    sigma2 = 0.0
    if total_w > 0:
        m1 = sum(b[1] for b in acc.values()) / total_w
        sigma2 = max(0.0, sum(b[2] for b in acc.values()) / total_w - m1 * m1)
    relevant = sum(acc[lvl][0] * _LEVEL_RELEVANCE[lvl] for lvl in LATTICE_LEVELS)
    return {
        "mean": mu,
        "prior": prior,
        "parent": level_means["related"],
        "levels": level_means,
        "level_weights": nested_w,
        "key_weight": acc["key"][0],
        "pooled_weight": nested_w["setup"] - acc["key"][0],
        "key_n": key_n,
        "n": sum(key_n.values()),
        "confidence": relevant / (PRIOR_STRENGTH + relevant),
        "sd": math.sqrt(sigma2 / (PRIOR_STRENGTH + relevant)) if sigma2 > 0 else 0.0,
        "version": ADAPTIVE_STATE_VERSION,
    }


def estimate(db_path: str, engine: str, symbol: str, setup: str, regime: str, metric: str, *, now: float | None = None) -> dict[str, Any]:
    """Hierarchical shrunk mean of one metric at one key (see ``_lattice``)."""
    engine_id = str(engine or "").upper()
    prior = _prior(engine_id, metric)
    if engine_id not in _PRIORS:
        return {"mean": prior, "n": 0.0, "prior": prior, "parent": prior, "confidence": 0.0, "sd": 0.0, "version": ADAPTIVE_STATE_VERSION}
    rows = _state_rows(db_path, engine_id, (metric,))
    return _lattice(rows, engine_id=engine_id, symbol=symbol, setup=setup, regime=regime, weights={metric: 1.0}, prior=prior, now=now)


def _tilt(mean: float, prior: float) -> float:
    if prior == 0:
        return 0.0
    return math.tanh((mean - prior) / abs(prior))


# DAY net expectancy reads two separately learned measurements of the same
# economic quantity, net after costs under the live exit contract: realized
# trade net (filled opportunities) and the lifecycle replay of unfilled
# qualified candidates (LIFECYCLE_WEIGHT). An opportunity contributes one or
# the other, never both. Fixed-horizon markouts are diagnostics only: a DAY
# position is not closed at a fixed clock, and the 60m markout misjudged
# setups whose lifecycle runs for hours.
DAY_NET_PARTS: tuple[tuple[str, float], ...] = (("trade_net", 1.0), ("lifecycle_net", LIFECYCLE_WEIGHT))


def day_net_expectancy(db_path: str, symbol: str, setup: str, regime: str, *, now: float | None = None) -> dict[str, Any]:
    """Expected DAY net edge after costs for one key, pooled hierarchically
    (key -> same setup sharing symbol or regime -> setup -> engine -> 0).
    No sample-count floor: one observation moves the posterior by its weight."""
    weights = dict(DAY_NET_PARTS)
    lat = _lattice(
        _state_rows(db_path, DAY_ENGINE, tuple(weights)),
        engine_id=DAY_ENGINE,
        symbol=symbol,
        setup=setup,
        regime=regime,
        weights=weights,
        prior=_prior(DAY_ENGINE, "trade_net"),
        now=now,
    )
    return {
        "mean": lat["mean"],
        "parent": lat["parent"],
        "prior": lat["prior"],
        "levels": lat["levels"],
        "level_weights": lat["level_weights"],
        "n_trade": lat["key_n"]["trade_net"],
        "n_lifecycle": lat["key_n"]["lifecycle_net"],
        "weight": lat["key_weight"],
        "pooled_weight": lat["pooled_weight"],
        "confidence": lat["confidence"],
        "sd": lat["sd"],
    }


def day_size_mult(expected_net: float, risk: float) -> float:
    """Bounded size from expected net per unit of adverse risk (as SCALP sizes
    its final edge). Negative evidence shrinks toward the floor; it never blocks."""
    lo, hi = SIZE_BOUNDS[DAY_ENGINE]
    tilt = math.tanh(float(expected_net) / risk) if risk > 0 else 0.0
    return _clamp(1.0 + 0.30 * tilt, lo, hi)


def day_decision(db_path: str, symbol: str, setup: str, regime: str, *, now: float | None = None) -> dict[str, Any]:
    """What the next DAY candidate reads. Ranking, size, objective and runner only.

    Expected net edge after costs is the rank score and sets the bounded size;
    learned move potential (MFE) sets the objective. Neither removes a candidate.
    """
    from backend.config.trading_economics import canonical_roundtrip_cost_pct

    mfe = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_mfe", now=now)
    mae = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_mae", now=now)
    timing = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_time_to_mfe_min", now=now)
    continuation = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_continuation", now=now)
    forward = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "markout_forward", now=now)
    net = day_net_expectancy(db_path, symbol, setup, regime, now=now)
    cost = canonical_roundtrip_cost_pct()
    size_mult = day_size_mult(net["mean"], max(0.0, mae["mean"]) + cost)
    abstain_flag, abstain_reason = _abstain(net["mean"])
    economic = {
        "engine_id": DAY_ENGINE,
        "economic_version": current_economic_version(DAY_ENGINE),
        "expected_gross": net["mean"] + cost,
        "expected_cost": cost,
        "adaptive_correction": net["mean"] - net["prior"],
        "uncertainty": net["sd"],
        "expected_net_edge": net["mean"],
        "size_effect": size_mult - 1.0,
        "levels": {lvl: round(v, 6) for lvl, v in net["levels"].items()},
        "level_weights": {lvl: round(v, 3) for lvl, v in net["level_weights"].items()},
        "n_trade": round(net["n_trade"], 3),
        "n_lifecycle": round(net["n_lifecycle"], 3),
    }
    return {
        "adaptive_state_version": ADAPTIVE_STATE_VERSION,
        "economic_version": current_economic_version(DAY_ENGINE),
        "engine_id": DAY_ENGINE,
        "symbol": str(symbol or "").upper(),
        "setup": str(setup or "").upper(),
        "regime": str(regime or "").lower(),
        "expected_move": mfe["mean"],
        "expected_move_prior": mfe["prior"],
        "expected_net": net["mean"],
        "expected_net_parent": net["parent"],
        "net_confidence": net["confidence"],
        "confidence": net["confidence"],
        "abstain": abstain_flag,
        "abstain_reason": abstain_reason,
        "abstain_net_edge": net["mean"],
        "abstain_confidence": net["confidence"],
        "abstain_live_veto": False,
        "uncertainty": net["sd"],
        "economic": economic,
        "mfe": mfe["mean"],
        "mae": mae["mean"],
        "time_to_mfe_min": timing["mean"],
        "continuation": continuation["mean"],
        "size_mult": size_mult,
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


def _model_key(engine_id: str, model: str) -> str:
    """Linear models are scoped to the economic version like metric state."""
    return f"{model}@{current_economic_version(engine_id)}"


def _load_linear(conn: sqlite3.Connection, engine_id: str, model: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT payload FROM adaptive_linear_model WHERE engine_id=? AND model=?",
        (engine_id, _model_key(engine_id, model)),
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


def update_linear_model(db_path: str, engine: str, model: str, features: dict | None, target: float, *, now: float | None = None) -> None:
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
            (engine_id, _model_key(engine_id, model), json.dumps(st, separators=(",", ":")), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time()))),
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


def learn_micro_weight(db_path: str, *, strategy_version: str, tilt: float, miss: float, now: float | None = None) -> bool:
    """Fold one resolved claim into the micro-weight moments.

    ``tilt`` is the decision-time micro term before the weight; ``miss`` is the
    realized label minus the executable edge before micro (what micro had to
    explain). A zero tilt carries no information and is skipped.
    """
    t, e = float(tilt), float(miss)
    if t == 0.0 or not (math.isfinite(t) and math.isfinite(e)):
        return False
    sym, stp, reg = MICRO_WEIGHT_KEY
    wrote = False
    for metric, value in zip(MICRO_WEIGHT_METRICS, (t * t, t * e), strict=True):
        wrote = observe(db_path, engine=SCALP_ENGINE, symbol=sym, setup=stp, regime=reg, metric=metric, value=value, strategy_version=strategy_version, now=now) or wrote
    return wrote


def micro_weight(db_path: str, *, now: float | None = None) -> dict[str, float]:
    """Learned weight on the micro term, in [0, 1], prior 1.

    Ridge regression of ``miss`` on ``tilt`` toward slope 1: the micro model
    keeps its full weight until resolved claims show its tilt does not explain
    what the edge before micro missed, then the weight falls continuously,
    down to 0 when the tilt points the wrong way. It can never amplify.
    """
    rows = [r for r in _state_rows(db_path, SCALP_ENGINE, MICRO_WEIGHT_METRICS) if (str(r["symbol"]), str(r["setup"]), str(r["regime"])) == MICRO_WEIGHT_KEY]
    moment = float(now) if now is not None else _data_clock(rows)
    stats = {str(r["metric"]): (float(r["n"] or 0.0) * _decay_factor(str(r["updated_at"] or ""), moment, SCALP_ENGINE), float(r["ewma"])) for r in rows}
    n_tt, m_tt = stats.get("micro_tilt_sq", (0.0, 0.0))
    n_te, m_te = stats.get("micro_tilt_x_miss", (0.0, 0.0))
    info = PRIOR_STRENGTH * MICRO_WEIGHT_PRIOR_VAR
    sum_tt = n_tt * max(0.0, m_tt)
    sum_te = n_te * m_te
    return {"weight": _clamp((info + sum_te) / (info + sum_tt), 0.0, 1.0), "n": n_tt, "slope_raw": (sum_te / sum_tt) if sum_tt > 0 else 1.0}


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


def scalp_claim_calibration(db_path: str, symbol: str, setup: str, regime: str, *, now: float | None = None) -> dict[str, Any]:
    """How much of a strategy claim's base edge shows up in realized net.

    The mean residual alone assumes realized net moves 1:1 with the claimed base
    edge. ``claim_capture`` = 1 + the within-key slope of residual on base edge,
    estimated down the same within-setup hierarchy as every other estimate
    (setup -> related -> key). At each level the slope is a ridge estimate toward
    its parent slope with ``PRIOR_STRENGTH`` claims of variance ``CLAIM_SLOPE_PRIOR_VAR``
    as the parent's weight; the top parent is slope 0 (capture 1). Kept in
    [0, 1]. ``claim_base_mean`` is the hierarchical mean claimed base edge, the
    point where the slope pivots. Cold: capture 1, no change to the edge.
    """
    sym, stp, reg = _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower()
    claim_rows = _state_rows(db_path, SCALP_ENGINE, CLAIM_MOMENT_METRICS)
    moment = float(now) if now is not None else _data_clock(claim_rows)
    keys: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in claim_rows:
        if _is_forensic_key(SCALP_ENGINE, row["symbol"], row["setup"], row["regime"]):
            continue
        rec = keys.setdefault((_norm_symbol(row["symbol"]), str(row["setup"] or "").upper(), str(row["regime"] or "").lower()), {"row": row})
        rec[str(row["metric"])] = float(row["ewma"])
        rec["n"] = min(rec.get("n", math.inf), float(row["n"] or 0.0) * _decay_factor(str(row["updated_at"] or ""), moment, SCALP_ENGINE))
    acc = {lvl: [0.0, 0.0, 0.0, 0.0] for lvl in LATTICE_LEVELS}
    for rec in keys.values():
        if any(m not in rec for m in CLAIM_MOMENT_METRICS) or rec["n"] <= 0:
            continue
        n = rec["n"]
        b = rec["claim_base_edge"]
        bucket = acc[_level_of(rec["row"], sym, stp, reg)]
        bucket[0] += n
        bucket[1] += n * b
        bucket[2] += n * (rec["claim_base_edge_x_residual"] - b * rec["claim_residual"])
        bucket[3] += n * max(0.0, rec["claim_base_edge_sq"] - b * b)
    slope_info = PRIOR_STRENGTH * CLAIM_SLOPE_PRIOR_VAR
    slope = 0.0
    center: float | None = None
    nested_w: dict[str, float] = {}
    for i, level in enumerate(LATTICE_LEVELS):
        w, sb, cov, var = (sum(acc[lvl][j] for lvl in LATTICE_LEVELS[i:]) for j in range(4))
        nested_w[level] = w
        if level == "engine":
            continue
        slope = (slope_info * slope + cov) / (slope_info + var)
        if w > 0:
            center = sb / w if center is None else (PRIOR_STRENGTH * center + sb) / (PRIOR_STRENGTH + w)
    return {
        "claim_capture": _clamp(1.0 + slope, 0.0, 1.0),
        "claim_slope": slope,
        "claim_base_mean": center if center is not None else 0.0,
        "claim_key_n": acc["key"][0],
        "n_claim": nested_w["setup"],
        "claim_level_weights": nested_w,
    }


def _blend(parts: list[tuple[float, float]], prior: float) -> tuple[float, float]:
    weighted = [(mean, weight) for mean, weight in parts if weight > 0]
    if not weighted:
        return prior, 0.0
    weight = sum(item[1] for item in weighted)
    mean = (PRIOR_STRENGTH * prior + sum(m * w for m, w in weighted)) / (PRIOR_STRENGTH + weight)
    return mean, weight


def scalp_decision(db_path: str, symbol: str, setup: str, regime: str, features: dict | None = None, *, now: float | None = None) -> dict[str, Any]:
    """Learned adjustments for the next SCALP candidate. Never an edge by itself.

    The candidate's own base executable edge (raw expected move - live cost) is
    built in scalp_v2.executable_edge. This view supplies the hierarchical mean
    residual, the claim calibration (capture slope and pivot) and the weighted
    microstructure residual added to it, plus confidence, risk, target and hold.
    A cold key contributes a residual of exactly 0.
    """
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_TIME_STOP_MIN

    residual = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "edge_residual", now=now)
    residual_strategy = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "edge_residual_strategy", now=now)
    net = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_net", now=now)
    mfe = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_mfe", now=now)
    path_mae = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "markout_mae", now=now)
    timing = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "trade_time_to_mfe_min", now=now)
    forward = estimate(db_path, SCALP_ENGINE, symbol, setup, regime, "markout_forward", now=now)
    learned_mfe, _ = _blend([(mfe["mean"], mfe["n"]), (max(0.0, forward["mean"]), forward["n"] * MARKOUT_WEIGHT)], mfe["prior"])
    hard_hold = float(SCALP_V2_TIME_STOP_MIN)
    confidence = float(residual["confidence"])
    adaptive_residual = _clamp(residual["mean"], -SCALP_RESIDUAL_MAX, SCALP_RESIDUAL_MAX)
    # Bounded microstructure residual: shrunk by the model's own sample count so
    # a cold model barely moves the edge, then by the learned micro weight.
    # Zero-centred (excludes the model bias).
    micro_tilt, micro_n = micro_edge_tilt(db_path, SCALP_ENGINE, features)
    micro_conf = micro_n / (micro_n + MICRO_MODEL_CONF_K) if micro_n > 0 else 0.0
    micro_unweighted = micro_conf * micro_tilt
    weight = micro_weight(db_path, now=now)
    micro_residual = micro_unweighted * weight["weight"]
    claim = scalp_claim_calibration(db_path, symbol, setup, regime, now=now)
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
        "residual_uncertainty": float(residual_strategy["sd"]),
        "residual_levels": {lvl: round(v, 6) for lvl, v in residual_strategy.get("levels", {}).items()},
        "economic_version": current_economic_version(SCALP_ENGINE),
        "micro_residual": micro_residual,
        "micro_residual_unweighted": micro_unweighted,
        "micro_weight": weight["weight"],
        "micro_weight_n": weight["n"],
        "micro_tilt": round(micro_residual, 6),
        "micro_tilt_raw": round(micro_tilt, 6),
        "micro_model_n": micro_n,
        "claim_capture": claim["claim_capture"],
        "claim_slope": claim["claim_slope"],
        "claim_base_mean": claim["claim_base_mean"],
        "claim_key_n": claim["claim_key_n"],
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
    entered_at: float | None = None,
    now: float | None = None,
) -> bool:
    """Realized-trade update. Dust, non-current versions and positions entered
    before the engine's economic anchor do not move state."""
    if is_dust or not version_current:
        return False
    engine_id = str(engine or "").upper()
    if entered_at is not None and float(entered_at) < anchor_epoch(engine_id):
        return False
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
        wrote = observe(db_path, engine=engine_id, symbol=symbol, setup=setup, regime=regime, metric=metric, value=float(value), strategy_version=strategy_version, now=now) or wrote
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
            state_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(adaptive_metric_state)")}
            if (
                "economic_version" in state_cols
                and conn.execute(
                    "SELECT 1 FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND metric='trade_net' LIMIT 1",
                    (DAY_ENGINE, current_economic_version(DAY_ENGINE)),
                ).fetchone()
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


def _economic_of(row: sqlite3.Row) -> dict[str, Any]:
    try:
        econ = json.loads(row["economic_json"] or "{}")
    except (TypeError, ValueError, IndexError):
        return {}
    return econ if isinstance(econ, dict) else {}


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
    """Fold separator-symbol duplicates onto the normalized key."""
    rows = list(conn.execute("SELECT rowid, * FROM adaptive_metric_state WHERE symbol LIKE '%/%' OR symbol LIKE '%-%'"))
    for row in rows:
        norm = _norm_symbol(row["symbol"])
        clash = conn.execute(
            "SELECT rowid FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND symbol=? AND setup=? AND regime=? AND metric=?",
            (row["engine_id"], row["economic_version"], norm, row["setup"], row["regime"], row["metric"]),
        ).fetchone()
        if clash is None:
            conn.execute("UPDATE adaptive_metric_state SET symbol=? WHERE rowid=?", (norm, row["rowid"]))
        else:
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
    lifecycle: Any = None,
    economic: dict | None = None,
    opportunity_id: str = "",
) -> int | None:
    """Store one decision-time candidate for causal forward markouts. Returns the row id.

    ``lifecycle`` (DAY ``LifecycleParams``) makes the row a lifecycle candidate:
    the live exit contract is replayed from its decision ask once bars exist.
    ``economic`` is the decision-time expected-net-edge breakdown, persisted for
    audit and, for SCALP, the micro-weight label. ``opportunity_id`` groups
    repeated signals of one DAY opportunity; only its first record is learnable.
    """
    engine_id = str(engine or "").upper()
    version = current_strategy_version(engine_id)
    if not version or float(ref_price or 0) <= 0 or not str(setup or "").strip():
        return None
    feats_json = _stored_features(features)
    horizon = _label_horizon_for(db_path, engine_id, symbol, setup, regime)
    raw_move: float | None = None
    with contextlib.suppress(TypeError, ValueError):
        raw_move = float(raw_expected_move) if raw_expected_move is not None and float(raw_expected_move) > 0 else None
    lifecycle_json = ""
    if lifecycle is not None:
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            lifecycle_json = lifecycle.to_json() if hasattr(lifecycle, "to_json") else json.dumps(dict(lifecycle), separators=(",", ":"))
    economic_json = json.dumps(economic, separators=(",", ":"), default=str) if isinstance(economic, dict) else "{}"
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO adaptive_candidate_markouts (
                engine_id, symbol, setup, regime, strategy_version, signaled,
                ref_price, roundtrip_cost, evaluated_at, features_json, label_horizon,
                raw_expected_move, raw_move_source, candidate_state,
                economic_version, lifecycle_json, economic_json, opportunity_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                current_economic_version(engine_id),
                lifecycle_json,
                economic_json,
                str(opportunity_id or ""),
            ),
        )
        conn.commit()
        return int(cur.lastrowid) if cur.lastrowid is not None else None


def mark_candidate_filled(db_path: str, row_id: int | None) -> bool:
    """Flag a recorded candidate as filled: its realized close carries the
    opportunity, so the opportunity's lifecycle label is never also learned."""
    if not row_id:
        return False
    try:
        with _connect(db_path) as conn:
            cur = conn.execute("UPDATE adaptive_candidate_markouts SET filled=1 WHERE id=?", (int(row_id),))
            conn.commit()
            return cur.rowcount == 1
    except sqlite3.Error:
        return False


def _opportunity_learnable(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    """First record of an unfilled opportunity under the row's economic version."""
    if int(row["filled"] or 0):
        return False
    opp = str(row["opportunity_id"] or "")
    if not opp:
        return True
    first = conn.execute(
        "SELECT MIN(id), MAX(filled) FROM adaptive_candidate_markouts WHERE engine_id=? AND opportunity_id=? AND economic_version=?",
        (row["engine_id"], opp, row["economic_version"]),
    ).fetchone()
    return first is not None and int(first[0] or 0) == int(row["id"]) and not int(first[1] or 0)


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


SCALP_TICK_HORIZONS_SEC = (1, 5, 10)


def _lifecycle_label(row: sqlite3.Row, *, moment: float, bars_1m: Callable[[str, float, float], list]) -> dict[str, Any]:
    from backend.services.day_v2.lifecycle_sim import DAY_LIFECYCLE_MAX_MIN, LifecycleParams, simulate_lifecycle

    params = LifecycleParams.from_json(row["lifecycle_json"])
    if params is None:
        return {"final": True, "net": None, "reason": "BAD_LIFECYCLE_PARAMS"}
    start = float(params.entry_time)
    end = min(moment, start + DAY_LIFECYCLE_MAX_MIN * 60.0) + 60.0
    try:
        bars = bars_1m(str(row["symbol"]), start, end)
    except Exception:
        bars = []
    return simulate_lifecycle(params, bars, roundtrip_cost=float(row["roundtrip_cost"] or 0), now=moment)


def resolve_markouts(
    db_path: str,
    quote: Callable[[str, float], float | None],
    *,
    now: float | None = None,
    path_low: Callable[[str, float, float], float | None] | None = None,
    bars_1m: Callable[[str, float, float], list] | None = None,
    tick_quote: Callable[[str, float, float], float | None] | None = None,
) -> int:
    """Fill due forward marks and fold the decision-time labels into state.

    Every horizon is stored. The learned fixed-horizon label is the return at
    the horizon stamped when the candidate was recorded, not the best later
    horizon. A row is claimed (learned=1 / lifecycle_learned=1) before the state
    write so a locked retry cannot train the same label twice. Only rows stamped
    with the engine's current economic version are learned. Returns rows newly
    learned on the fixed-horizon label.

    SCALP rows also learn the edge residual (label minus the decision-time base
    executable edge), the claim calibration, the micro weight and the gross path
    MAE over the committed horizon, which is the live risk estimate.
    ``path_low(symbol, start, end)`` supplies the bar-low path when available;
    ``tick_quote(symbol, start, end)`` the last tape print in a window, for the
    1/5/10 s marks where the tape has one.

    DAY rows recorded with lifecycle inputs replay the live exit contract from
    the decision ask over ``bars_1m(symbol, start, end)`` (stored 1m OHLCV by
    default). Once final, the first record of an opportunity that was never
    filled learns ``lifecycle_net``.
    """
    moment = float(now if now is not None else time.time())
    if bars_1m is None:
        from backend.services.day_v2.lifecycle_sim import ohlcv_bars_1m

        def bars_1m(sym: str, start: float, end: float) -> list:
            return ohlcv_bars_1m(db_path, sym, start, end)

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
                ref = float(row["ref_price"])
                cost = float(row["roundtrip_cost"] or 0)
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
                    if price is None or ref <= 0 or not math.isfinite(float(price or 0)):
                        marks[key] = None
                    else:
                        marks[key] = (float(price) - ref) / ref - cost
                if engine_id == SCALP_ENGINE and tick_quote is not None:
                    for horizon in SCALP_TICK_HORIZONS_SEC:
                        key = f"{horizon}s"
                        if key in marks:
                            continue
                        due = float(row["evaluated_at"]) + horizon
                        if moment < due:
                            done = False
                            continue
                        tick = None
                        with contextlib.suppress(Exception):
                            tick = tick_quote(str(row["symbol"]), float(row["evaluated_at"]), due)
                        valid = tick is not None and ref > 0 and math.isfinite(float(tick)) and float(tick) > 0
                        marks[key] = (float(tick) - ref) / ref - cost if valid else None
                if engine_id == DAY_ENGINE and str(row["lifecycle_json"] or "") and not isinstance(marks.get("lifecycle"), dict):
                    life_label = _lifecycle_label(row, moment=moment, bars_1m=bars_1m)
                    if life_label.get("final"):
                        marks["lifecycle"] = {k: life_label.get(k) for k in ("net", "gross", "reason", "minutes", "mfe", "mae", "censored")}
                    else:
                        done = False
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
                version_ok = str(row["strategy_version"]) == current_strategy_version(engine_id) and str(row["economic_version"] or "") == current_economic_version(engine_id)
                learn_fixed = forward is not None and not row_learned and version_ok and learnable
                if learn_fixed:
                    # Claim first. A failed state write must not re-train this row.
                    cur = conn.execute(
                        "UPDATE adaptive_candidate_markouts SET markouts_json=?, learned=1, resolved=? WHERE id=? AND learned=0",
                        (json.dumps(marks), 1 if done else 0, row["id"]),
                    )
                    conn.commit()
                    learn_fixed = cur.rowcount == 1
                if learn_fixed:
                    observe(
                        db_path,
                        engine=engine_id,
                        symbol=row["symbol"],
                        setup=row["setup"],
                        regime=row["regime"],
                        metric="markout_forward",
                        value=forward,
                        strategy_version=str(row["strategy_version"]),
                        now=moment,
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
                            now=moment,
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
                            now=moment,
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
                                now=moment,
                            )
                    if engine_id == SCALP_ENGINE and micro_target is not None:
                        with contextlib.suppress(Exception):
                            feats = json.loads(row["features_json"] or "{}")
                            if isinstance(feats, dict) and feats:
                                update_linear_model(db_path, SCALP_ENGINE, "micro_edge", feats, micro_target, now=moment)
                        econ = _economic_of(row)
                        if "micro_unweighted" in econ and "pre_micro_edge" in econ:
                            learn_micro_weight(
                                db_path,
                                strategy_version=str(row["strategy_version"]),
                                tilt=float(econ["micro_unweighted"]),
                                miss=float(forward) - float(econ["pre_micro_edge"]),
                                now=moment,
                            )
                    learned += 1
                life = marks.get("lifecycle")
                if (
                    engine_id == DAY_ENGINE
                    and isinstance(life, dict)
                    and life.get("net") is not None
                    and version_ok
                    and state in LEARNABLE_DAY_STATES
                    and not int(row["lifecycle_learned"] or 0)
                    and _opportunity_learnable(conn, row)
                ):
                    cur = conn.execute("UPDATE adaptive_candidate_markouts SET lifecycle_learned=1 WHERE id=? AND lifecycle_learned=0", (row["id"],))
                    conn.commit()
                    if cur.rowcount == 1:
                        observe(
                            db_path,
                            engine=DAY_ENGINE,
                            symbol=row["symbol"],
                            setup=row["setup"],
                            regime=row["regime"],
                            metric="lifecycle_net",
                            value=float(life["net"]),
                            strategy_version=str(row["strategy_version"]),
                            now=moment,
                        )
                conn.execute(
                    "UPDATE adaptive_candidate_markouts SET markouts_json=?, resolved=? WHERE id=?",
                    (json.dumps(marks), 1 if done else 0, row["id"]),
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


def _ohlc_rows(conn: sqlite3.Connection, symbol: str, interval: str, limit: int, as_of: float | None = None) -> list[tuple[float, float, float, float]]:
    """Newest-last (ts asc) high/low/close for a symbol+interval, across name variants.

    ``as_of`` restricts to bars opened before that epoch (historical replay).
    Live reads carry no bound: ``ts`` is DATETIME (numeric affinity), so a
    numeric-looking sentinel would compare below every text timestamp.
    """
    raw = str(symbol or "").upper().replace("-", "").replace("/", "")
    bound = () if as_of is None else (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(as_of))),)
    sql = f"SELECT ts, high, low, close FROM feature_ohlcv WHERE symbol=? AND interval=?{' AND ts<?' if bound else ''} ORDER BY ts DESC LIMIT ?"
    for variant in (raw, raw.replace("USDT", "-USDT"), raw.replace("USDT", "/USDT")):
        try:
            rows = conn.execute(sql, (variant, interval, *bound, int(limit))).fetchall()
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


def _trend_bucket(conn: sqlite3.Connection, symbol: str, interval: str, lookback: int, as_of: float | None = None) -> str:
    rows = _ohlc_rows(conn, symbol, interval, lookback, as_of)
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


def _vol_bucket(conn: sqlite3.Connection, symbol: str, as_of: float | None = None) -> str:
    rows = _ohlc_rows(conn, symbol, "15m", 30, as_of)
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


def market_regime_tag(db_path: str, symbol: str, as_of: float | None = None) -> str:
    """Coarse, inspectable regime key: BTC 1h trend + this symbol's 15m vol bucket.

    Returns e.g. 'btcup_volhi'. Empty string when data is missing, so callers
    fall back to whatever regime string they already had (never crashes, never
    gates). Read/write alignment is preserved because the tag is stamped on the
    entry decision and reused at close. ``as_of`` evaluates it from bars opened
    before that epoch (historical replay); live callers read the newest bars.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5)
    except sqlite3.Error:
        return ""
    try:
        btc = _trend_bucket(conn, "BTCUSDT", "1h", 24, as_of)
        vol = _vol_bucket(conn, symbol, as_of)
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
        "half_life_days_by_engine": {DAY_ENGINE: half_life_days(DAY_ENGINE), SCALP_ENGINE: half_life_days(SCALP_ENGINE)},
        "economic_versions": {DAY_ENGINE: current_economic_version(DAY_ENGINE), SCALP_ENGINE: current_economic_version(SCALP_ENGINE)},
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
                "SELECT DISTINCT symbol, setup, regime FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND setup != ? ORDER BY symbol, setup, regime",
                (engine_id, current_economic_version(engine_id), MICRO_WEIGHT_KEY[1]),
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
                            "uncertainty": round(view["uncertainty"], 6),
                            "size_mult": round(view["size_mult"], 4),
                            "objective_atr_mult": round(view["objective_atr_mult"], 4),
                            "confidence": round(view["confidence"], 3),
                            "net_confidence": round(view["net_confidence"], 3),
                            "abstain": view["abstain"],
                            "n": view["n_mfe"],
                            "n_trade_net": view["n_trade_net"],
                            "n_lifecycle": view["economic"]["n_lifecycle"],
                            "n_forward": view["n_forward"],
                            "levels": view["economic"]["levels"],
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
                            "residual_uncertainty": round(view["residual_uncertainty"], 6),
                            "claim_capture": round(view["claim_capture"], 4),
                            "claim_base_mean": round(view["claim_base_mean"], 6),
                            "micro_weight": round(view["micro_weight"], 4),
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
            "learned_weight": {k: round(float(v), 4) for k, v in micro_weight(db_path).items()},
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
                "SELECT DISTINCT symbol, setup, regime FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND setup != ? ORDER BY symbol, setup, regime",
                (engine_id, current_economic_version(engine_id), MICRO_WEIGHT_KEY[1]),
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
    short (evidence, never learned). BLOCKED rows qualified but hit a capacity
    rule. Other qualified rows are SELECTED when filled (or the same bar's DAY
    decision was FILLED) and REJECTED otherwise. Values are net of the stamped
    round-trip cost from the decision-time executable price; ``lifecycle`` is
    the live exit contract replayed from that price.
    """
    since = time.time() - float(window_days) * 86400.0
    version = current_strategy_version(DAY_ENGINE)
    out: dict[str, Any] = {
        "strategy_version": version,
        "economic_version": current_economic_version(DAY_ENGINE),
        "window_days": float(window_days),
        "horizons_min": list(DAY_HORIZONS_MIN),
        "states": {},
    }
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
        filled_col = "filled" if "filled" in cols else "0"
        rows = conn.execute(
            f"SELECT symbol, setup, evaluated_at, markouts_json, {state_col} AS state, {filled_col} AS filled "
            "FROM adaptive_candidate_markouts WHERE engine_id=? AND strategy_version=? AND evaluated_at>=?",
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
        if state == CANDIDATE_QUALIFIED_BLOCKED:
            state = "BLOCKED"
        elif state != CANDIDATE_NEAR_QUALIFIED:
            result = results.get((_norm_symbol(row["symbol"]), float(row["evaluated_at"])), "")
            state = "SELECTED" if (int(row["filled"] or 0) or result == "FILLED") else "REJECTED"
        try:
            marks = json.loads(row["markouts_json"] or "{}")
        except (TypeError, ValueError):
            continue
        marks = marks if isinstance(marks, dict) else {}
        per_h = agg.setdefault(state, {}).setdefault(str(row["setup"] or ""), {})
        for h in DAY_HORIZONS_MIN:
            value = _mark_at(marks, h)
            if value is not None:
                per_h.setdefault(str(h), []).append(value)
        life = marks.get("lifecycle")
        if isinstance(life, dict) and life.get("net") is not None:
            per_h.setdefault("lifecycle", []).append(float(life["net"]))
    for state, setups in agg.items():
        out["states"][state] = {
            setup: {
                h: {
                    "n": len(v),
                    "mean": round(sum(v) / len(v), 6),
                    "median": round(sorted(v)[len(v) // 2], 6),
                    "positive_rate": round(sum(1 for x in v if x > 0) / len(v), 3),
                }
                for h, v in sorted(per_h.items(), key=lambda kv: (kv[0] == "lifecycle", float(kv[0]) if kv[0] != "lifecycle" else 0.0))
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
    "SCALP_TICK_HORIZONS_SEC",
    "abstention_report",
    "adaptive_state_report",
    "calibration_report",
    "continuation_ratio",
    "current_economic_version",
    "day_candidate_markout_report",
    "day_decision",
    "day_net_expectancy",
    "day_size_mult",
    "estimate",
    "learn_claim_label",
    "learn_from_close",
    "learn_micro_weight",
    "mark_candidate_filled",
    "market_regime_tag",
    "micro_edge_tilt",
    "micro_weight",
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
