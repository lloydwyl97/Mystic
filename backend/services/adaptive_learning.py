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
``lifecycle_net`` (the market path replayed from the decision ask to
catastrophic protection or the label censor) and the fixed-horizon
``markout_*`` diagnostics. SCALP claims update the claim calibration. Both are
market opportunity. ``policy_gap`` is what Mystic's own exit policy realized on
a filled opportunity minus that opportunity's market label; adding it turns
market opportunity into expected realized result under the policy. DAY net
expectancy sets bounded size and rank, the SCALP executable edge sets
eligibility, rank and size, and both are in policy units.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import sqlite3
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from backend.services.scalp_v2.raw_move_source import STRATEGY_CLAIM, is_directional, normalize_raw_move_source
from backend.services.strategy_version import ADAPTIVE_STATE_VERSION, adaptive_format, current_lifecycle_label, economic_anchor, economic_version, engine_versions, exit_contract_of

PRIOR_STRENGTH = 8.0
EWMA_ALPHA = 0.25
MARKOUT_WEIGHT = 0.35
# DAY lifecycle labels replay the market path on 1m bars without a learned
# continuation terminal, so they hold to catastrophic protection or the censor.
# Entry reads them after the learned policy gap moves them into policy units.
# They count slightly below a realized close so simulated evidence never
# outweighs the same amount of fills.
LIFECYCLE_WEIGHT = 0.75
# Closes before the current calibration learner was live are not replayed into
# it. Later compatible closes with a stored prediction and no calibration
# observation are. This bounds the repair; it is not a sample-size gate.
CALIBRATION_BACKFILL_FROM = "2026-10-07T15:25:25Z"

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
        # Realized net of a filled opportunity minus its own lifecycle label.
        "policy_gap": 0.0,
        # Realized policy net minus the policy value predicted at entry.
        # Prior 0. Losses pull the next forecast down; wins can pull it up.
        "policy_calibration": 0.0,
        "hold_remaining_up": 0.0,
        "hold_remaining_down": 0.0,
        "markout_forward": 0.0,
        "markout_mae": 0.006,
    },
    SCALP_ENGINE: {
        "trade_mfe": 0.0025,
        "trade_mae": 0.0015,
        "trade_net": 0.0015,
        # Realized net of a filled claim minus its forward net markout at the
        # claim's label horizon.
        "policy_gap": 0.0,
        # Realized policy net minus the policy value predicted at entry.
        # Prior 0. Losses pull the next forecast down; wins can pull it up.
        "policy_calibration": 0.0,
        "hold_remaining_up": 0.0,
        "hold_remaining_down": 0.0,
        "trade_time_to_mfe_min": 8.0,
        "markout_forward": 0.0015,
        # Gross adverse price excursion of the candidate path over its committed
        # horizon (SCALP). This is the live risk estimate.
        "markout_mae": 0.0015,
        # Realized net markout minus (raw projection - cost): the bias of the raw
        # strategy projection, per raw-move source. Diagnostic only; the edge
        # reads the claim calibration below.
        "edge_residual": 0.0,
        "edge_residual_strategy": 0.0,
        # Decayed moments of (strategy projection, realized gross move) for the
        # claim calibration. Gross = net markout + the decision-time cost. Zero
        # priors: no move is expected and the projection carries no weight
        # until resolved claims show it predicts the move.
        "claim_raw": 0.0,
        "claim_raw_sq": 0.0,
        "claim_gross": 0.0,
        "claim_raw_x_gross": 0.0,
        # Decayed moments of (decision-time micro tilt, what the edge before micro
        # missed) for the learned micro weight.
        "micro_tilt_sq": 0.0,
        "micro_tilt_x_miss": 0.0,
    },
}

# Continuation horizons, in seconds. Priors are 0: no hold and no exit is assumed.
# These metrics are not read by entry ranking or sizing.
HOLD_ADVANTAGE_HORIZONS = (30, 60, 120, 300, 600, 900, 1200, 1800, 3600, 7200, 14400, 21600, 43200)
for _engine_priors in _PRIORS.values():
    for _horizon in HOLD_ADVANTAGE_HORIZONS:
        _engine_priors[f"hold_adv_{_horizon}"] = 0.0

CLAIM_MOMENT_METRICS = ("claim_raw", "claim_raw_sq", "claim_gross", "claim_raw_x_gross")
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
# A representable market state the handcrafted detector did not fire. It is a
# separate learned population, not a funding ban: cold, its expectancy is the
# neutral prior, and capital follows only once that expectancy is a positive net.
DAY_MARKET_STATE_SUFFIX = "__MARKET"

# Bound on the SCALP key's learned gross move against its setup's (the adaptive
# residual). Equal to the claim cap.
SCALP_RESIDUAL_MAX = 0.006


def residual_metric(raw_move_source: str | None) -> str:
    """Residual key per raw-move source. Both are forensic (realized net minus the
    claim taken at face value); eligibility reads the claim calibration instead.
    ``edge_residual`` (ATR-estimate rows) never touches a strategy claim."""
    return "edge_residual_strategy" if is_directional(raw_move_source) else "edge_residual"


# Metrics folded as a decayed running mean (weight 1/n) instead of the fast EWMA.
# Causal calibration on current-version SCALP markouts: a 0.25 EWMA residual
# tracks the last few labels and produced 4x more positive predictions with no
# better realization; the running mean converges to the key's actual bias.
_HOLD_ADVANTAGE_METRICS = tuple(f"hold_adv_{horizon}" for horizon in HOLD_ADVANTAGE_HORIZONS)
MEAN_FORM_METRICS = frozenset(
    {
        "edge_residual",
        "edge_residual_strategy",
        "lifecycle_net",
        "trade_net",
        "policy_gap",
        "policy_calibration",
        "hold_remaining_up",
        "hold_remaining_down",
        *CLAIM_MOMENT_METRICS,
        *MICRO_WEIGHT_METRICS,
        *_HOLD_ADVANTAGE_METRICS,
    }
)

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


def _epoch_any(value: Any) -> float | None:
    """Epoch seconds from an epoch number or an ISO timestamp (naive is UTC)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = None
    if number is not None and math.isfinite(number):
        return number
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


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
        ("realized_net", "REAL"),
        ("policy_learned", "INTEGER NOT NULL DEFAULT 0"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE adaptive_candidate_markouts ADD COLUMN {col} {ddl}")
    if "calibration_learned" not in cols:
        conn.execute("ALTER TABLE adaptive_candidate_markouts ADD COLUMN calibration_learned INTEGER NOT NULL DEFAULT 0")
        # Rows that finished the old combined step are not calibration candidates
        # for the close-time backfill. A flag with no metric row is repaired
        # later, one key at a time, and is not treated as already observed.
        conn.execute("UPDATE adaptive_candidate_markouts SET calibration_learned=1 WHERE policy_learned=1 AND calibration_learned=0")
        conn.commit()
    # resolve_markouts runs every SCALP cycle; without these, its key repair and
    # unresolved scan are full table scans that grow with the markout history.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_key ON adaptive_candidate_markouts(engine_id, symbol, setup, regime, learned)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_unresolved ON adaptive_candidate_markouts(resolved, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_opportunity ON adaptive_candidate_markouts(engine_id, opportunity_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adaptive_markouts_symbol_time ON adaptive_candidate_markouts(engine_id, symbol, evaluated_at)")
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
    weight: float = 1.0,
) -> bool:
    """Fold one current-version observation into the key under the current economic version.

    Returns False for another strategy version, an unknown metric, a value that
    is not finite or a non-positive weight. ``now`` is the observation time
    (defaults to wall clock); a chronological rebuild passes each label's own
    time so decay matches live learning. ``weight`` is the observation's share
    of one independent sample (mean-form metrics). The decayed second moment
    ``m2`` feeds the reported uncertainty.
    """
    engine_id = str(engine or "").upper()
    if engine_id not in _PRIORS or metric not in _PRIORS[engine_id]:
        return False
    if str(strategy_version or "") != current_strategy_version(engine_id):
        return False
    if value is None or not math.isfinite(float(value)):
        return False
    with _connect(db_path) as conn:
        folded = _fold_observation(
            conn,
            engine=engine,
            symbol=symbol,
            setup=setup,
            regime=regime,
            metric=metric,
            value=value,
            strategy_version=strategy_version,
            now=now,
            weight=weight,
        )
        if folded:
            conn.commit()
    return folded


def _fold_observation(
    conn: sqlite3.Connection,
    *,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    metric: str,
    value: float,
    strategy_version: str,
    now: float | None = None,
    weight: float = 1.0,
) -> bool:
    """Fold one observation on ``conn`` without committing.

    Callers own the transaction. ``observe`` commits its own connection.
    The close-time calibration learner commits this write together with the
    learned flag."""
    engine_id = str(engine or "").upper()
    if engine_id not in _PRIORS or metric not in _PRIORS[engine_id]:
        return False
    if str(strategy_version or "") != current_strategy_version(engine_id):
        return False
    if value is None or not math.isfinite(float(value)):
        return False
    w = float(weight) if metric in MEAN_FORM_METRICS else 1.0
    if not (math.isfinite(w) and w > 0):
        return False
    value = float(value)
    key = (engine_id, current_economic_version(engine_id), _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower(), metric)
    moment = float(now if now is not None else time.time())
    row = conn.execute(
        "SELECT n, ewma, m2, updated_at FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND symbol=? AND setup=? AND regime=? AND metric=?",
        key,
    ).fetchone()
    if row is None or float(row["n"]) <= 0:
        n, mean, m2 = w, value, 0.0
    else:
        # Age out the prior sample count so stale evidence stops dominating,
        # then fold in the new observation at its weight.
        decay = _decay_factor(str(row["updated_at"] or ""), moment, engine_id)
        n = float(row["n"]) * decay + w
        alpha = w / n if metric in MEAN_FORM_METRICS else EWMA_ALPHA
        delta = value - float(row["ewma"])
        mean = float(row["ewma"]) + alpha * delta
        m2 = max(0.0, float(row["m2"] or 0.0)) * decay + w * delta * (value - mean)
    conn.execute(
        """
        INSERT INTO adaptive_metric_state (engine_id, economic_version, symbol, setup, regime, metric, n, ewma, m2, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(engine_id, economic_version, symbol, setup, regime, metric) DO UPDATE SET
            n=excluded.n, ewma=excluded.ewma, m2=excluded.m2, updated_at=excluded.updated_at
        """,
        (*key, n, mean, max(0.0, m2), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))),
    )
    return True


def _claim_and_fold(
    conn: sqlite3.Connection,
    *,
    row_id: int,
    flag: str,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    metric: str,
    value: float,
    strategy_version: str,
    moment: float,
) -> bool:
    """Set one learned flag and fold its observation in a single savepoint.

    A crash before release leaves the flag unset and the observation unwritten.
    A later retry writes both once. Releasing the outermost savepoint commits
    both; a nested savepoint waits for the caller's commit, which then persists
    both or neither."""
    if flag == "calibration_learned":
        claim = "UPDATE adaptive_candidate_markouts SET calibration_learned=1 WHERE id=? AND calibration_learned=0"
    elif flag == "policy_learned":
        claim = "UPDATE adaptive_candidate_markouts SET policy_learned=1 WHERE id=? AND policy_learned=0"
    else:
        return False
    conn.execute("SAVEPOINT mystic_learn")
    try:
        cur = conn.execute(claim, (int(row_id),))
        if cur.rowcount != 1:
            conn.execute("ROLLBACK TO mystic_learn")
            conn.execute("RELEASE mystic_learn")
            return False
        folded = _fold_observation(
            conn,
            engine=engine,
            symbol=symbol,
            setup=setup,
            regime=regime,
            metric=metric,
            value=value,
            strategy_version=strategy_version,
            now=moment,
        )
        if not folded:
            conn.execute("ROLLBACK TO mystic_learn")
            conn.execute("RELEASE mystic_learn")
            return False
        conn.execute("RELEASE mystic_learn")
        return True
    except Exception:
        conn.execute("ROLLBACK TO mystic_learn")
        conn.execute("RELEASE mystic_learn")
        raise


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
    """Innermost level of ``row`` for the key. A blank regime is unknown, not a
    regime: it never matches, so blank rows count at coin + setup and setup only."""
    if str(row["setup"] or "").upper() != stp:
        return "engine"
    same_sym = _norm_symbol(row["symbol"]) == sym
    same_reg = bool(reg) and str(row["regime"] or "").lower() == reg
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
    include_engine: bool = False,
) -> dict[str, Any]:
    """Hierarchical posterior mean of a (weighted) metric set at one key.

    Evidence is decayed sample count times the metric weight. Starting from the
    prior, each nested level from SETUP to KEY is the mean of all evidence
    inside it, shrunk toward the level above by ``PRIOR_STRENGTH / (PRIOR_STRENGTH + W)``.
    With no evidence at a level the parent passes through unchanged. There is no
    sample floor. The ENGINE level moves the posterior only when
    ``include_engine`` is set. Entry estimates leave it off.
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
        if level == "engine" and not include_engine:
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


def estimate(
    db_path: str,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    metric: str,
    *,
    now: float | None = None,
    include_engine: bool = False,
) -> dict[str, Any]:
    """Hierarchical shrunk mean of one metric at one key (see ``_lattice``).

    ``include_engine`` lets continuation borrow across setups of the same
    engine. Entry estimates leave it false, so one setup's trades never move
    another setup's expected net.
    """
    engine_id = str(engine or "").upper()
    prior = _prior(engine_id, metric)
    if engine_id not in _PRIORS:
        return {"mean": prior, "n": 0.0, "prior": prior, "parent": prior, "confidence": 0.0, "sd": 0.0, "version": ADAPTIVE_STATE_VERSION}
    rows = _state_rows(db_path, engine_id, (metric,))
    return _lattice(rows, engine_id=engine_id, symbol=symbol, setup=setup, regime=regime, weights={metric: 1.0}, prior=prior, now=now, include_engine=include_engine)


def _tilt(mean: float, prior: float) -> float:
    if prior == 0:
        return 0.0
    return math.tanh((mean - prior) / abs(prior))


# DAY net expectancy is the expected realized net after costs under Mystic's
# live policy. Realized trade net (filled opportunities) is already in those
# units. The lifecycle replay of unfilled qualified candidates
# (LIFECYCLE_WEIGHT) is market opportunity, so each of its rows enters shifted
# by the learned policy gap. An opportunity contributes one or the other, never
# both. Fixed-horizon markouts are diagnostics only: a DAY position is not
# closed at a fixed clock, and the 60m markout misjudged setups whose lifecycle
# runs for hours.
DAY_NET_PARTS: tuple[tuple[str, float], ...] = (("trade_net", 1.0), ("lifecycle_net", LIFECYCLE_WEIGHT))


def policy_gap(db_path: str, engine: str, symbol: str, setup: str, regime: str, *, now: float | None = None, rows: list | None = None) -> dict[str, Any]:
    """Learned realized-minus-market gap of the live policy at one key.

    The exit policy is shared by every setup of the engine, so the estimate
    borrows engine-wide evidence. Prior 0: a cold engine is priced at its market
    label. Fee-losing fills push it down and fills that beat their market label
    push it up.
    """
    engine_id = str(engine or "").upper()
    if rows is None:
        rows = _state_rows(db_path, engine_id, ("policy_gap",))
    gap_rows = [r for r in rows if str(r["metric"]) == "policy_gap"]
    return _lattice(gap_rows, engine_id=engine_id, symbol=symbol, setup=setup, regime=regime, weights={"policy_gap": 1.0}, prior=_prior(engine_id, "policy_gap"), now=now, include_engine=True)


def policy_calibration(db_path: str, engine: str, symbol: str, setup: str, regime: str, *, now: float | None = None, rows: list | None = None) -> dict[str, Any]:
    """Shrunk mean of realized policy net minus the policy value predicted at entry.

    Prior 0. A loss below its forecast pulls the next forecast down. A result
    above its forecast can pull the next forecast up. There is no fixed offset.
    """
    engine_id = str(engine or "").upper()
    if rows is None:
        rows = _state_rows(db_path, engine_id, ("policy_calibration",))
    cal_rows = [r for r in rows if str(r["metric"]) == "policy_calibration"]
    return _lattice(
        cal_rows,
        engine_id=engine_id,
        symbol=symbol,
        setup=setup,
        regime=regime,
        weights={"policy_calibration": 1.0},
        prior=_prior(engine_id, "policy_calibration"),
        now=now,
        include_engine=True,
    )


def _shifted(row: Any, shift: float) -> dict[str, Any]:
    out = dict(row)
    out["ewma"] = float(row["ewma"]) + float(shift)
    return out


def day_net_expectancy(db_path: str, symbol: str, setup: str, regime: str, *, now: float | None = None) -> dict[str, Any]:
    """Expected DAY net edge after costs under the live policy for one key,
    pooled hierarchically (key -> same setup sharing symbol or regime -> setup -> 0).
    No sample-count floor: one observation moves the posterior by its weight.
    ``market_alpha`` is the same posterior of the unshifted lifecycle labels."""
    weights = dict(DAY_NET_PARTS)
    rows = _state_rows(db_path, DAY_ENGINE, (*weights, "policy_gap"))
    now = float(now) if now is not None else _data_clock(rows)
    gap = policy_gap(db_path, DAY_ENGINE, symbol, setup, regime, now=now, rows=rows)
    calibration = policy_calibration(db_path, DAY_ENGINE, symbol, setup, regime, now=now)
    shift = float(gap["mean"])
    net_rows = [_shifted(r, shift) if str(r["metric"]) == "lifecycle_net" else r for r in rows if str(r["metric"]) in weights]
    lat = _lattice(
        net_rows,
        engine_id=DAY_ENGINE,
        symbol=symbol,
        setup=setup,
        regime=regime,
        weights=weights,
        prior=_prior(DAY_ENGINE, "trade_net"),
        now=now,
    )
    market = _lattice(
        [r for r in rows if str(r["metric"]) == "lifecycle_net"],
        engine_id=DAY_ENGINE,
        symbol=symbol,
        setup=setup,
        regime=regime,
        weights={"lifecycle_net": LIFECYCLE_WEIGHT},
        prior=_prior(DAY_ENGINE, "lifecycle_net"),
        now=now,
    )
    return {
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
        "market_alpha": market["mean"],
        "policy_gap": shift,
        "n_policy_gap": gap["level_weights"]["engine"],
        "uncalibrated_mean": lat["mean"],
        "policy_calibration": float(calibration["mean"]),
        "n_policy_calibration": calibration["level_weights"]["engine"],
        "mean": lat["mean"] + float(calibration["mean"]),
    }


def day_learned_setup(setup: str, *, detector_fired: bool) -> str:
    """Population key for one DAY state.

    The detector names a tighter state of the same family. An unfired structural
    state does not inherit that state's evidence, and the suffix is not a
    permission bit: ``day_decision`` prices either key the same way.
    """
    name = str(setup or "").upper()
    if name.endswith(DAY_MARKET_STATE_SUFFIX):
        name = name[: -len(DAY_MARKET_STATE_SUFFIX)]
    return name if detector_fired else f"{name}{DAY_MARKET_STATE_SUFFIX}"


def day_geometry_setup(setup: str) -> str:
    """Setup family used for structure and the exit objective, without the learned-population suffix."""
    name = str(setup or "").upper()
    if name.endswith(DAY_MARKET_STATE_SUFFIX):
        return name[: -len(DAY_MARKET_STATE_SUFFIX)]
    return name


def hold_remaining_metric(unrealized_net: float) -> str:
    """Which continuation posterior the current mark reads. The sign is state, not a gate."""
    return "hold_remaining_up" if float(unrealized_net) > 0.0 else "hold_remaining_down"


def continuation_terminal(
    db_path: str,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    unrealized_net: float,
    *,
    now: float | None = None,
    features: dict | None = None,
) -> float | None:
    """Expected net if the position is kept, from the current mark plus learned remaining value.

    Remaining value is a neutral prior of 0, so a cold state holds. Negative
    evidence that keeping the position gave back money makes the terminal worse
    than cashing out. Later positive evidence raises it again.
    """
    try:
        mark = float(unrealized_net)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mark):
        return None
    engine_id = str(engine or "").upper()
    if engine_id not in _PRIORS:
        return None
    from backend.services.continuation_surface import advantage_authority, installed_aggregator, surface_advantage

    moment = float(now if now is not None else time.time())
    if advantage_authority(db_path, engine_id):
        state = features if isinstance(features, dict) else {"net": mark}
        advantage = surface_advantage(db_path, engine_id, symbol, setup, regime, state, moment, how=installed_aggregator(db_path, engine_id))
        if advantage is None or not math.isfinite(float(advantage)):
            return mark
        return mark + float(advantage)
    remaining = estimate(db_path, engine_id, symbol, setup, regime, hold_remaining_metric(mark), now=now, include_engine=True)
    mean = float(remaining["mean"])
    if not math.isfinite(mean):
        return None
    return mark + mean


def learned_hold_or_exit(*, expected_terminal_net: float | None, unrealized_net: float) -> str:
    """Compare cashing out now with the learned terminal net of this state.

    ``exit`` when the learned terminal is worse than the net available now.
    A missing or non-finite terminal is neutral, so the position holds.
    Age is not an input.
    """
    if expected_terminal_net is None:
        return "hold"
    try:
        terminal = float(expected_terminal_net)
        mark = float(unrealized_net)
    except (TypeError, ValueError):
        return "hold"
    if not math.isfinite(terminal) or not math.isfinite(mark):
        return "hold"
    if terminal < mark:
        return "exit"
    return "hold"


def day_size_mult(expected_net: float, risk: float) -> float:
    """Bounded size from expected net per unit of adverse risk (as SCALP sizes
    its final edge). Negative evidence shrinks toward the floor; it never blocks."""
    lo, hi = SIZE_BOUNDS[DAY_ENGINE]
    tilt = math.tanh(float(expected_net) / risk) if risk > 0 else 0.0
    return _clamp(1.0 + 0.30 * tilt, lo, hi)


def day_decision(db_path: str, symbol: str, setup: str, regime: str, *, features: dict | None = None, now: float | None = None) -> dict[str, Any]:
    """What the next DAY candidate reads. Ranking, size, objective and runner only.

    Expected net edge after costs is the hierarchical realized net. It is the
    rank score and sets the bounded size. Learned move potential (MFE) sets the
    objective. Ranking never removes a candidate. ``features`` are recorded with
    the candidate; they do not change this number.
    """
    from backend.config.trading_economics import canonical_roundtrip_cost_pct

    mfe = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_mfe", now=now)
    mae = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_mae", now=now)
    timing = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_time_to_mfe_min", now=now)
    continuation = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "trade_continuation", now=now)
    forward = estimate(db_path, DAY_ENGINE, symbol, setup, regime, "markout_forward", now=now)
    net = day_net_expectancy(db_path, symbol, setup, regime, now=now)
    _ = features
    expected = float(net["mean"])
    cost = canonical_roundtrip_cost_pct()
    size_mult = day_size_mult(expected, max(0.0, mae["mean"]) + cost)
    abstain_flag, abstain_reason = _abstain(expected)
    # The objective distance only breaks ranking ties. The economic quantity is
    # the hierarchical realized net. State features are recorded, not scored.
    economic = {
        "engine_id": DAY_ENGINE,
        "economic_version": current_economic_version(DAY_ENGINE),
        "raw_expected_move": 0.0,
        "hierarchical_net": net["mean"],
        "state_tilt": 0.0,
        "state_model_n": 0.0,
        "learned_expected_gross_return": expected + cost,
        "calibration_adjustment": expected + cost,
        "expected_gross": expected + cost,
        "expected_cost": cost,
        "learned_expected_net_return": expected,
        "adaptive_correction": expected - net["prior"],
        "uncertainty": net["sd"],
        "downside": mae["mean"],
        "confidence_weight": net["confidence"],
        "setup_expected_edge": net["levels"]["setup"],
        "correlation_adjustment": {"applied": False, "edge": 0.0, "size_mult": 1.0},
        "expected_net_edge": expected,
        "final_learned_net_edge": expected,
        "size_effect": size_mult - 1.0,
        "levels": {lvl: round(v, 6) for lvl, v in net["levels"].items()},
        "level_weights": {lvl: round(v, 3) for lvl, v in net["level_weights"].items()},
        "n_trade": round(net["n_trade"], 3),
        "n_lifecycle": round(net["n_lifecycle"], 3),
        "market_alpha": net["market_alpha"],
        "policy_gap": net["policy_gap"],
        "n_policy_gap": round(net["n_policy_gap"], 3),
        "uncalibrated_policy_value": net["uncalibrated_mean"],
        "policy_calibration": net["policy_calibration"],
        "n_policy_calibration": round(net["n_policy_calibration"], 3),
        "policy_value": expected,
    }
    return {
        "adaptive_state_version": adaptive_format(DAY_ENGINE),
        "economic_version": current_economic_version(DAY_ENGINE),
        "engine_id": DAY_ENGINE,
        "symbol": str(symbol or "").upper(),
        "setup": str(setup or "").upper(),
        "regime": str(regime or "").lower(),
        "expected_move": mfe["mean"],
        "expected_move_prior": mfe["prior"],
        "expected_net": expected,
        "expected_net_parent": net["parent"],
        "net_confidence": net["confidence"],
        "confidence": net["confidence"],
        "abstain": abstain_flag,
        "abstain_reason": abstain_reason,
        "abstain_net_edge": expected,
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


def update_linear_model(db_path: str, engine: str, model: str, features: dict | None, target: float, *, now: float | None = None, weight: float = 1.0) -> None:
    """One online (normalised-LMS) step, scaled by the sample's ``weight`` in [0, 1].
    Standardisation stats adapt via EWMA; weights are clamped."""
    step = _clamp(float(weight), 0.0, 1.0) if math.isfinite(float(weight)) else 0.0
    if step <= 0:
        return
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
        a = MICRO_MODEL_STD_ALPHA * step
        lr = MICRO_MODEL_LR * step
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
        st["bias"] = float(st["bias"]) + lr * err
        for f in feats:
            w = float(st["w"].get(f, 0.0)) + lr * err * z[f]
            st["w"][f] = _clamp(w, -MICRO_MODEL_W_MAX, MICRO_MODEL_W_MAX)
        st["n"] = float(st.get("n", 0)) + step
        conn.execute(
            """
            INSERT INTO adaptive_linear_model (engine_id, model, payload, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(engine_id, model) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
            """,
            (engine_id, _model_key(engine_id, model), json.dumps(st, separators=(",", ":")), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time()))),
        )
        conn.commit()


def micro_edge_tilt(db_path: str, engine: str, features: dict | None) -> tuple[float, float]:
    """Bounded, zero-centred microstructure edge tilt (excludes the bias/mean).
    Returns (tilt, n), n the weighted sample count. Empty features or a cold
    model return (0.0, 0)."""
    feats = _micro_features(features)
    if not any(v != 0.0 for v in feats.values()):
        return 0.0, 0
    engine_id = str(engine or "").upper()
    try:
        with _connect(db_path) as conn:
            st = _load_linear(conn, engine_id, "micro_edge")
    except sqlite3.Error:
        return 0.0, 0
    n = float(st.get("n", 0) or 0)
    if n <= 0:
        return 0.0, 0
    z = _standardize(st, feats)
    tilt = sum(float(st["w"].get(f, 0.0)) * z[f] for f in feats)
    return _clamp(tilt, -MICRO_MODEL_TILT_MAX, MICRO_MODEL_TILT_MAX), n


def learn_micro_weight(db_path: str, *, strategy_version: str, tilt: float, miss: float, now: float | None = None, weight: float = 1.0) -> bool:
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
        wrote = observe(db_path, engine=SCALP_ENGINE, symbol=sym, setup=stp, regime=reg, metric=metric, value=value, strategy_version=strategy_version, now=now, weight=weight) or wrote
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
    raw: float,
    gross: float,
    now: float | None = None,
    weight: float = 1.0,
) -> bool:
    """Fold one resolved strategy claim into the key's (projection, gross) moments.

    ``raw`` is the decision-time structural projection (>= 0, 0 included);
    ``gross`` is the realized gross move, net markout plus the decision-time cost;
    ``weight`` is the claim's label uniqueness (``claim_uniqueness``).
    """
    b = float(raw)
    g = float(gross)
    wrote = False
    for metric, value in zip(CLAIM_MOMENT_METRICS, (b, b * b, g, b * g), strict=True):
        wrote = observe(db_path, engine=SCALP_ENGINE, symbol=symbol, setup=setup, regime=regime, metric=metric, value=value, strategy_version=strategy_version, now=now, weight=weight) or wrote
    return wrote


def claim_uniqueness(conn: sqlite3.Connection, row: sqlite3.Row, horizon_sec: float) -> float:
    """Share of a SCALP claim's label window not covered by the previous claim on
    the same symbol, min(1, gap / horizon). Claims whose windows overlap share one
    forward price path, so a burst of claims counts about once per horizon."""
    if horizon_sec <= 0:
        return 1.0
    prev = conn.execute(
        "SELECT MAX(evaluated_at) FROM adaptive_candidate_markouts WHERE engine_id=? AND symbol=? AND raw_move_source=? AND economic_version=? AND (evaluated_at<? OR (evaluated_at=? AND id<?))",
        (SCALP_ENGINE, row["symbol"], STRATEGY_CLAIM, row["economic_version"], row["evaluated_at"], row["evaluated_at"], row["id"]),
    ).fetchone()
    if prev is None or prev[0] is None:
        return 1.0
    return _clamp((float(row["evaluated_at"]) - float(prev[0])) / float(horizon_sec), 0.0, 1.0)


def scalp_claim_calibration(db_path: str, symbol: str, setup: str, regime: str, *, now: float | None = None) -> dict[str, Any]:
    """Expected gross directional move of a strategy claim, learned from realized moves.

    E[gross | projection] = gross mean + capture x (projection - projection mean).
    The gross mean is the usual hierarchical posterior of ``claim_gross``
    (setup -> related -> key) from prior 0. ``claim_capture`` is the within-key
    slope of gross on the projection, a ridge estimate at each level toward its
    parent with ``PRIOR_STRENGTH`` claims of variance ``CLAIM_SLOPE_PRIOR_VAR``
    as the parent's weight, from slope 0 at the top, kept in [0, 1]. Cold, the
    expected move is 0 whatever the projection: target geometry, ATR or a floor
    is never a move by itself. ``claim_raw_center`` is the hierarchical mean
    projection, the slope's pivot. Claim labels arrive weighted by their
    uniqueness, so ``n`` counts independent windows rather than overlapping ones.
    """
    sym, stp, reg = _norm_symbol(symbol), str(setup or "").upper(), str(regime or "").lower()
    claim_rows = _state_rows(db_path, SCALP_ENGINE, CLAIM_MOMENT_METRICS)
    moment = float(now) if now is not None else _data_clock(claim_rows)
    gross = _lattice(claim_rows, engine_id=SCALP_ENGINE, symbol=symbol, setup=setup, regime=regime, weights={"claim_gross": 1.0}, prior=0.0, now=moment)
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
        b = rec["claim_raw"]
        bucket = acc[_level_of(rec["row"], sym, stp, reg)]
        bucket[0] += n
        bucket[1] += n * b
        bucket[2] += n * (rec["claim_raw_x_gross"] - b * rec["claim_gross"])
        bucket[3] += n * max(0.0, rec["claim_raw_sq"] - b * b)
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
        "claim_gross_mean": gross["mean"],
        "claim_gross_setup": gross["levels"]["setup"],
        "claim_gross_levels": gross["levels"],
        "claim_capture": _clamp(slope, 0.0, 1.0),
        "claim_slope": slope,
        "claim_raw_center": center if center is not None else 0.0,
        "claim_key_n": acc["key"][0],
        "n_claim": nested_w["setup"],
        "claim_level_weights": nested_w,
        "claim_confidence": gross["confidence"],
        "claim_uncertainty": gross["sd"],
    }


def scalp_expected_gross(view: dict[str, Any], raw: float) -> dict[str, float]:
    """Calibrated expected gross move of a claim with projection ``raw`` under ``view``.

    ``calibrated``: the setup-level gross mean plus the learned capture of the
    projection's deviation from its mean. ``adaptive``: the key's gross mean
    against its setup's, bounded by ``SCALP_RESIDUAL_MAX`` on both sides. A key
    that has earned more than its setup raises the move; one that has earned
    less lowers it. Their sum is the one expected directional move.
    """
    capture = _clamp(float(view.get("claim_capture") or 0.0), 0.0, 1.0)
    setup_mean = float(view.get("claim_gross_setup") or 0.0)
    calibrated = setup_mean + capture * (max(0.0, float(raw or 0.0)) - float(view.get("claim_raw_center") or 0.0))
    adaptive = _clamp(float(view.get("claim_gross_mean") or 0.0) - setup_mean, -SCALP_RESIDUAL_MAX, SCALP_RESIDUAL_MAX)
    return {"capture": capture, "calibrated": calibrated, "adaptive": adaptive, "expected": calibrated + adaptive}


def _blend(parts: list[tuple[float, float]], prior: float) -> tuple[float, float]:
    weighted = [(mean, weight) for mean, weight in parts if weight > 0]
    if not weighted:
        return prior, 0.0
    weight = sum(item[1] for item in weighted)
    mean = (PRIOR_STRENGTH * prior + sum(m * w for m, w in weighted)) / (PRIOR_STRENGTH + weight)
    return mean, weight


def scalp_decision(db_path: str, symbol: str, setup: str, regime: str, features: dict | None = None, *, now: float | None = None) -> dict[str, Any]:
    """Learned inputs for the next SCALP candidate. Never an edge by itself.

    scalp_v2.executable_edge prices the candidate: calibrated expected move
    (``scalp_claim_calibration`` via ``scalp_expected_gross``) - live cost +
    bounded micro residual. This view supplies the calibration, the weighted
    microstructure residual, confidence, risk, target and hold. Cold, the
    expected move is 0 and the micro residual is 0.
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
    # Bounded microstructure residual: shrunk by the model's own sample count so
    # a cold model barely moves the edge, then by the learned micro weight.
    # Zero-centred (excludes the model bias).
    micro_tilt, micro_n = micro_edge_tilt(db_path, SCALP_ENGINE, features)
    micro_conf = micro_n / (micro_n + MICRO_MODEL_CONF_K) if micro_n > 0 else 0.0
    micro_unweighted = micro_conf * micro_tilt
    weight = micro_weight(db_path, now=now)
    micro_residual = micro_unweighted * weight["weight"]
    claim = scalp_claim_calibration(db_path, symbol, setup, regime, now=now)
    gap = policy_gap(db_path, SCALP_ENGINE, symbol, setup, regime, now=now)
    calibration = policy_calibration(db_path, SCALP_ENGINE, symbol, setup, regime, now=now)
    # Telemetry only: the canonical executable edge is the single live negative-edge gate.
    learned = forward if forward["n"] > 0 else net
    learned_net = learned["mean"]
    abstain, abstain_reason = _abstain(learned_net)
    return {
        "adaptive_state_version": adaptive_format(SCALP_ENGINE),
        "engine_id": SCALP_ENGINE,
        "symbol": str(symbol or "").upper(),
        "setup": str(setup or "").upper(),
        "regime": str(regime or "").lower(),
        "adaptive_residual": _clamp(residual["mean"], -SCALP_RESIDUAL_MAX, SCALP_RESIDUAL_MAX),
        "adaptive_residual_raw": residual["mean"],
        "n_residual": residual["n"],
        "raw_claim_bias": residual_strategy["mean"],
        "n_raw_claim_bias": residual_strategy["n"],
        "economic_version": current_economic_version(SCALP_ENGINE),
        "micro_residual": micro_residual,
        "micro_residual_unweighted": micro_unweighted,
        "micro_weight": weight["weight"],
        "micro_weight_n": weight["n"],
        "micro_tilt": round(micro_residual, 6),
        "micro_tilt_raw": round(micro_tilt, 6),
        "micro_model_n": micro_n,
        "state_tilt": 0.0,
        "state_model_n": 0.0,
        "claim_gross_mean": claim["claim_gross_mean"],
        "claim_gross_setup": claim["claim_gross_setup"],
        "claim_gross_levels": {lvl: round(v, 6) for lvl, v in claim["claim_gross_levels"].items()},
        "claim_capture": claim["claim_capture"],
        "claim_slope": claim["claim_slope"],
        "claim_raw_center": claim["claim_raw_center"],
        "claim_key_n": claim["claim_key_n"],
        "n_claim": claim["n_claim"],
        "claim_uncertainty": claim["claim_uncertainty"],
        "confidence": claim["claim_confidence"],
        "policy_gap": gap["mean"],
        "n_policy_gap": gap["level_weights"]["engine"],
        "policy_calibration": calibration["mean"],
        "n_policy_calibration": calibration["level_weights"]["engine"],
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
    unrealized_marks: Sequence[float] | None = None,
    opportunity_id: str = "",
    exit_reason: str | None = None,
    candidate_id: int | None = None,
) -> bool:
    """Realized-trade update. Dust, non-current versions, positions entered
    before the engine's economic anchor and closes of a retired exit policy
    (``exit_contract_of`` not CURRENT) do not move state. ``candidate_id`` (the
    entry's own row) or ``opportunity_id`` links the close to its filled
    candidate so calibration learns at the close and the policy gap learns when
    its market label is final."""
    if is_dust or not version_current:
        return False
    engine_id = str(engine or "").upper()
    if entered_at is not None and float(entered_at) < anchor_epoch(engine_id):
        return False
    if exit_contract_of(engine_id, entered_at=entered_at, exit_reason=exit_reason) != "CURRENT":
        return False
    wrote = False
    if net_pct is not None and (opportunity_id or candidate_id):
        wrote = record_policy_outcome(db_path, engine=engine_id, opportunity_id=opportunity_id, net_pct=float(net_pct), now=now, candidate_id=candidate_id) or wrote
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
    if net_pct is not None and unrealized_marks:
        clean: list[float] = []
        for raw in unrealized_marks:
            try:
                mark = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(mark):
                clean.append(mark)
        step = max(1, len(clean) // 8)
        sampled = clean[::step][:8]
        if sampled:
            share = 1.0 / len(sampled)
            for mark in sampled:
                wrote = (
                    observe(
                        db_path,
                        engine=engine_id,
                        symbol=symbol,
                        setup=setup,
                        regime=regime,
                        metric=hold_remaining_metric(mark),
                        value=float(net_pct) - mark,
                        strategy_version=strategy_version,
                        now=now,
                        weight=share,
                    )
                    or wrote
                )
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
                if exit_contract_of(DAY_ENGINE, entered_at=_epoch_any(sell["entry_timestamp"]), exit_reason=sell["exit_reason"]) != "CURRENT":
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


POLICY_SEED_EXCLUDED_EXITS = ("DUST_WRITEOFF", "HUMAN_MANUAL_SELL", "MANUAL_UNMATCHED")


def seed_policy_gap(db_path: str, engine: str, *, apply: bool = False) -> dict[str, Any]:
    """Replay current-version closes of ``engine`` into ``policy_gap``, in time order.

    Each close entered at or after the economic anchor is linked to its own
    candidate row: DAY by the filled row of its opportunity (the SELL written
    within 30 s of the learned close), SCALP by the admitted claim on the same
    symbol recorded in the 30 s before its entry. The
    realized net is the one ``learn_from_close`` learned. Each gap is folded at
    the later of the close and the market label's resolution. Refuses once any
    ``policy_gap`` state exists for the engine; a row already learned is never
    learned twice. Dry run unless ``apply``.
    """
    engine_id = str(engine or "").upper()
    out: dict[str, Any] = {"engine": engine_id, "applied": False, "refused": "", "pairs": []}
    if engine_id not in _PRIORS:
        out["refused"] = "UNKNOWN_ENGINE"
        return out
    version = current_economic_version(engine_id)
    anchor = anchor_epoch(engine_id)
    plan: list[dict[str, Any]] = []
    try:
        conn = _connect(db_path)
        try:
            if conn.execute(
                "SELECT 1 FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND metric='policy_gap' LIMIT 1",
                (engine_id, version),
            ).fetchone():
                out["refused"] = "POLICY_GAP_STATE_EXISTS"
                return out
            excluded = ",".join("?" * len(POLICY_SEED_EXCLUDED_EXITS))
            outcomes = conn.execute(
                f"SELECT symbol, entry_timestamp, exit_timestamp, net_profit_pct, close_reason FROM trade_learning_outcomes "
                f"WHERE UPPER(COALESCE(engine_id,''))=? AND entry_timestamp>=? AND net_profit_pct IS NOT NULL AND COALESCE(close_reason,'') NOT IN ({excluded}) ORDER BY exit_timestamp",
                (engine_id, anchor, *POLICY_SEED_EXCLUDED_EXITS),
            ).fetchall()
            for outcome in outcomes:
                entered, closed = float(outcome["entry_timestamp"]), float(outcome["exit_timestamp"])
                if exit_contract_of(engine_id, entered_at=entered, exit_reason=outcome["close_reason"]) != "CURRENT":
                    continue
                symbol = str(outcome["symbol"] or "")
                sell = conn.execute(
                    "SELECT scalp_opportunity_id FROM paper_trades WHERE UPPER(side)='SELL' AND UPPER(COALESCE(engine_id,''))=? AND symbol=? "
                    "AND ABS(strftime('%s', substr(timestamp, 1, 19)) - ?) <= 30 ORDER BY ABS(strftime('%s', substr(timestamp, 1, 19)) - ?) LIMIT 1",
                    (engine_id, symbol, closed, closed),
                ).fetchone()
                opp = str(sell["scalp_opportunity_id"] or "") if sell is not None else ""
                if engine_id == DAY_ENGINE:
                    row = (
                        conn.execute(
                            "SELECT * FROM adaptive_candidate_markouts WHERE engine_id=? AND opportunity_id=? AND filled=1 AND economic_version=? ORDER BY id ASC LIMIT 1",
                            (engine_id, opp, version),
                        ).fetchone()
                        if opp
                        else None
                    )
                else:
                    row = conn.execute(
                        "SELECT * FROM adaptive_candidate_markouts WHERE engine_id=? AND symbol=? AND signaled=1 AND economic_version=? AND evaluated_at BETWEEN ? AND ? "
                        "ORDER BY evaluated_at DESC LIMIT 1",
                        (engine_id, _norm_symbol(symbol), version, entered - 30.0, entered + 1.0),
                    ).fetchone()
                if row is None or int(row["policy_learned"] or 0):
                    continue
                marks = json.loads(row["markouts_json"] or "{}")
                market = _market_label(engine_id, row, marks)
                if market is None:
                    # Stored now; resolve_markouts learns it once the label is final.
                    out.setdefault("pending", []).append({"row_id": int(row["id"]), "opportunity_id": opp, "realized_net": float(outcome["net_profit_pct"])})
                    continue
                if engine_id == DAY_ENGINE:
                    life = marks.get("lifecycle") if isinstance(marks.get("lifecycle"), dict) else {}
                    resolved_at = float(row["evaluated_at"]) + float(life.get("minutes") or 0.0) * 60.0
                else:
                    resolved_at = float(row["evaluated_at"]) + (float(row["label_horizon"] or 0) or 600.0)
                plan.append(
                    {
                        "row_id": int(row["id"]),
                        "opportunity_id": opp,
                        "symbol": _norm_symbol(symbol),
                        "setup": str(row["setup"]),
                        "regime": str(row["regime"]),
                        "close_reason": str(outcome["close_reason"] or ""),
                        "realized_net": float(outcome["net_profit_pct"]),
                        "market_label": market,
                        "gap": float(outcome["net_profit_pct"]) - market,
                        "at": max(closed, resolved_at),
                    }
                )
            plan.sort(key=lambda p: p["at"])
            out["pairs"] = plan
            if apply:
                for pair in [*plan, *out.get("pending", [])]:
                    conn.execute(
                        "UPDATE adaptive_candidate_markouts SET filled=1, realized_net=?, opportunity_id=CASE WHEN ?<>'' THEN ? ELSE opportunity_id END WHERE id=?",
                        (pair["realized_net"], pair["opportunity_id"], pair["opportunity_id"], pair["row_id"]),
                    )
                    conn.commit()
                    if "at" in pair:
                        row = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE id=?", (pair["row_id"],)).fetchone()
                        _learn_policy_gap(conn, db_path, row, json.loads(row["markouts_json"] or "{}"), pair["realized_net"], pair["at"])
                out["applied"] = True
        finally:
            conn.close()
    except (sqlite3.Error, ValueError) as exc:
        out["refused"] = f"READ_FAILED {type(exc).__name__}"
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


def _stored_features(features: dict | None, engine_id: str = SCALP_ENGINE) -> str:
    if not isinstance(features, dict) or not features:
        return "{}"
    if engine_id == DAY_ENGINE:
        # Decision-time DAY state, kept for research. No DAY learner reads it.
        day: dict[str, float] = {}
        for k, v in features.items():
            with contextlib.suppress(TypeError, ValueError):
                if math.isfinite(float(v)):
                    day[str(k)] = float(v)
        return json.dumps(day, separators=(",", ":"), sort_keys=True) if day else "{}"
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
    feats_json = _stored_features(features, engine_id)
    horizon = _label_horizon_for(db_path, engine_id, symbol, setup, regime)
    raw_move: float | None = None
    with contextlib.suppress(TypeError, ValueError):
        raw_move = max(0.0, float(raw_expected_move)) if raw_expected_move is not None and math.isfinite(float(raw_expected_move)) else None
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


def link_candidate_fill(db_path: str, row_id: int | None, opportunity_id: str = "") -> bool:
    """Flag a recorded candidate as filled under the position's opportunity id,
    so the position's close finds the candidate's market label."""
    if not row_id:
        return False
    try:
        with _connect(db_path) as conn:
            cur = conn.execute(
                "UPDATE adaptive_candidate_markouts SET filled=1, opportunity_id=CASE WHEN ?<>'' THEN ? ELSE opportunity_id END WHERE id=?",
                (str(opportunity_id or ""), str(opportunity_id or ""), int(row_id)),
            )
            conn.commit()
            return cur.rowcount == 1
    except sqlite3.Error:
        return False


def _market_label(engine_id: str, row: Any, marks: dict) -> float | None:
    """The filled candidate's own market label: DAY lifecycle net, SCALP forward
    net at the claim's label horizon. None until it is final, and for a DAY
    label produced by a retired exit policy."""
    if engine_id == DAY_ENGINE:
        life = marks.get("lifecycle")
        value = life.get("net") if isinstance(life, dict) and current_lifecycle_label(life.get("reason")) else None
    else:
        label_h = float(row["label_horizon"] or 0) or 600.0
        value = _mark_at(marks, label_h)
    try:
        out = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return out if out is not None and math.isfinite(out) else None


def predicted_policy_value(decision: dict | None) -> float | None:
    """Policy value the entry actually issued. None when it was not stored."""
    econ = decision.get("economic") if isinstance(decision, dict) else None
    raw = econ.get("policy_value") if isinstance(econ, dict) else None
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _predicted_policy_value(row: Any) -> float | None:
    raw = _economic_of(row).get("policy_value")
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def observe_policy_calibration(
    db_path: str,
    *,
    engine: str,
    symbol: str,
    setup: str,
    regime: str,
    realized: float,
    predicted: float,
    now: float,
) -> bool:
    """One running-mean observation of realized net minus the entry forecast.

    Prior 0, both signs, no sample floor and no added penalty. The market
    lifecycle label is not an input."""
    if not math.isfinite(float(realized)) or not math.isfinite(float(predicted)):
        return False
    return observe(
        db_path,
        engine=str(engine or "").upper(),
        symbol=symbol,
        setup=setup,
        regime=regime,
        metric="policy_calibration",
        value=float(realized) - float(predicted),
        strategy_version=current_strategy_version(str(engine or "").upper()),
        now=float(now),
    )


def _learn_policy_calibration(conn: sqlite3.Connection, db_path: str, row: Any, realized: float, moment: float) -> bool:
    """Fold one close into ``policy_calibration`` at the close. Exactly once.

    The learned flag and the observation commit together. A crash leaves the
    close unlearned so the next pass can write both."""
    del db_path
    engine_id = str(row["engine_id"])
    if not math.isfinite(float(realized)):
        return False
    if str(row["strategy_version"]) != current_strategy_version(engine_id) or str(row["economic_version"] or "") != current_economic_version(engine_id):
        return False
    predicted = _predicted_policy_value(row)
    if predicted is None:
        return False
    return _claim_and_fold(
        conn,
        row_id=int(row["id"]),
        flag="calibration_learned",
        engine=engine_id,
        symbol=str(row["symbol"]),
        setup=str(row["setup"]),
        regime=str(row["regime"]),
        metric="policy_calibration",
        value=float(realized) - float(predicted),
        strategy_version=current_strategy_version(engine_id),
        moment=float(moment),
    )


def _learn_policy_gap(conn: sqlite3.Connection, db_path: str, row: Any, marks: dict, realized: float, moment: float) -> bool:
    """Fold realized minus the row's market label into ``policy_gap`` once.

    This waits until that label is final. It does not write ``policy_calibration``.
    The flag and the gap observation commit together."""
    del db_path
    engine_id = str(row["engine_id"])
    market = _market_label(engine_id, row, marks)
    if market is None or not math.isfinite(float(realized)):
        return False
    if str(row["strategy_version"]) != current_strategy_version(engine_id) or str(row["economic_version"] or "") != current_economic_version(engine_id):
        return False
    return _claim_and_fold(
        conn,
        row_id=int(row["id"]),
        flag="policy_learned",
        engine=engine_id,
        symbol=str(row["symbol"]),
        setup=str(row["setup"]),
        regime=str(row["regime"]),
        metric="policy_gap",
        value=float(realized) - float(market),
        strategy_version=str(row["strategy_version"]),
        moment=float(moment),
    )


def _utc_epoch(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    elif "+" not in text[10:] and not text.endswith("Z"):
        text = text.replace(" ", "T", 1) + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _calibration_backfill_floor() -> float:
    parsed = _utc_epoch(CALIBRATION_BACKFILL_FROM)
    return float(parsed if parsed is not None else 0.0)


def _close_moment(conn: sqlite3.Connection, row: Any, *, apply_floor: bool = True) -> float | None:
    """Sell time of this filled candidate when the close is a current-policy exit.

    The close-time backfill also requires the sell to be at or after the
    calibration repair. ``apply_floor=False`` is the mismatch repair: a flagged
    current-policy close is eligible at its own sell time."""
    engine_id = str(row["engine_id"])
    symbol = _norm_symbol(row["symbol"])
    try:
        evaluated = float(row["evaluated_at"] or 0.0)
    except (TypeError, ValueError):
        return None
    try:
        sells = conn.execute(
            "SELECT exit_reason, timestamp AS closed_at, entry_timestamp AS entered_at FROM paper_trades "
            "WHERE UPPER(side)='SELL' AND UPPER(COALESCE(engine_id,''))=? AND REPLACE(REPLACE(UPPER(symbol),'/',''),'-','')=?",
            (engine_id, symbol),
        ).fetchall()
    except sqlite3.Error:
        return None
    best = None
    best_gap = None
    best_entered = None
    best_closed = None
    for sell in sells:
        entered = _utc_epoch(sell["entered_at"])
        closed = _utc_epoch(sell["closed_at"])
        if entered is None or closed is None:
            continue
        gap = abs(entered - evaluated)
        if gap > 1200.0:
            continue
        if best_gap is None or gap < best_gap:
            best = sell
            best_gap = gap
            best_entered = entered
            best_closed = closed
    if best is None or best_entered is None or best_closed is None:
        return None
    reason = str(best["exit_reason"] or "")
    if "DUST" in reason.upper():
        return None
    entered = best_entered
    closed = best_closed
    if apply_floor and closed < _calibration_backfill_floor():
        return None
    if exit_contract_of(engine_id, entered_at=entered, exit_reason=reason) != "CURRENT":
        return None
    return closed


_calibration_backfilled: set[str] = set()


def _calibration_identity(row: Any) -> tuple[str, str, str, str, str] | None:
    """Current-version key of one candidate, or None when the row is another contract."""
    engine_id = str(row["engine_id"])
    if str(row["strategy_version"]) != current_strategy_version(engine_id) or str(row["economic_version"] or "") != current_economic_version(engine_id):
        return None
    return (
        engine_id,
        current_economic_version(engine_id),
        _norm_symbol(row["symbol"]),
        str(row["setup"] or "").upper(),
        str(row["regime"] or "").lower(),
    )


def repair_marked_calibration_without_observation(conn: sqlite3.Connection) -> tuple[int, int]:
    """Restore the learned-or-not invariant for current-version closes.

    A key with no ``policy_calibration`` row is not observed. Every flagged
    close on that key that has a stored forecast and a current-policy sell is
    folded once, in one savepoint, at its sell time. A flagged close that
    cannot be an observation has the flag cleared. A key that already has an
    observation is left unchanged, so a second pass cannot duplicate it.
    Fills, accounting and lifecycle history are not written."""
    rows = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE filled=1 AND realized_net IS NOT NULL AND calibration_learned=1").fetchall()
    groups: dict[tuple[str, str, str, str, str], list[Any]] = {}
    for row in rows:
        identity = _calibration_identity(row)
        if identity is None:
            continue
        groups.setdefault(identity, []).append(row)
    folded = 0
    cleared = 0
    for identity, members in groups.items():
        present = conn.execute(
            "SELECT 1 FROM adaptive_metric_state WHERE engine_id=? AND economic_version=? AND symbol=? AND setup=? AND regime=? AND metric='policy_calibration'",
            identity,
        ).fetchone()
        if present is not None:
            continue
        ready: list[tuple[float, int, Any, float, float]] = []
        for row in members:
            predicted = _predicted_policy_value(row)
            moment = _close_moment(conn, row, apply_floor=False)
            try:
                realized = float(row["realized_net"])
            except (TypeError, ValueError):
                realized = float("nan")
            if predicted is None or moment is None or not math.isfinite(realized):
                cur = conn.execute("UPDATE adaptive_candidate_markouts SET calibration_learned=0 WHERE id=? AND calibration_learned=1", (row["id"],))
                conn.commit()
                cleared += int(cur.rowcount or 0)
                continue
            ready.append((float(moment), int(row["id"]), row, float(predicted), realized))
        if not ready:
            continue
        ready.sort()
        conn.execute("SAVEPOINT mystic_calibration_repair")
        try:
            wrote = 0
            for moment, _row_id, row, predicted, realized in ready:
                ok = _fold_observation(
                    conn,
                    engine=str(row["engine_id"]),
                    symbol=str(row["symbol"]),
                    setup=str(row["setup"]),
                    regime=str(row["regime"]),
                    metric="policy_calibration",
                    value=realized - predicted,
                    strategy_version=str(row["strategy_version"]),
                    now=moment,
                )
                if not ok:
                    raise RuntimeError("calibration observation rejected")
                wrote += 1
            conn.execute("RELEASE mystic_calibration_repair")
            folded += wrote
        except Exception:
            conn.execute("ROLLBACK TO mystic_calibration_repair")
            conn.execute("RELEASE mystic_calibration_repair")
            logger.warning("CALIBRATION_REPAIR_KEY_FAILED symbol=%s setup=%s regime=%s", identity[2], identity[3], identity[4])
    if folded or cleared:
        logger.info("CALIBRATION_REPAIR folded=%s cleared=%s", folded, cleared)
    return folded, cleared


def backfill_close_calibration(db_path: str) -> int:
    """Teach calibration for current-policy closes the lifecycle wait skipped.

    One observation per filled candidate. A close already marked learned is
    repaired only when its key has no calibration observation. Fills,
    accounting and lifecycle history are not written."""
    key = os.path.abspath(db_path)
    if key in _calibration_backfilled:
        return 0
    taught = 0
    conn = _connect(db_path)
    try:
        repair_marked_calibration_without_observation(conn)
        rows = conn.execute("SELECT * FROM adaptive_candidate_markouts WHERE filled=1 AND realized_net IS NOT NULL AND calibration_learned=0").fetchall()
        for row in rows:
            moment = _close_moment(conn, row)
            if moment is None:
                continue
            try:
                realized = float(row["realized_net"])
            except (TypeError, ValueError):
                continue
            if _learn_policy_calibration(conn, db_path, row, realized, moment):
                taught += 1
    finally:
        conn.close()
    _calibration_backfilled.add(key)
    return taught


def record_policy_outcome(db_path: str, *, engine: str, opportunity_id: str, net_pct: float, now: float | None = None, candidate_id: int | None = None) -> bool:
    """Store a close's realized net and learn calibration immediately.

    ``policy_gap`` is learned in the same call only when the market label is
    already final. Otherwise ``resolve_markouts`` learns the gap later and does
    not observe calibration again.

    The candidate is the entry's own row (``candidate_id``), else the newest
    filled row of the opportunity still without a realized net: one opportunity
    can be filled again after an earlier position closed."""
    engine_id = str(engine or "").upper()
    if engine_id not in _PRIORS or not (opportunity_id or candidate_id) or not math.isfinite(float(net_pct)):
        return False
    moment = float(now if now is not None else time.time())
    try:
        with _connect(db_path) as conn:
            row = None
            if candidate_id:
                row = conn.execute(
                    "SELECT * FROM adaptive_candidate_markouts WHERE id=? AND engine_id=? AND filled=1 AND economic_version=?",
                    (int(candidate_id), engine_id, current_economic_version(engine_id)),
                ).fetchone()
            if row is None and opportunity_id:
                row = conn.execute(
                    "SELECT * FROM adaptive_candidate_markouts WHERE engine_id=? AND opportunity_id=? AND filled=1 AND economic_version=? AND realized_net IS NULL ORDER BY id DESC LIMIT 1",
                    (engine_id, str(opportunity_id), current_economic_version(engine_id)),
                ).fetchone()
            if row is None or row["realized_net"] is not None:
                return False
            conn.execute("UPDATE adaptive_candidate_markouts SET realized_net=? WHERE id=?", (float(net_pct), row["id"]))
            conn.commit()
            marks = json.loads(row["markouts_json"] or "{}")
            calibrated = _learn_policy_calibration(conn, db_path, row, float(net_pct), moment)
            gap = _learn_policy_gap(conn, db_path, row, marks, float(net_pct), moment)
            return bool(calibrated or gap)
    except (sqlite3.Error, ValueError):
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


def executable_bid_net(ref: float, bid: float, stored_cost: float) -> float:
    """Return from an ask entry to a later bid, minus costs the bid does not already contain.

    ``canonical_roundtrip_cost_pct`` adds the exit half-spread because a mid or
    trade mark does not include it. The bid already includes that half, so
    subtracting the stored cost again would charge it twice. Fee and slippage
    stay. A stored cost at or below that fee-and-slippage floor is unchanged.
    """
    from backend.config.trading_economics import SLIPPAGE_BUFFER, TAKER_FEE

    beyond = 2.0 * float(TAKER_FEE) + 2.0 * float(SLIPPAGE_BUFFER)
    exit_half = max(0.0, float(stored_cost) - beyond)
    return (float(bid) - float(ref)) / float(ref) - (float(stored_cost) - exit_half)


def executable_bid_at(db_path: str, symbol: str, target: float, horizon: int) -> float | None:
    """Bid aligned to ``target``. A later bar close is not a substitute."""
    from backend.services.full_state_research import scalp_observations
    from backend.services.horizon_alignment import align_observation

    aligned = align_observation(scalp_observations(db_path, symbol, float(target), int(horizon)), float(target), int(horizon))
    if aligned.get("status") != "OK" or aligned.get("source") == "bar_close":
        return None
    price = aligned.get("price")
    if price is None or not math.isfinite(float(price)) or float(price) <= 0:
        return None
    return float(price)


def _repair_short_marks(db_path: str, row: Any, marks: dict) -> None:
    """Replace stored 30s/60s marks with an executable bid, or drop an unlearned leak."""
    if str(row["engine_id"]) != SCALP_ENGINE:
        return
    ref = float(row["ref_price"] or 0)
    cost = float(row["roundtrip_cost"] or 0)
    learned = int(row["learned"] or 0)
    for horizon in (30, 60):
        key = str(horizon)
        if key not in marks:
            continue
        price = None
        with contextlib.suppress(Exception):
            price = executable_bid_at(db_path, str(row["symbol"]), float(row["evaluated_at"]) + horizon, horizon)
        if price is not None and ref > 0:
            marks[key] = executable_bid_net(ref, float(price), cost)
        elif not learned:
            marks.pop(key, None)


def repair_stored_short_marks(db_path: str, *, limit: int = 500) -> int:
    """Rewrite stored 30s/60s marks from an executable bid. Does not retrain state."""
    changed = 0
    try:
        conn = _connect(db_path)
    except sqlite3.Error:
        return 0
    try:
        rows = conn.execute(
            """SELECT id, engine_id, symbol, ref_price, roundtrip_cost, evaluated_at, learned, markouts_json
               FROM adaptive_candidate_markouts WHERE engine_id=?
               AND (markouts_json LIKE '%"30"%' OR markouts_json LIKE '%"60"%')
               ORDER BY id DESC LIMIT ?""",
            (SCALP_ENGINE, int(limit)),
        ).fetchall()
        for row in rows:
            marks = json.loads(row["markouts_json"] or "{}")
            before = json.dumps(marks, sort_keys=True)
            _repair_short_marks(db_path, row, marks)
            if json.dumps(marks, sort_keys=True) == before:
                continue
            conn.execute("UPDATE adaptive_candidate_markouts SET markouts_json=? WHERE id=?", (json.dumps(marks), row["id"]))
            changed += 1
        if changed:
            conn.commit()
    except sqlite3.Error:
        return changed
    finally:
        conn.close()
    return changed


_short_marks_repaired = False


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

    SCALP claim rows (admitted or not) also learn the claim calibration (the
    projection and the realized gross move), the raw-claim bias, the micro model
    and weight on what the decision-time edge before micro missed, and the gross
    path MAE over the committed horizon, which is the live risk estimate.
    ``path_low(symbol, start, end)`` supplies the bar-low path when available;
    ``tick_quote(symbol, start, end)`` the last tape print in a window, for the
    1/5/10 s marks where the tape has one.

    DAY rows recorded with lifecycle inputs replay the live exit contract from
    the decision ask over ``bars_1m(symbol, start, end)`` (stored 1m OHLCV by
    default). Once final, the first record of an opportunity that was never
    filled learns ``lifecycle_net``.
    """
    global _short_marks_repaired
    moment = float(now if now is not None else time.time())
    if not _short_marks_repaired:
        repair_stored_short_marks(db_path)
        _short_marks_repaired = True
    try:
        backfill_close_calibration(db_path)
    except (sqlite3.Error, ValueError):
        logger.warning("CALIBRATION_BACKFILL_FAILED", exc_info=True)
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
        # One window per engine: DAY rows wait up to the lifecycle censor, so a
        # shared oldest-first window fills with them and starves SCALP labels.
        # Rows still owed their primary label go first inside each window.
        rows = []
        for window_engine in (DAY_ENGINE, SCALP_ENGINE):
            rows.extend(
                conn.execute(
                    "SELECT * FROM adaptive_candidate_markouts WHERE resolved=0 AND engine_id=? ORDER BY learned ASC, id ASC LIMIT 200",
                    (window_engine,),
                ).fetchall()
            )
        for row in rows:
            try:
                engine_id = str(row["engine_id"])
                horizons = DAY_HORIZONS_MIN if engine_id == DAY_ENGINE else SCALP_HORIZONS_SEC
                unit = 60.0 if engine_id == DAY_ENGINE else 1.0
                grace = 900.0 if engine_id == DAY_ENGINE else 60.0
                marks = json.loads(row["markouts_json"] or "{}")
                _repair_short_marks(db_path, row, marks)
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
                        if engine_id == SCALP_ENGINE and int(horizon) in (30, 60):
                            price = executable_bid_at(db_path, str(row["symbol"]), due, int(horizon))
                        else:
                            price = quote(str(row["symbol"]), due)
                    except Exception:
                        price = None
                    if price is None and moment < due + grace:
                        done = False
                        continue
                    if price is None or ref <= 0 or not math.isfinite(float(price or 0)):
                        marks[key] = None
                    elif engine_id == SCALP_ENGINE and int(horizon) in (30, 60):
                        marks[key] = executable_bid_net(ref, float(price), cost)
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
                stored_life = marks.get("lifecycle")
                if engine_id == DAY_ENGINE and str(row["lifecycle_json"] or "") and not (isinstance(stored_life, dict) and current_lifecycle_label(stored_life.get("reason"))):
                    # A label stored under a retired exit policy is replayed under the current one.
                    marks.pop("lifecycle", None)
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
                uniqueness = 1.0
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
                    if is_directional(raw_source):
                        # A claim row with no stored projection is a 0 projection.
                        raw_move = max(0.0, float(raw_move)) if raw_move is not None else 0.0
                    if forward is not None and raw_move is not None and (is_directional(raw_source) or float(raw_move) > 0):
                        residual = scalp_edge_residual(forward_net=forward, raw_expected_move=float(raw_move), roundtrip_cost=float(row["roundtrip_cost"] or 0))
                    # The micro model learns what the decision-time edge before micro
                    # missed, on claim rows only; ATR-estimate rows never shape live edge.
                    pre_micro = _economic_of(row).get("pre_micro_edge")
                    micro_target = float(forward) - float(pre_micro) if (forward is not None and is_directional(raw_source) and isinstance(pre_micro, (int, float))) else None
                    if is_directional(raw_source) and forward is not None:
                        uniqueness = claim_uniqueness(conn, row, label_h)
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
                                raw=float(raw_move),
                                gross=float(forward) + float(row["roundtrip_cost"] or 0),
                                now=moment,
                                weight=uniqueness,
                            )
                    if engine_id == SCALP_ENGINE and micro_target is not None:
                        with contextlib.suppress(Exception):
                            feats = json.loads(row["features_json"] or "{}")
                            if isinstance(feats, dict) and feats:
                                update_linear_model(db_path, SCALP_ENGINE, "micro_edge", feats, micro_target, now=moment, weight=uniqueness)
                        econ = _economic_of(row)
                        if "micro_unweighted" in econ and "pre_micro_edge" in econ:
                            learn_micro_weight(
                                db_path,
                                strategy_version=str(row["strategy_version"]),
                                tilt=float(econ["micro_unweighted"]),
                                miss=float(forward) - float(econ["pre_micro_edge"]),
                                now=moment,
                                weight=uniqueness,
                            )
                    learned += 1
                life = marks.get("lifecycle")
                if (
                    engine_id == DAY_ENGINE
                    and isinstance(life, dict)
                    and life.get("net") is not None
                    and current_lifecycle_label(life.get("reason"))
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
                if int(row["filled"] or 0) and row["realized_net"] is not None and not int(row["calibration_learned"] or 0):
                    if float(row["evaluated_at"] or 0.0) >= _calibration_backfill_floor():
                        _learn_policy_calibration(conn, db_path, row, float(row["realized_net"]), moment)
                if int(row["filled"] or 0) and row["realized_net"] is not None and not int(row["policy_learned"] or 0):
                    _learn_policy_gap(conn, db_path, row, marks, float(row["realized_net"]), moment)
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
    lo_s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(epoch) - 62.0))
    hi_s = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(epoch) + 1.0))
    try:
        # The bar covering ``epoch`` by time, so a late resolve reads the same
        # minute as a prompt one instead of falling off a newest-bars window.
        for variant in variants:
            try:
                rows = conn.execute(
                    "SELECT ts, close FROM feature_ohlcv WHERE symbol=? AND interval='1m' AND ts>=? AND ts<? ORDER BY ts DESC",
                    (variant, lo_s, hi_s),
                ).fetchall()
            except sqlite3.Error:
                return None
            for ts, close in rows:
                opened = _parse_ts(ts)
                if opened is None or close is None:
                    continue
                if opened <= epoch < opened + 62:
                    return float(close)
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
        "adaptive_state_versions": {DAY_ENGINE: adaptive_format(DAY_ENGINE), SCALP_ENGINE: adaptive_format(SCALP_ENGINE)},
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
                            "claim_gross_mean": round(view["claim_gross_mean"], 6),
                            "claim_gross_setup": round(view["claim_gross_setup"], 6),
                            "claim_capture": round(view["claim_capture"], 4),
                            "claim_raw_center": round(view["claim_raw_center"], 6),
                            "claim_uncertainty": round(view["claim_uncertainty"], 6),
                            "raw_claim_bias": round(view["raw_claim_bias"], 6),
                            "micro_weight": round(view["micro_weight"], 4),
                            "risk_estimate": round(view["risk_estimate"], 6),
                            "target_pct": round(view["target_pct"], 5),
                            "hold_min": round(view["hold_min"], 2),
                            "confidence": round(view["confidence"], 3),
                            "abstain": view["abstain"],
                            "n": view["claim_key_n"],
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


CLOSE_LINEAGE_VERSION = "CLOSE_LINEAGE_V1"
_CONTINUATION_LEARNER_TTL_SEC = 300.0
_continuation_learner_cache: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def entry_lineage(
    *,
    engine: str,
    candidate_id: Any,
    opportunity_id: str,
    rank_position: Any = None,
    rank_of: Any = None,
    rank_score: Any = None,
    size_mult: Any = None,
) -> dict[str, Any]:
    """Stable entry identity stamped on the adaptive decision before the order."""
    try:
        cand = int(candidate_id) if candidate_id is not None else None
    except (TypeError, ValueError):
        cand = None
    return {
        "engine_id": str(engine or "").upper(),
        "candidate_id": cand,
        "opportunity_id": str(opportunity_id or ""),
        "rank_position": int(rank_position) if _finite(rank_position) is not None else None,
        "rank_of": int(rank_of) if _finite(rank_of) is not None else None,
        "rank_score": _finite(rank_score),
        "size_mult": _finite(size_mult),
    }


def continuation_learner(db_path: str, engine: str) -> dict[str, Any]:
    """Version of the continuation state the exit read (installed meta and authority)."""
    engine_id = str(engine or "").upper()
    key = (str(db_path), engine_id)
    hit = _continuation_learner_cache.get(key)
    now = time.time()
    if hit is not None and now - hit[0] < _CONTINUATION_LEARNER_TTL_SEC:
        return dict(hit[1])
    out: dict[str, Any] = {"economic_version": current_economic_version(engine_id), "learning_version": "", "source": "", "installed_at": ""}
    with contextlib.suppress(Exception):
        from backend.services.continuation_surface import advantage_authority, installed_aggregator

        authority = bool(advantage_authority(db_path, engine_id))
        out["authority"] = "HOLD_ADVANTAGE" if authority else "HOLD_REMAINING"
        if authority:
            out["aggregator"] = str(installed_aggregator(db_path, engine_id) or "")
    with contextlib.suppress(sqlite3.Error):
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            row = conn.execute(
                "SELECT learning_version, source, updated_at FROM continuation_learning_meta WHERE engine_id=? AND economic_version=?",
                (engine_id, out["economic_version"]),
            ).fetchone()
        finally:
            conn.close()
        if row is not None:
            out.update({"learning_version": str(row[0] or ""), "source": str(row[1] or ""), "installed_at": str(row[2] or "")})
    _continuation_learner_cache[key] = (now, dict(out))
    return out


def continuation_snapshot(db_path: str, engine: str, *, mark_net: Any, terminal_net: Any, action: str, reason: str, at: float | None = None) -> dict[str, Any]:
    """The latest hold-vs-exit reading of an open position, kept for its close lineage."""
    mark = _finite(mark_net)
    terminal = _finite(terminal_net)
    return {
        "learner": continuation_learner(db_path, engine),
        "at": float(at if at is not None else time.time()),
        "mark_net": mark,
        "terminal_net": terminal,
        "hold_advantage": (terminal - mark) if terminal is not None and mark is not None else None,
        "action": str(action or ""),
        "reason": str(reason or ""),
    }


def close_lineage(
    decision: dict[str, Any] | None,
    *,
    engine: str,
    symbol: str,
    position_trade_id: str,
    sell_trade_id: str,
    opportunity_id: str,
    entry_price: Any,
    exit_price: Any,
    fees_usd: Any,
    net_usd: Any,
    net_pct: Any,
    hold_seconds: Any,
    raw_exit_reason: str,
    exit_reason: str,
    entered_at: Any,
    closed_at: Any,
    continuation: dict[str, Any] | None,
    versions: dict[str, Any] | None,
) -> dict[str, Any]:
    """Entry decision, continuation reading, exit and realized result of one close, by stable ids."""
    decision = decision if isinstance(decision, dict) else {}
    engine_id = str(engine or "").upper()
    entry = decision.get("lineage") if isinstance(decision.get("lineage"), dict) else {}
    if engine_id == SCALP_ENGINE:
        edge = decision.get("executable_edge") if isinstance(decision.get("executable_edge"), dict) else {}
        econ = edge.get("economic") if isinstance(edge.get("economic"), dict) else {}
        market = econ.get("market_edge", edge.get("market_edge_pct"))
        value = econ.get("policy_value", decision.get("final_executable_edge"))
    else:
        econ = decision.get("economic") if isinstance(decision.get("economic"), dict) else {}
        market = econ.get("market_alpha")
        value = econ.get("policy_value", decision.get("expected_net"))
    entry_px, exit_px = _finite(entry_price), _finite(exit_price)
    gross = (exit_px - entry_px) / entry_px if entry_px and exit_px is not None else None
    entered = _epoch_any(entered_at)
    return {
        "lineage_version": CLOSE_LINEAGE_VERSION,
        "engine_id": engine_id,
        "symbol": str(symbol or "").upper(),
        "position_trade_id": str(position_trade_id or ""),
        "sell_trade_id": str(sell_trade_id or ""),
        "opportunity_id": str(opportunity_id or entry.get("opportunity_id") or ""),
        "entry": {
            "candidate_id": entry.get("candidate_id"),
            "economic_version": str(decision.get("economic_version") or ""),
            "adaptive_state_version": str(decision.get("adaptive_state_version") or ""),
            "setup": str(decision.get("setup") or ""),
            "regime": str(decision.get("regime") or ""),
            "market_alpha": _finite(market),
            "policy_gap": _finite(econ.get("policy_gap")),
            "policy_value": _finite(value),
            "rank_position": entry.get("rank_position"),
            "rank_of": entry.get("rank_of"),
            "rank_score": entry.get("rank_score"),
            "size_mult": _finite(entry.get("size_mult", decision.get("size_mult"))),
        },
        "continuation": dict(continuation) if isinstance(continuation, dict) else None,
        "exit": {
            "raw_reason": str(raw_exit_reason or ""),
            "reason": str(exit_reason or ""),
            "contract": exit_contract_of(engine_id, entered_at=entered, exit_reason=str(raw_exit_reason or exit_reason or "")),
            "price": exit_px,
            "closed_at": _epoch_any(closed_at),
        },
        "realized": {
            "entry_price": entry_px,
            "gross_pct": gross,
            "fees_usd": _finite(fees_usd),
            "net_usd": _finite(net_usd),
            "net_pct": _finite(net_pct),
            "hold_seconds": _finite(hold_seconds),
        },
        "versions": {k: str(v) for k, v in (versions or {}).items() if v is not None},
    }


def with_close_lineage(decision: dict[str, Any] | None, lineage: dict[str, Any]) -> dict[str, Any]:
    """The SELL row's adaptive decision: the entry decision plus its close lineage."""
    out = dict(decision) if isinstance(decision, dict) else {}
    out["close_lineage"] = lineage
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
    "CLOSE_LINEAGE_VERSION",
    "DAY_HORIZONS_MIN",
    "SCALP_HORIZONS_SEC",
    "SCALP_TICK_HORIZONS_SEC",
    "abstention_report",
    "adaptive_state_report",
    "calibration_report",
    "close_lineage",
    "continuation_learner",
    "continuation_ratio",
    "continuation_snapshot",
    "current_economic_version",
    "day_candidate_markout_report",
    "day_decision",
    "day_net_expectancy",
    "day_size_mult",
    "entry_lineage",
    "estimate",
    "learn_claim_label",
    "learn_from_close",
    "learn_micro_weight",
    "link_candidate_fill",
    "mark_candidate_filled",
    "market_regime_tag",
    "micro_edge_tilt",
    "micro_weight",
    "observe",
    "ohlcv_quote",
    "persist_trade_adaptive",
    "policy_gap",
    "record_candidate",
    "record_policy_outcome",
    "resolve_markouts",
    "scalp_claim_calibration",
    "scalp_decision",
    "scalp_expected_gross",
    "seed_day_trade_net",
    "update_linear_model",
    "with_close_lineage",
]
