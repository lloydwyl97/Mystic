"""ENTRY_POLICY_VALUE_V3: causal, state-conditioned, continuous regularized DAY entry estimator.

Replaces the coarse 186-cell lookup table of ``ENTRY_POLICY_VALUE_V2`` while
preserving ``ENTRY_POLICY_VALUE_V1`` and ``ENTRY_POLICY_VALUE_V2`` intact as
reproducible historical artifacts.

Core contract:
  1. Target semantics are identical to V2:
     - ``REAL_FUNDED_EXECUTION``: actual fill-to-fill realized accounting net.
     - ``COUNTERFACTUAL_POLICY``: decision-time ask to actual future executable
       policy exit bid under the deployed DAY continuation/catastrophic policy.
     - Targets are NEVER clipped or winsorized; full left-tail crash outcomes
       remain visible to estimation and residual dispersion.
  2. Strict population & directional-regime separation:
     - ``DETECTOR_FIRED`` and ``MARKET_FALLBACK`` (``*__MARKET``) never share
       an unconditional intercept, feature normalizer, or slope vector.
     - Directional BTC regimes (``btcdown``, ``btcflat``, ``btcup``) are fitted
       in isolated submodels so opposite market directions never pool.
  3. Cold-start prior:
     - Zero-information prior equals ``-canonical_roundtrip_cost_pct()``
       (``-6.60 bps`` under current canonical economics): zero market alpha
       minus unavoidable execution friction. No sample-count gate exists.
  4. Two-level causal fundability & correlation clustering:
     - Same ``(symbol, 15m_bar, population)`` setup candidates share one
       fundability cluster whose total fit weight never exceeds ``1.0``.
     - Cross-coin contemporaneous correlation on the same ``15m`` bar is
       measured via Kish's clustered effective sample size ``N_eff`` across
       ``bar_bucket`` weights and scales shared Ridge penalties accordingly.
  5. Causal standardization & deterministic digest:
     - Continuous feature means/stds and ``+-4 sigma`` feature winsorization are fit
       strictly on causal observations with ``label_available_at <= decision_time``.
     - Persisted submodels carry a deterministic SHA-256 ``model_digest``.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import os
import sqlite3
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.config.trading_economics import canonical_roundtrip_cost_pct
from backend.services.adaptive_learning import (
    DAY_ENGINE,
    PRIOR_STRENGTH,
    _invalidate_adaptive_caches,
    _is_forensic_key,
    _norm_symbol,
    _prior,
    half_life_days,
)
from backend.services.candle_contract import _parse_ts
from backend.services.policy_episode import UNIVERSE, canon_symbol

logger = logging.getLogger(__name__)

ENTRY_POLICY_VALUE_V3 = "ENTRY_POLICY_VALUE_V3"
ENTRY_POLICY_NET_V3 = "entry_policy_net_v3"
ENTRY_POLICY_V3_FEATURE_VERSION = "DAY_ENTRY_STATE_V3_14C_32H"

POPULATION_DETECTOR_FIRED = "DETECTOR_FIRED"
POPULATION_MARKET_FALLBACK = "MARKET_FALLBACK"

DIRECTIONAL_REGIMES: tuple[str, ...] = ("btcdown", "btcflat", "btcup")
VOLATILITY_REGIMES: tuple[str, ...] = ("vollo", "volhi")

SETUP_FAMILIES: tuple[str, ...] = (
    "HTF_TREND_PULLBACK",
    "RANGE_BOUNCE",
    "BREAKOUT_CONTINUATION",
    "VWAP_REVERSION",
    "EXHAUSTION_MR",
)
SYMBOLS: tuple[str, ...] = tuple(UNIVERSE)  # ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
SYMBOL_SETUP_PAIRS: tuple[tuple[str, str], ...] = tuple((sym, stp) for sym in SYMBOLS for stp in SETUP_FAMILIES)

# Categorical hierarchy layout inside each isolated (population, directional_regime) submodel:
#   [0]      : submodel intercept (1)
#   [1..5]   : 5 base setup indicators (shared across symbols)
#   [6..7]   : 2 volatility tag indicators (vollo, volhi; shared across symbols)
#   [8..11]  : 4 symbol indicators (per-symbol)
#   [12..31] : 20 exact (symbol, base_setup) indicators (per-symbol)
N_SHARED_CAT = 1 + len(SETUP_FAMILIES) + len(VOLATILITY_REGIMES)  # 8
N_SYMBOL_CAT = len(SYMBOLS) + len(SYMBOL_SETUP_PAIRS)  # 24
N_CAT_DIMS = N_SHARED_CAT + N_SYMBOL_CAT  # 32

V3_CONTINUOUS_FEATURE_NAMES: tuple[str, ...] = (
    "sma20_dist",
    "vwap20_dist",
    "bar_return",
    "ret4_15m",
    "h1_ret",
    "atr15_pct",
    "atr1h_pct",
    "vol_expansion",
    "spread_pct",
    "rsi_centered",
    "bb_centered",
    "range_centered",
    "session_sin",
    "session_cos",
)
N_CONT_DIMS = len(V3_CONTINUOUS_FEATURE_NAMES)  # 14
V3_WINSOR_LIMIT = 4.0
# Per-symbol exact-cell shrinkage strength matches V2 multi-level responsiveness (~37.5% at n=1).
V3_SYMBOL_CELL_STRENGTH = 5.0

V3_FEATURE_CONTRACT: tuple[dict[str, str], ...] = (
    {
        "name": "sma20_dist",
        "semantic_meaning": "Normalized distance of 15m close from 20-bar 15m SMA: (close - sma20) / close",
        "source": "policy_episodes.snapshot_json.features.sma20_dist | day_state_features(bars_15m)",
        "decision_timestamp": "Last completed 15m bar with bar_close_ts <= decided_at",
        "missing_value_behavior": "Marked unobserved (z=0.0 at causal conditional mean; no synthetic raw substitution)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "vwap20_dist",
        "semantic_meaning": "Normalized distance of 15m close from 20-bar 15m typical-price VWAP: (close - vwap20) / close",
        "source": "day_state_features.vwap20_dist | feature_ohlcv (interval='15m', open_ts + 900 <= decided_at)",
        "decision_timestamp": "20 completed 15m bars closed <= decided_at",
        "missing_value_behavior": "Falls back to sma20_dist if positive 20-bar volume unavailable; else unobserved (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "bar_return",
        "semantic_meaning": "Most recently closed 15m bar close-to-close return: (close_0 - close_-1) / close_0",
        "source": "policy_episodes.snapshot_json.features.bar_return | day_state_features(bars_15m)",
        "decision_timestamp": "Last completed 15m bar closed <= decided_at",
        "missing_value_behavior": "Marked unobserved (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "ret4_15m",
        "semantic_meaning": "Short-horizon 60m (4-bar 15m) momentum return: (close_0 - close_-4) / close_0",
        "source": "day_state_features.ret4_15m | feature_ohlcv (interval='15m', open_ts + 900 <= decided_at)",
        "decision_timestamp": "Last 5 completed 15m bars closed <= decided_at",
        "missing_value_behavior": "Marked unobserved if <5 completed 15m bars exist (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "h1_ret",
        "semantic_meaning": "Higher-timeframe 1h momentum return across completed 1h bars",
        "source": "policy_episodes.snapshot_json.features.h1_ret | day_state_features(bars_1h)",
        "decision_timestamp": "Completed 1h bars closed <= decided_at",
        "missing_value_behavior": "Marked unobserved (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "atr15_pct",
        "semantic_meaning": "Continuous 15m Wilder ATR(14) normalized by price: ATR14_15m / close",
        "source": "policy_episodes.snapshot_json.features.atr15_pct | day_state_features(bars_15m)",
        "decision_timestamp": "Completed 15m bars closed <= decided_at",
        "missing_value_behavior": "Marked unobserved (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "atr1h_pct",
        "semantic_meaning": "Continuous 1h Wilder ATR(14) normalized by price: ATR14_1h / close",
        "source": "policy_episodes.snapshot_json.features.atr1h_pct | day_state_features(bars_1h)",
        "decision_timestamp": "Completed 1h bars closed <= decided_at",
        "missing_value_behavior": "Marked unobserved (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "vol_expansion",
        "semantic_meaning": "Intraday volatility expansion ratio vs diffusive sqrt(time) baseline: (atr15_pct / max(0.5 * atr1h_pct, 1e-5)) - 1.0",
        "source": "Derived deterministically at decision time from atr15_pct and atr1h_pct",
        "decision_timestamp": "decided_at",
        "missing_value_behavior": "Marked unobserved when atr15_pct or atr1h_pct unavailable (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "spread_pct",
        "semantic_meaning": "Executable decision-time relative bid-ask spread: (ask - bid) / mid",
        "source": "policy_episodes.entry_spread | snapshot_json.spread | exact_policy_capture.executable_book",
        "decision_timestamp": "decided_at",
        "missing_value_behavior": "Marked unobserved if live/stored spread absent (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "rsi_centered",
        "semantic_meaning": "Centered 15m Wilder RSI(14): (RSI14 / 100.0) - 0.5",
        "source": "policy_episodes.snapshot_json.features.rsi | day_state_features(bars_15m)",
        "decision_timestamp": "Completed 15m bars closed <= decided_at",
        "missing_value_behavior": "Marked unobserved (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "bb_centered",
        "semantic_meaning": "Centered 20-bar 2-sigma Bollinger Band position: bb_pct - 0.5",
        "source": "policy_episodes.snapshot_json.features.bb_pct | day_state_features(bars_15m)",
        "decision_timestamp": "Completed 15m bars closed <= decided_at",
        "missing_value_behavior": "Marked unobserved (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "range_centered",
        "semantic_meaning": "Centered 20-bar high-low range position: range_location - 0.5",
        "source": "policy_episodes.snapshot_json.features.range_location | day_state_features(bars_15m)",
        "decision_timestamp": "Completed 15m bars closed <= decided_at",
        "missing_value_behavior": "Marked unobserved (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "session_sin",
        "semantic_meaning": "UTC session phase sine: sin(2*pi * (decided_at mod 86400) / 86400)",
        "source": "policy_episodes.decided_at | state_features.decision_timestamp",
        "decision_timestamp": "decided_at",
        "missing_value_behavior": "Marked unobserved when decision_timestamp omitted (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
    {
        "name": "session_cos",
        "semantic_meaning": "UTC session phase cosine: cos(2*pi * (decided_at mod 86400) / 86400)",
        "source": "policy_episodes.decided_at | state_features.decision_timestamp",
        "decision_timestamp": "decided_at",
        "missing_value_behavior": "Marked unobserved when decision_timestamp omitted (z=0.0)",
        "normalization_method": "Causal decayed-weight z-score per (population, directional_regime) + [-4.0, +4.0] winsorization",
    },
)

_EPISODE_V3_COLS: tuple[tuple[str, str], ...] = (
    ("entry_policy_v3_learned", "INTEGER NOT NULL DEFAULT 0"),
    ("entry_policy_v3_target", "REAL"),
    ("entry_policy_v3_target_type", "TEXT"),
    ("entry_policy_v3_weight", "REAL"),
    ("entry_policy_v3_cluster_id", "TEXT"),
    ("entry_policy_v3_population", "TEXT"),
    ("entry_policy_v3_dir_regime", "TEXT"),
)

_V3_OBS_DDL = """
CREATE TABLE IF NOT EXISTS entry_policy_v3_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    obs_id TEXT NOT NULL UNIQUE,
    episode_id INTEGER,
    engine_id TEXT NOT NULL,
    economic_version TEXT NOT NULL,
    symbol TEXT NOT NULL,
    setup TEXT NOT NULL,
    base_setup TEXT NOT NULL,
    regime TEXT NOT NULL,
    population TEXT NOT NULL,
    directional_regime TEXT NOT NULL,
    vol_regime TEXT NOT NULL,
    decided_at REAL NOT NULL,
    label_available_at REAL NOT NULL,
    bar_bucket INTEGER NOT NULL,
    cluster_id TEXT NOT NULL,
    cluster_capped INTEGER NOT NULL DEFAULT 1,
    raw_weight REAL NOT NULL,
    sample_weight REAL NOT NULL,
    target REAL NOT NULL,
    target_type TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    features_json TEXT NOT NULL,
    observed_mask_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_V3_OBS_IDX_DDL = """
CREATE INDEX IF NOT EXISTS idx_ep_v3_obs_cell_avail
ON entry_policy_v3_observations(engine_id, economic_version, population, directional_regime, label_available_at)
"""

_V3_OBS_EP_IDX_DDL = """
CREATE INDEX IF NOT EXISTS idx_ep_v3_obs_episode
ON entry_policy_v3_observations(episode_id)
"""

_V3_MODELS_DDL = """
CREATE TABLE IF NOT EXISTS entry_policy_v3_models (
    engine_id TEXT NOT NULL,
    economic_version TEXT NOT NULL,
    population TEXT NOT NULL,
    directional_regime TEXT NOT NULL,
    feature_version TEXT NOT NULL,
    obs_count INTEGER NOT NULL,
    nominal_weight REAL NOT NULL,
    effective_sample_size REAL NOT NULL,
    prior_net REAL NOT NULL,
    winsor_limit REAL NOT NULL,
    residual_sd REAL NOT NULL,
    feature_means_json TEXT NOT NULL,
    feature_stds_json TEXT NOT NULL,
    coefficients_json TEXT NOT NULL,
    cov_inv_json TEXT NOT NULL,
    fitted_at REAL NOT NULL,
    model_digest TEXT NOT NULL,
    PRIMARY KEY (engine_id, economic_version, population, directional_regime)
)
"""

# Process-local submodel fit cache keyed by (abs_db_path, obs_signature, pop, dreg, moment_rounded)
_V3_FIT_CACHE: dict[tuple[str, int, float, str, str, float], dict[str, Any]] = {}


def invalidate_v3_cache() -> None:
    _V3_FIT_CACHE.clear()


def v3_feature_contract() -> tuple[dict[str, str], ...]:
    """Return the immutable V3 decision-time feature specification."""
    return V3_FEATURE_CONTRACT


def v3_prior_net() -> float:
    """Zero-alpha economic expectation: 0 market edge minus canonical round-trip execution cost."""
    return -float(canonical_roundtrip_cost_pct())


def v3_prior_sd() -> float:
    """Prior residual dispersion before observations accumulate, sourced from DAY MAE prior."""
    base = float(_prior(DAY_ENGINE, "trade_mae"))
    return base if base > 0.0 else float(canonical_roundtrip_cost_pct())


def classify_population(setup: str) -> str:
    """Separate true detector-fired setups from ``*__MARKET`` fallback evaluations."""
    s = str(setup or "").strip().upper()
    if s.endswith("__MARKET"):
        return POPULATION_MARKET_FALLBACK
    return POPULATION_DETECTOR_FIRED


def base_setup_name(setup: str) -> str:
    s = str(setup or "").strip().upper()
    if s.endswith("__MARKET"):
        return s[:-8]
    return s


def extract_directional_regime(regime: str) -> tuple[str, str]:
    """Split ``regime`` into ``(directional_regime, volatility_regime)``.

    ``btcdown_vollo`` -> ``('btcdown', 'vollo')``.
    Blank regimes map to ``('untagged', '')`` so blank rows never leak into
    directional BTC regime models.
    """
    raw = str(regime or "").strip().lower()
    if not raw:
        return "untagged", ""
    if "_" in raw:
        parts = raw.split("_", 1)
        return (parts[0] or "untagged"), parts[1]
    return raw, ""


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_v3_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_V3_OBS_DDL)
    conn.execute(_V3_OBS_IDX_DDL)
    conn.execute(_V3_OBS_EP_IDX_DDL)
    conn.execute(_V3_MODELS_DDL)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "policy_episodes" in tables:
        existing = {r[1] for r in conn.execute("PRAGMA table_info(policy_episodes)").fetchall()}
        for col, ddl in _EPISODE_V3_COLS:
            if col not in existing:
                conn.execute(f"ALTER TABLE policy_episodes ADD COLUMN {col} {ddl}")


def build_categorical_vector(symbol: str, base_setup: str, vol_regime: str) -> np.ndarray:
    """Construct the 32-dim hierarchical indicator vector inside ``(population, directional_regime)``."""
    sym = _norm_symbol(symbol)
    bstp = str(base_setup or "").strip().upper()
    vreg = str(vol_regime or "").strip().lower()
    vec = np.zeros(N_CAT_DIMS, dtype=float)
    vec[0] = 1.0
    for idx, stp in enumerate(SETUP_FAMILIES):
        if bstp == stp:
            vec[1 + idx] = 1.0
    if vreg == "vollo":
        vec[6] = 1.0
    elif vreg == "volhi":
        vec[7] = 1.0
    offset_sym = N_SHARED_CAT
    for idx, sy in enumerate(SYMBOLS):
        if sym == sy:
            vec[offset_sym + idx] = 1.0
    offset_pair = offset_sym + len(SYMBOLS)
    for idx, (sy, stp) in enumerate(SYMBOL_SETUP_PAIRS):
        if sym == sy and bstp == stp:
            vec[offset_pair + idx] = 1.0
    return vec


def load_causal_ohlcv_15m_index(
    conn: sqlite3.Connection,
    *,
    min_epoch: float | None = None,
) -> dict[str, tuple[list[float], list[tuple[float, float, float, float, float, float]]]]:
    """Load 15m ``feature_ohlcv`` bars indexed by symbol for fast causal binary-search lookup."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "feature_ohlcv" not in tables:
        return {}
    min_ts_str = "1970-01-01 00:00:00"
    if min_epoch is not None and math.isfinite(min_epoch) and min_epoch > 86400.0 * 5:
        min_ts_str = datetime.fromtimestamp(float(min_epoch) - 86400.0 * 3, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    out: dict[str, tuple[list[float], list[tuple[float, float, float, float, float, float]]]] = {}
    sym_map = {"BTCUSDT": ("BTC-USDT", "BTCUSDT"), "ETHUSDT": ("ETH-USDT", "ETHUSDT"), "SOLUSDT": ("SOL-USDT", "SOLUSDT"), "XRPUSDT": ("XRP-USDT", "XRPUSDT")}
    for norm_sym, aliases in sym_map.items():
        rows = conn.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM feature_ohlcv
            WHERE symbol IN (?, ?) AND interval='15m' AND ts >= ?
            ORDER BY ts ASC
            """,
            (aliases[0], aliases[1], min_ts_str),
        ).fetchall()
        if not rows:
            continue
        parsed: list[tuple[float, float, float, float, float, float]] = []
        for r in rows:
            te = _parse_ts(r[0])
            if te is None:
                continue
            try:
                parsed.append((float(te), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])))
            except (TypeError, ValueError):
                continue
        if parsed:
            parsed.sort(key=lambda item: item[0])
            # Deduplicate identical bar timestamps if both symbol formats existed
            dedup_ts: list[float] = []
            dedup_bars: list[tuple[float, float, float, float, float, float]] = []
            for b in parsed:
                if dedup_ts and dedup_ts[-1] == b[0]:
                    dedup_bars[-1] = b
                else:
                    dedup_ts.append(b[0])
                    dedup_bars.append(b)
            out[norm_sym] = (dedup_ts, dedup_bars)
    return out


def _finite_float(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def extract_v3_continuous_features(
    raw_features: Mapping[str, Any] | None,
    *,
    symbol: str = "",
    decided_at: float | None = None,
    entry_spread: float | None = None,
    ohlcv_index: Mapping[str, tuple[list[float], list[tuple[float, float, float, float, float, float]]]] | None = None,
    include_session_from_decided_at: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | None]]:
    """Extract the 14 causal continuous state features and their observed mask.

    Only completed bars with ``bar_open_ts + 900 <= decided_at + 2.0`` are ever
    consulted from ``ohlcv_index``. Missing values are never approximated from
    future data; unobserved dimensions receive ``observed_mask[j] = False`` and
    evaluate at ``z_j = 0.0`` (the causal conditional mean).
    """
    sf: Mapping[str, Any] = raw_features if isinstance(raw_features, Mapping) else {}
    vals = np.zeros(N_CONT_DIMS, dtype=float)
    mask = np.zeros(N_CONT_DIMS, dtype=bool)

    sma20_d = _finite_float(sf.get("sma20_dist"))
    bar_ret = _finite_float(sf.get("bar_return"))
    h1_ret = _finite_float(sf.get("h1_ret"))
    vwap20_d = _finite_float(sf.get("vwap20_dist"))
    ret4 = _finite_float(sf.get("ret4_15m"))

    norm_sym = _norm_symbol(symbol) if symbol else ""
    if (vwap20_d is None or ret4 is None) and ohlcv_index and norm_sym in ohlcv_index and decided_at is not None:
        ts_list, bar_list = ohlcv_index[norm_sym]
        # A 15m bar opened at T closes at T + 900; allow 2s clock jitter on bar-close stamp
        pos = bisect.bisect_right(ts_list, float(decided_at) - 898.0)
        if pos >= 20:
            w20 = bar_list[pos - 20 : pos]
            c0 = w20[-1][4]
            if vwap20_d is None and c0 > 0.0:
                vol_sum = sum(max(0.0, b[5]) for b in w20)
                if vol_sum > 0.0:
                    vwap20 = sum(((b[2] + b[3] + b[4]) / 3.0) * max(0.0, b[5]) for b in w20) / vol_sum
                    vwap20_d = (c0 - vwap20) / c0
                elif sma20_d is not None:
                    vwap20_d = sma20_d
            if ret4 is None and c0 > 0.0 and w20[-5][4] > 0.0:
                ret4 = (c0 - w20[-5][4]) / c0
        elif pos >= 5 and ret4 is None:
            c0 = bar_list[pos - 1][4]
            cm4 = bar_list[pos - 5][4]
            if c0 > 0.0 and cm4 > 0.0:
                ret4 = (c0 - cm4) / c0

    if vwap20_d is None and sma20_d is not None and not ohlcv_index:
        # When live bars lacked volume, SMA20 is the exact equal-volume VWAP limit
        vwap20_d = sma20_d

    atr15 = _finite_float(sf.get("atr15_pct"))
    atr1h = _finite_float(sf.get("atr1h_pct"))
    vol_exp = _finite_float(sf.get("vol_expansion"))
    if vol_exp is None and atr15 is not None and atr1h is not None and atr1h > 0.0:
        vol_exp = (max(0.0, atr15) / max(0.5 * atr1h, 1e-5)) - 1.0

    sp = _finite_float(entry_spread)
    if sp is None:
        sp = _finite_float(sf.get("spread_pct"))
    if sp is None:
        sp = _finite_float(sf.get("spread"))
    if sp is not None and sp < 0.0:
        sp = None

    rsi_raw = _finite_float(sf.get("rsi"))
    rsi_c = (rsi_raw - 0.5) if rsi_raw is not None else _finite_float(sf.get("rsi_centered"))
    bb_raw = _finite_float(sf.get("bb_pct"))
    bb_c = (bb_raw - 0.5) if bb_raw is not None else _finite_float(sf.get("bb_centered"))
    rng_raw = _finite_float(sf.get("range_location"))
    rng_c = (rng_raw - 0.5) if rng_raw is not None else _finite_float(sf.get("range_centered"))

    dec_ts = _finite_float(sf.get("decision_timestamp"))
    if dec_ts is None and include_session_from_decided_at:
        dec_ts = _finite_float(decided_at)
    sess_sin = _finite_float(sf.get("session_sin"))
    sess_cos = _finite_float(sf.get("session_cos"))
    if (sess_sin is None or sess_cos is None) and dec_ts is not None and dec_ts > 0.0:
        angle = ((dec_ts % 86400.0) / 86400.0) * (2.0 * math.pi)
        sess_sin = math.sin(angle)
        sess_cos = math.cos(angle)

    ordered: list[float | None] = [
        sma20_d,
        vwap20_d,
        bar_ret,
        ret4,
        h1_ret,
        max(0.0, atr15) if atr15 is not None else None,
        max(0.0, atr1h) if atr1h is not None else None,
        vol_exp,
        sp,
        rsi_c,
        bb_c,
        rng_c,
        sess_sin,
        sess_cos,
    ]
    named: dict[str, float | None] = {}
    for idx, (fname, fval) in enumerate(zip(V3_CONTINUOUS_FEATURE_NAMES, ordered, strict=True)):
        if fval is not None and math.isfinite(fval):
            vals[idx] = float(fval)
            mask[idx] = True
            named[fname] = float(fval)
        else:
            named[fname] = None
    return vals, mask, named


def _compute_model_digest(
    *,
    population: str,
    directional_regime: str,
    obs_ids: list[str],
    targets: np.ndarray,
    weights: np.ndarray,
    X_cont: np.ndarray,
    M_cont: np.ndarray,
    prior_net: float,
    means: np.ndarray,
    stds: np.ndarray,
    beta: np.ndarray,
) -> str:
    """Deterministic SHA-256 digest covering all Section 35 state components."""
    hasher = hashlib.sha256()
    header = {
        "economic_version": ENTRY_POLICY_VALUE_V3,
        "feature_version": ENTRY_POLICY_V3_FEATURE_VERSION,
        "population": population,
        "regime_contract": {
            "directional_regime": directional_regime,
            "volatility_regimes": list(VOLATILITY_REGIMES),
            "setup_families": list(SETUP_FAMILIES),
            "symbols": list(SYMBOLS),
        },
        "feature_names": list(V3_CONTINUOUS_FEATURE_NAMES),
        "winsor_limit": V3_WINSOR_LIMIT,
        "obs_count": len(obs_ids),
        "prior_net": round(float(prior_net), 12),
    }
    hasher.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    hasher.update(b"|obs_ids:")
    hasher.update("\n".join(obs_ids).encode("utf-8"))
    hasher.update(b"|targets:")
    hasher.update((np.round(targets, 12) + 0.0).astype("<f8").tobytes())
    hasher.update(b"|weights:")
    hasher.update((np.round(weights, 12) + 0.0).astype("<f8").tobytes())
    hasher.update(b"|features:")
    masked_x = np.where(M_cont, np.round(X_cont, 10) + 0.0, 0.0)
    hasher.update(masked_x.astype("<f8").tobytes())
    hasher.update(M_cont.astype(np.uint8).tobytes())
    hasher.update(b"|means:")
    hasher.update((np.round(means, 12) + 0.0).astype("<f8").tobytes())
    hasher.update(b"|stds:")
    hasher.update((np.round(stds, 12) + 0.0).astype("<f8").tobytes())
    hasher.update(b"|beta:")
    hasher.update((np.round(beta, 12) + 0.0).astype("<f8").tobytes())
    return hasher.hexdigest()


def _cold_digest(population: str, directional_regime: str, prior_net: float) -> str:
    return _compute_model_digest(
        population=population,
        directional_regime=directional_regime,
        obs_ids=[],
        targets=np.zeros(0, dtype=float),
        weights=np.zeros(0, dtype=float),
        X_cont=np.zeros((0, N_CONT_DIMS), dtype=float),
        M_cont=np.zeros((0, N_CONT_DIMS), dtype=bool),
        prior_net=prior_net,
        means=np.zeros(N_CONT_DIMS, dtype=float),
        stds=np.ones(N_CONT_DIMS, dtype=float),
        beta=np.zeros(N_CAT_DIMS + N_CONT_DIMS, dtype=float),
    )


def fit_v3_submodel(
    observations: list[dict[str, Any]],
    *,
    population: str,
    directional_regime: str,
    now_epoch: float,
) -> dict[str, Any]:
    """Fit one causal regularized linear submodel for ``(population, directional_regime)``.

    Enforces:
      - Cluster cap: sum of raw weights per ``cluster_id`` (``symbol:bar:population``)
        cannot exceed ``1.0`` when ``cluster_capped=True``.
      - 14-day exponential decay from ``label_available_at`` to ``now_epoch``.
      - Kish cross-coin bar-cluster effective sample size ``N_eff`` over ``bar_bucket``.
      - Causal feature mean/std and ``+-4 sigma`` feature winsorization (targets unclipped).
    """
    prior_net = v3_prior_net()
    prior_sd = v3_prior_sd()
    if not observations:
        return {
            "population": population,
            "directional_regime": directional_regime,
            "obs_count": 0,
            "nominal_weight": 0.0,
            "effective_sample_size": 0.0,
            "prior_net": prior_net,
            "winsor_limit": V3_WINSOR_LIMIT,
            "means": np.zeros(N_CONT_DIMS, dtype=float),
            "stds": np.ones(N_CONT_DIMS, dtype=float),
            "obs_counts_cont": np.zeros(N_CONT_DIMS, dtype=int),
            "beta": np.zeros(N_CAT_DIMS + N_CONT_DIMS, dtype=float),
            "a_eff_inv": np.diag(
                np.concatenate(
                    [
                        np.full(N_SHARED_CAT, 1.0 / float(PRIOR_STRENGTH)),
                        np.full(N_SYMBOL_CAT, 1.0 / float(V3_SYMBOL_CELL_STRENGTH)),
                        np.full(N_CONT_DIMS, 1.0 / (2.0 * N_CONT_DIMS * float(PRIOR_STRENGTH))),
                    ]
                )
            ),
            "residual_sd": prior_sd,
            "fitted_at": float(now_epoch),
            "model_digest": _cold_digest(population, directional_regime, prior_net),
            "by_sym_stp_w": {},
            "by_sym_stp_bars": {},
        }

    # 1. Bound same-(symbol, 15m_bar, population) cluster raw weights to <= 1.0
    cluster_sums: dict[str, float] = defaultdict(float)
    for obs in observations:
        if obs.get("cluster_capped", True):
            cluster_sums[str(obs["cluster_id"])] += max(0.0, float(obs["sample_weight"]))

    hl_sec = float(half_life_days(DAY_ENGINE)) * 86400.0
    ln2_hl = math.log(2.0) / hl_sec if hl_sec > 0.0 else 0.0

    k = len(observations)
    obs_ids: list[str] = []
    bars = np.empty(k, dtype=np.int64)
    w = np.empty(k, dtype=float)
    targets = np.empty(k, dtype=float)
    X_cat = np.empty((k, N_CAT_DIMS), dtype=float)
    X_cont = np.empty((k, N_CONT_DIMS), dtype=float)
    M_cont = np.empty((k, N_CONT_DIMS), dtype=bool)

    by_sym_stp_w: dict[tuple[str, str], float] = defaultdict(float)
    by_sym_stp_bars: dict[tuple[str, str], dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for idx, obs in enumerate(observations):
        cid = str(obs["cluster_id"])
        raw_w = max(0.0, float(obs["sample_weight"]))
        if obs.get("cluster_capped", True):
            csum = cluster_sums[cid]
            bounded_w = raw_w / max(1.0, csum) if csum > 0.0 else 0.0
        else:
            bounded_w = raw_w
        age = max(0.0, float(now_epoch) - float(obs["label_available_at"]))
        wi = bounded_w * math.exp(-ln2_hl * age)
        w[idx] = wi
        obs_ids.append(str(obs["obs_id"]))
        bar_b = int(obs["bar_bucket"])
        bars[idx] = bar_b
        targets[idx] = float(obs["target"])
        X_cat[idx] = obs["x_cat"]
        X_cont[idx] = obs["x_cont"]
        M_cont[idx] = obs["m_cont"]
        key_pair = (str(obs["symbol"]), str(obs["base_setup"]))
        by_sym_stp_w[key_pair] += wi
        by_sym_stp_bars[key_pair][bar_b] += wi

    w_sum = float(np.sum(w))
    if w_sum <= 1e-15:
        return fit_v3_submodel([], population=population, directional_regime=directional_regime, now_epoch=now_epoch)

    # 2. Cross-coin 15m bar-cluster Kish effective sample size
    _, inv = np.unique(bars, return_inverse=True)
    bar_weights = np.bincount(inv, weights=w)
    neff_bar = (w_sum * w_sum) / max(float(np.sum(bar_weights * bar_weights)), 1e-18)
    infl = max(1.0, w_sum / max(neff_bar, 1e-12))

    # 3. Causal feature standardization & +-4 sigma feature winsorization (only over observed entries)
    means = np.zeros(N_CONT_DIMS, dtype=float)
    stds = np.ones(N_CONT_DIMS, dtype=float)
    obs_counts_cont = np.zeros(N_CONT_DIMS, dtype=int)
    Z_cont = np.zeros((k, N_CONT_DIMS), dtype=float)
    for j in range(N_CONT_DIMS):
        mj = M_cont[:, j]
        cnt_j = int(np.sum(mj))
        obs_counts_cont[j] = cnt_j
        if cnt_j <= 0:
            continue
        wj = w[mj]
        wj_sum = float(np.sum(wj))
        if wj_sum <= 1e-15:
            continue
        xj = X_cont[mj, j]
        muj = float(np.sum(wj * xj) / wj_sum)
        means[j] = muj
        if cnt_j >= 2:
            varj = float(np.sum(wj * ((xj - muj) ** 2)) / wj_sum)
            stdj = math.sqrt(max(varj, 1e-8))
            stds[j] = stdj
            Z_cont[mj, j] = np.clip((xj - muj) / stdj, -V3_WINSOR_LIMIT, V3_WINSOR_LIMIT)
        else:
            stds[j] = 1.0
            Z_cont[mj, j] = 0.0

    # 4. Hierarchical + continuous Gauss-Markov Ridge estimation around prior_net
    X = np.hstack([X_cat, Z_cont])
    y_exc = targets - prior_net

    lam = np.empty(N_CAT_DIMS + N_CONT_DIMS, dtype=float)
    lam[:N_SHARED_CAT] = float(PRIOR_STRENGTH) * infl
    lam[N_SHARED_CAT:N_CAT_DIMS] = float(V3_SYMBOL_CELL_STRENGTH)
    lam[N_CAT_DIMS:] = 2.0 * float(N_CONT_DIMS) * float(PRIOR_STRENGTH) * infl

    WX = X * w[:, None]
    xtwx = X.T @ WX
    A = xtwx + np.diag(lam)
    rhs = WX.T @ y_exc
    beta = np.linalg.solve(A, rhs)

    # Effective precision matrix (deflated for bar clustering on shared blocks) for prediction SE
    lam_base = np.empty(N_CAT_DIMS + N_CONT_DIMS, dtype=float)
    lam_base[:N_SHARED_CAT] = float(PRIOR_STRENGTH)
    lam_base[N_SHARED_CAT:N_CAT_DIMS] = float(V3_SYMBOL_CELL_STRENGTH)
    lam_base[N_CAT_DIMS:] = 2.0 * float(N_CONT_DIMS) * float(PRIOR_STRENGTH)
    a_eff = (xtwx / infl) + np.diag(lam_base)
    try:
        a_eff_inv = np.linalg.inv(a_eff)
    except np.linalg.LinAlgError:
        a_eff_inv = np.linalg.pinv(a_eff)

    # 5. Unclipped residual dispersion shrunk from prior_sd toward empirical residual SD
    preds = prior_net + (X @ beta)
    residuals = targets - preds
    rss_w = float(np.sum(w * (residuals**2)))
    res_var = (float(PRIOR_STRENGTH) * (prior_sd**2) + rss_w) / (float(PRIOR_STRENGTH) + neff_bar)
    residual_sd = math.sqrt(max(res_var, 1e-12))

    digest = _compute_model_digest(
        population=population,
        directional_regime=directional_regime,
        obs_ids=obs_ids,
        targets=targets,
        weights=w,
        X_cont=X_cont,
        M_cont=M_cont,
        prior_net=prior_net,
        means=means,
        stds=stds,
        beta=beta,
    )

    return {
        "population": population,
        "directional_regime": directional_regime,
        "obs_count": k,
        "nominal_weight": w_sum,
        "effective_sample_size": neff_bar,
        "prior_net": prior_net,
        "winsor_limit": V3_WINSOR_LIMIT,
        "means": means,
        "stds": stds,
        "obs_counts_cont": obs_counts_cont,
        "beta": beta,
        "a_eff_inv": a_eff_inv,
        "residual_sd": residual_sd,
        "fitted_at": float(now_epoch),
        "model_digest": digest,
        "by_sym_stp_w": dict(by_sym_stp_w),
        "by_sym_stp_bars": {kp: dict(bm) for kp, bm in by_sym_stp_bars.items()},
    }


def predict_v3_submodel(
    submodel: Mapping[str, Any],
    *,
    symbol: str,
    setup: str,
    regime: str,
    x_cont: np.ndarray,
    m_cont: np.ndarray,
    eval_timestamp: float,
) -> dict[str, Any]:
    """Score one DAY candidate state from a fitted ``(population, directional_regime)`` submodel."""
    sym = _norm_symbol(symbol)
    stp_full = str(setup or "").strip().upper()
    bstp = base_setup_name(stp_full)
    pop = classify_population(stp_full)
    dreg, vreg = extract_directional_regime(regime)

    x_cat = build_categorical_vector(sym, bstp, vreg)
    z_cont = np.zeros(N_CONT_DIMS, dtype=float)
    obs_counts_cont = submodel["obs_counts_cont"]
    means = submodel["means"]
    stds = submodel["stds"]
    for j in range(N_CONT_DIMS):
        if bool(m_cont[j]) and int(obs_counts_cont[j]) >= 2:
            z_cont[j] = float(np.clip((float(x_cont[j]) - float(means[j])) / float(stds[j]), -V3_WINSOR_LIMIT, V3_WINSOR_LIMIT))

    x = np.concatenate([x_cat, z_cont])
    beta = submodel["beta"]
    prior_net = float(submodel["prior_net"])
    expected_net = prior_net + float(x @ beta)

    # Decompose hierarchical contributions for transparent diagnostics
    icpt_contrib = float(x_cat[:N_SHARED_CAT] @ beta[:N_SHARED_CAT])
    sym_contrib = float(x_cat[N_SHARED_CAT:N_CAT_DIMS] @ beta[N_SHARED_CAT:N_CAT_DIMS])
    state_contrib = float(z_cont @ beta[N_CAT_DIMS:])

    res_sd = float(submodel["residual_sd"])
    a_eff_inv = submodel["a_eff_inv"]
    quad = max(0.0, float(x @ a_eff_inv @ x))
    pred_se = res_sd * math.sqrt(quad)
    # Total uncertainty = residual outcome dispersion (bounded away from zero, never a gate)
    # plus parameter estimation SE when observations exist
    if int(submodel["obs_count"]) == 0:
        total_sd = res_sd
    else:
        total_sd = math.sqrt(res_sd * res_sd + pred_se * pred_se)

    neff = float(submodel["effective_sample_size"])
    nom_w = float(submodel["nominal_weight"])
    key_pair = (sym, bstp)
    key_nom_w = float(submodel["by_sym_stp_w"].get(key_pair, 0.0))
    key_bars = submodel["by_sym_stp_bars"].get(key_pair, {})
    if key_bars and key_nom_w > 1e-15:
        kb_vals = np.array(list(key_bars.values()), dtype=float)
        key_neff = float((key_nom_w * key_nom_w) / max(float(np.sum(kb_vals * kb_vals)), 1e-18))
    else:
        key_neff = 0.0

    conf = neff / (neff + float(PRIOR_STRENGTH)) if neff > 0.0 else 0.0
    key_wt = key_neff / (key_neff + float(V3_SYMBOL_CELL_STRENGTH)) if key_neff > 0.0 else 0.0
    eng_wt = neff / (neff + float(PRIOR_STRENGTH)) if neff > 0.0 else 0.0

    parent_net = prior_net + icpt_contrib
    key_level_net = prior_net + icpt_contrib + sym_contrib

    return {
        "mean": expected_net,
        "expected_policy_net": expected_net,
        "expected_policy_net_bps": expected_net * 1e4,
        "prior": prior_net,
        "parent": parent_net,
        "n": neff,
        "effective_sample_size": neff,
        "nominal_weight": nom_w,
        "key_effective_sample_size": key_neff,
        "key_nominal_weight": key_nom_w,
        "model_sample_weight": neff,
        "confidence": conf,
        "key_weight": key_wt,
        "engine_weight": eng_wt,
        "sd": total_sd,
        "residual_sd": res_sd,
        "prediction_se": pred_se,
        "uncertainty": total_sd,
        "engine_sd": res_sd,
        "setup_sd": res_sd,
        "related_sd": res_sd,
        "key_sd": total_sd,
        "state_tilt": state_contrib,
        "levels": {
            "engine": prior_net + float(beta[0]),
            "setup": parent_net,
            "related": key_level_net,
            "key": expected_net,
        },
        "level_weights": {
            "engine": eng_wt,
            "setup": eng_wt,
            "related": key_wt,
            "key": key_wt,
        },
        "metric": ENTRY_POLICY_NET_V3,
        "entry_policy_version": ENTRY_POLICY_VALUE_V3,
        "feature_version": ENTRY_POLICY_V3_FEATURE_VERSION,
        "model_digest": str(submodel["model_digest"]),
        "population": pop,
        "directional_regime": dreg,
        "vol_regime": vreg,
        "timestamp": float(eval_timestamp),
    }


def _persist_submodel(conn: sqlite3.Connection, submodel: Mapping[str, Any]) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO entry_policy_v3_models (
            engine_id, economic_version, population, directional_regime, feature_version,
            obs_count, nominal_weight, effective_sample_size, prior_net, winsor_limit,
            residual_sd, feature_means_json, feature_stds_json, coefficients_json,
            cov_inv_json, fitted_at, model_digest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            DAY_ENGINE,
            ENTRY_POLICY_VALUE_V3,
            str(submodel["population"]),
            str(submodel["directional_regime"]),
            ENTRY_POLICY_V3_FEATURE_VERSION,
            int(submodel["obs_count"]),
            float(submodel["nominal_weight"]),
            float(submodel["effective_sample_size"]),
            float(submodel["prior_net"]),
            float(submodel["winsor_limit"]),
            float(submodel["residual_sd"]),
            json.dumps([float(v) for v in submodel["means"]], separators=(",", ":")),
            json.dumps([float(v) for v in submodel["stds"]], separators=(",", ":")),
            json.dumps([float(v) for v in submodel["beta"]], separators=(",", ":")),
            json.dumps([[float(v) for v in row] for row in submodel["a_eff_inv"]], separators=(",", ":")),
            float(submodel["fitted_at"]),
            str(submodel["model_digest"]),
        ),
    )


def _row_to_obs_dict(row: sqlite3.Row | Mapping[str, Any]) -> dict[str, Any]:
    sym = _norm_symbol(str(row["symbol"]))
    bstp = str(row["base_setup"])
    vreg = str(row["vol_regime"])
    fdict = json.loads(str(row["features_json"] or "{}"))
    mdict = json.loads(str(row["observed_mask_json"] or "{}"))
    x_cont = np.array([float(fdict.get(fn) or 0.0) for fn in V3_CONTINUOUS_FEATURE_NAMES], dtype=float)
    m_cont = np.array([bool(mdict.get(fn, False)) for fn in V3_CONTINUOUS_FEATURE_NAMES], dtype=bool)
    return {
        "obs_id": str(row["obs_id"]),
        "symbol": sym,
        "base_setup": bstp,
        "vol_regime": vreg,
        "decided_at": float(row["decided_at"]),
        "label_available_at": float(row["label_available_at"]),
        "bar_bucket": int(row["bar_bucket"]),
        "cluster_id": str(row["cluster_id"]),
        "cluster_capped": bool(int(row["cluster_capped"])),
        "sample_weight": float(row["sample_weight"]),
        "target": float(row["target"]),
        "x_cat": build_categorical_vector(sym, bstp, vreg),
        "x_cont": x_cont,
        "m_cont": m_cont,
    }


def entry_policy_value_v3(
    db_path: str,
    symbol: str,
    setup: str,
    regime: str,
    *,
    features: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Evaluate ``ENTRY_POLICY_VALUE_V3`` at ``now`` (or data-clock if ``now is None``)."""
    sym = _norm_symbol(symbol)
    stp_full = str(setup or "").strip().upper()
    pop = classify_population(stp_full)
    dreg, _vreg = extract_directional_regime(regime)

    if not db_path or not os.path.exists(db_path):
        moment = float(now) if now is not None else 0.0
        cold_sub = fit_v3_submodel([], population=pop, directional_regime=dreg, now_epoch=moment)
        xc, mc, _ = extract_v3_continuous_features(features, symbol=sym, decided_at=now)
        return predict_v3_submodel(
            cold_sub,
            symbol=sym,
            setup=stp_full,
            regime=regime,
            x_cont=xc,
            m_cont=mc,
            eval_timestamp=moment,
        )

    abs_db = os.path.abspath(db_path)
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        moment = float(now) if now is not None else 0.0
        cold_sub = fit_v3_submodel([], population=pop, directional_regime=dreg, now_epoch=moment)
        xc, mc, _ = extract_v3_continuous_features(features, symbol=sym, decided_at=now)
        return predict_v3_submodel(
            cold_sub,
            symbol=sym,
            setup=stp_full,
            regime=regime,
            x_cont=xc,
            m_cont=mc,
            eval_timestamp=moment,
        )

    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "entry_policy_v3_observations" not in tables:
            moment = float(now) if now is not None else 0.0
            cold_sub = fit_v3_submodel([], population=pop, directional_regime=dreg, now_epoch=moment)
            xc, mc, _ = extract_v3_continuous_features(features, symbol=sym, decided_at=now)
            return predict_v3_submodel(
                cold_sub,
                symbol=sym,
                setup=stp_full,
                regime=regime,
                x_cont=xc,
                m_cont=mc,
                eval_timestamp=moment,
            )

        if now is None:
            max_row = conn.execute(
                "SELECT MAX(label_available_at) FROM entry_policy_v3_observations WHERE engine_id=? AND economic_version=?",
                (DAY_ENGINE, ENTRY_POLICY_VALUE_V3),
            ).fetchone()
            moment = float(max_row[0]) if (max_row and max_row[0] is not None) else 0.0
        else:
            moment = float(now)

        sig_row = conn.execute(
            """
            SELECT COUNT(*), COALESCE(MAX(id), 0), COALESCE(MAX(label_available_at), 0.0)
            FROM entry_policy_v3_observations
            WHERE engine_id=? AND economic_version=? AND population=? AND directional_regime=?
              AND label_available_at <= ?
            """,
            (DAY_ENGINE, ENTRY_POLICY_VALUE_V3, pop, dreg, moment),
        ).fetchone()
        obs_cnt = int(sig_row[0] or 0)
        max_id = int(sig_row[1] or 0)
        max_avail = float(sig_row[2] or 0.0)
        cache_moment = round(moment, 3)
        cache_key = (abs_db, max_id, max_avail, pop, dreg, cache_moment)
        submodel = _V3_FIT_CACHE.get(cache_key)
        if submodel is None:
            if obs_cnt == 0:
                submodel = fit_v3_submodel([], population=pop, directional_regime=dreg, now_epoch=moment)
            else:
                rows = conn.execute(
                    """
                    SELECT obs_id, symbol, base_setup, vol_regime, decided_at, label_available_at,
                           bar_bucket, cluster_id, cluster_capped, sample_weight, target,
                           features_json, observed_mask_json
                    FROM entry_policy_v3_observations
                    WHERE engine_id=? AND economic_version=? AND population=? AND directional_regime=?
                      AND label_available_at <= ?
                    ORDER BY label_available_at ASC, id ASC
                    """,
                    (DAY_ENGINE, ENTRY_POLICY_VALUE_V3, pop, dreg, moment),
                ).fetchall()
                obs_list = [_row_to_obs_dict(r) for r in rows]
                submodel = fit_v3_submodel(
                    obs_list,
                    population=pop,
                    directional_regime=dreg,
                    now_epoch=moment,
                )
            if len(_V3_FIT_CACHE) > 256:
                _V3_FIT_CACHE.clear()
            _V3_FIT_CACHE[cache_key] = submodel

        xc, mc, _ = extract_v3_continuous_features(features, symbol=sym, decided_at=moment)
        return predict_v3_submodel(
            submodel,
            symbol=sym,
            setup=stp_full,
            regime=regime,
            x_cont=xc,
            m_cont=mc,
            eval_timestamp=moment,
        )
    finally:
        conn.close()


def record_entry_policy_v3_observation(
    db_path: str,
    *,
    symbol: str,
    setup: str,
    regime: str,
    value: float,
    weight: float = 1.0,
    now: float | None = None,
    decided_at: float | None = None,
    features: Mapping[str, Any] | None = None,
    target_type: str = "DIRECT_OBSERVATION",
    episode_id: int | None = None,
    obs_id: str | None = None,
    cluster_id: str | None = None,
    persist_model: bool = True,
) -> bool:
    """Record one causal V3 observation (and optionally update persisted submodel state)."""
    if not math.isfinite(float(value)) or float(weight) <= 0.0:
        return False
    sym = _norm_symbol(symbol)
    stp_full = str(setup or "").strip().upper()
    reg_norm = str(regime or "").strip().lower()
    if not sym or not stp_full:
        return False
    if _is_forensic_key(DAY_ENGINE, sym, stp_full, reg_norm):
        return False
    bstp = base_setup_name(stp_full)
    pop = classify_population(stp_full)
    dreg, vreg = extract_directional_regime(regime)
    avail_t = float(now) if now is not None else float(datetime.now(timezone.utc).timestamp())

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        ensure_v3_schema(conn)
        if decided_at is not None and math.isfinite(float(decided_at)):
            dec_t = float(decided_at)
            bar_b = int(dec_t // 900.0) * 900
            cid = str(cluster_id) if cluster_id else f"{sym}:{bar_b}:{pop}"
            capped = 1
            include_sess = True
        else:
            # Direct synthetic observation (e.g. unit-test teach_policy helper without bar decision time):
            # assign sequential synthetic slot so repeated calls at fixed T0 represent distinct samples.
            seq_row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM entry_policy_v3_observations").fetchone()
            seq = int(seq_row[0] or 1)
            dec_t = avail_t
            bar_b = int(avail_t // 900.0) * 900 + seq * 900
            cid = str(cluster_id) if cluster_id else f"direct:{seq}"
            capped = 1 if cluster_id else 0
            include_sess = False

        _xc, mc, named = extract_v3_continuous_features(
            features,
            symbol=sym,
            decided_at=dec_t,
            include_session_from_decided_at=include_sess,
        )
        f_json = json.dumps({k: (named[k] if named[k] is not None else 0.0) for k in V3_CONTINUOUS_FEATURE_NAMES}, sort_keys=True, separators=(",", ":"))
        m_json = json.dumps({k: bool(mc[i]) for i, k in enumerate(V3_CONTINUOUS_FEATURE_NAMES)}, sort_keys=True, separators=(",", ":"))

        if not obs_id:
            if episode_id is not None:
                obs_id = f"ep:{int(episode_id)}"
            else:
                seq_row = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM entry_policy_v3_observations").fetchone()
                obs_id = f"direct:{int(seq_row[0] or 1)}:{avail_t:.6f}"

        conn.execute(
            """
            INSERT OR REPLACE INTO entry_policy_v3_observations (
                obs_id, episode_id, engine_id, economic_version, symbol, setup, base_setup,
                regime, population, directional_regime, vol_regime, decided_at, label_available_at,
                bar_bucket, cluster_id, cluster_capped, raw_weight, sample_weight, target,
                target_type, feature_version, features_json, observed_mask_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                obs_id,
                int(episode_id) if episode_id is not None else None,
                DAY_ENGINE,
                ENTRY_POLICY_VALUE_V3,
                sym,
                stp_full,
                bstp,
                str(regime or "").strip().lower(),
                pop,
                dreg,
                vreg,
                dec_t,
                avail_t,
                bar_b,
                cid,
                capped,
                float(weight),
                float(weight),
                float(value),
                str(target_type),
                ENTRY_POLICY_V3_FEATURE_VERSION,
                f_json,
                m_json,
                _stamp(avail_t),
            ),
        )
        if persist_model:
            rows = conn.execute(
                """
                SELECT obs_id, symbol, base_setup, vol_regime, decided_at, label_available_at,
                       bar_bucket, cluster_id, cluster_capped, sample_weight, target,
                       features_json, observed_mask_json
                FROM entry_policy_v3_observations
                WHERE engine_id=? AND economic_version=? AND population=? AND directional_regime=?
                  AND label_available_at <= ?
                ORDER BY label_available_at ASC, id ASC
                """,
                (DAY_ENGINE, ENTRY_POLICY_VALUE_V3, pop, dreg, avail_t),
            ).fetchall()
            sub = fit_v3_submodel(
                [_row_to_obs_dict(r) for r in rows],
                population=pop,
                directional_regime=dreg,
                now_epoch=avail_t,
            )
            _persist_submodel(conn, sub)
        conn.commit()
    finally:
        conn.close()
    invalidate_v3_cache()
    _invalidate_adaptive_caches(db_path)
    return True


def apply_resolved_entry_policy_v3_labels(
    db_path: str,
    *,
    now: float | None = None,
) -> int:
    """Fold newly resolved ``policy_episodes`` into ``ENTRY_POLICY_VALUE_V3`` causally."""
    from backend.services.day_v2.entry_policy_value import (
        LABEL_REAL_FUNDED_EXECUTION,
        _fetch_episode_rows,
        plan_causal_v2_episodes,
    )

    if not db_path or not os.path.exists(db_path):
        return 0
    cutoff = float(now) if now is not None else float(datetime.now(timezone.utc).timestamp())

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "policy_episodes" not in tables:
            return 0
        ensure_v3_schema(conn)
        pending_check = conn.execute(
            """
            SELECT 1 FROM policy_episodes
            WHERE status='CLOSED'
              AND capture_version='EXACT_POLICY_V1'
              AND engine_id=?
              AND label_available_at IS NOT NULL
              AND label_available_at <= ?
              AND COALESCE(entry_policy_v3_learned, 0)=0
            LIMIT 1
            """,
            (DAY_ENGINE, cutoff),
        ).fetchone()
        if pending_check is None:
            return 0

        all_rows = _fetch_episode_rows(conn)
        v3_learned_map = {
            int(r[0]): int(r[1] or 0)
            for r in conn.execute(
                "SELECT id, COALESCE(entry_policy_v3_learned, 0) FROM policy_episodes WHERE engine_id=?",
                (DAY_ENGINE,),
            ).fetchall()
        }
        plan = plan_causal_v2_episodes(all_rows, cutoff=cutoff)
        pending_rows = [r for r in all_rows if v3_learned_map.get(int(r["id"] or 0), 0) == 0 and int(r["id"] or 0) in plan]
        if not pending_rows:
            return 0

        # Count active setups per (symbol, bar_bucket, population) using only decision-time metadata
        sym_bar_pop_setups: dict[tuple[str, int, str], set[str]] = defaultdict(set)
        for r in all_rows:
            rid = int(r["id"] or 0)
            p = plan.get(rid)
            if not p or p.get("overlap_group") == "SHADOW_OF_REAL":
                continue
            sym = _norm_symbol(str(r["symbol"] or ""))
            stp = str(r["setup"] or "").strip().upper()
            pop = classify_population(stp)
            bar_b = int(float(r["decided_at"]) // 900.0) * 900
            sym_bar_pop_setups[(sym, bar_b, pop)].add(stp)

        min_dec = min(float(r["decided_at"]) for r in pending_rows)
        ohlcv_idx = load_causal_ohlcv_15m_index(conn, min_epoch=min_dec)
        pending_rows.sort(key=lambda r: (float(r["label_available_at"]), int(r["id"] or 0)))

        # Read snapshot_json & entry_spread for pending episodes
        pending_ids = [int(r["id"]) for r in pending_rows]
        snap_by_id: dict[int, tuple[str, float | None]] = {}
        chunk = 500
        for start in range(0, len(pending_ids), chunk):
            sub_ids = pending_ids[start : start + chunk]
            ph = ",".join("?" for _ in sub_ids)
            for srow in conn.execute(
                f"SELECT id, COALESCE(snapshot_json, '{{}}'), entry_spread FROM policy_episodes WHERE id IN ({ph})",
                sub_ids,
            ).fetchall():
                snap_by_id[int(srow[0])] = (str(srow[1] or "{}"), _finite_float(srow[2]))

        touched_cells: set[tuple[str, str]] = set()
        applied = 0
        conn.execute("SAVEPOINT mystic_entry_policy_v3")
        try:
            for r in pending_rows:
                rid = int(r["id"])
                p = plan[rid]
                sym = _norm_symbol(str(r["symbol"] or ""))
                stp_full = str(r["setup"] or "").strip().upper()
                bstp = base_setup_name(stp_full)
                reg = str(r["regime"] or "").strip().lower()
                pop = classify_population(stp_full)
                dreg, vreg = extract_directional_regime(reg)
                dec_t = float(r["decided_at"])
                avail_t = float(p["label_available_at"])
                bar_b = int(dec_t // 900.0) * 900
                cid = f"{sym}:{bar_b}:{pop}"

                is_real = p.get("target_type") == LABEL_REAL_FUNDED_EXECUTION
                n_sib = max(1, len(sym_bar_pop_setups[(sym, bar_b, pop)]))
                w_fit = 1.0 if is_real else (float(p["specific_weight"]) / float(n_sib))

                if p.get("fold") and w_fit > 0.0 and p.get("target") is not None:
                    snap_raw, ep_spread = snap_by_id.get(rid, ("{}", None))
                    try:
                        snap_obj = json.loads(snap_raw)
                    except json.JSONDecodeError:
                        snap_obj = {}
                    sf = snap_obj.get("features") if isinstance(snap_obj, dict) else {}
                    if ep_spread is None and isinstance(snap_obj, dict):
                        ep_spread = _finite_float(snap_obj.get("spread"))
                    _xc, mc, named = extract_v3_continuous_features(
                        sf if isinstance(sf, Mapping) else {},
                        symbol=sym,
                        decided_at=dec_t,
                        entry_spread=ep_spread,
                        ohlcv_index=ohlcv_idx,
                        include_session_from_decided_at=True,
                    )
                    f_json = json.dumps(
                        {k: (named[k] if named[k] is not None else 0.0) for k in V3_CONTINUOUS_FEATURE_NAMES},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    m_json = json.dumps(
                        {k: bool(mc[i]) for i, k in enumerate(V3_CONTINUOUS_FEATURE_NAMES)},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO entry_policy_v3_observations (
                            obs_id, episode_id, engine_id, economic_version, symbol, setup, base_setup,
                            regime, population, directional_regime, vol_regime, decided_at, label_available_at,
                            bar_bucket, cluster_id, cluster_capped, raw_weight, sample_weight, target,
                            target_type, feature_version, features_json, observed_mask_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            f"ep:{rid}",
                            rid,
                            DAY_ENGINE,
                            ENTRY_POLICY_VALUE_V3,
                            sym,
                            stp_full,
                            bstp,
                            reg,
                            pop,
                            dreg,
                            vreg,
                            dec_t,
                            avail_t,
                            bar_b,
                            cid,
                            float(p["specific_weight"]),
                            float(w_fit),
                            float(p["target"]),
                            str(p["target_type"]),
                            ENTRY_POLICY_V3_FEATURE_VERSION,
                            f_json,
                            m_json,
                            _stamp(avail_t),
                        ),
                    )
                    touched_cells.add((pop, dreg))
                else:
                    w_fit = 0.0

                cur = conn.execute(
                    """
                    UPDATE policy_episodes
                    SET entry_policy_v3_learned=1,
                        entry_policy_v3_target=?,
                        entry_policy_v3_target_type=?,
                        entry_policy_v3_weight=?,
                        entry_policy_v3_cluster_id=?,
                        entry_policy_v3_population=?,
                        entry_policy_v3_dir_regime=?
                    WHERE id=? AND COALESCE(entry_policy_v3_learned, 0)=0
                    """,
                    (
                        p.get("target"),
                        p.get("target_type"),
                        float(w_fit),
                        cid,
                        pop,
                        dreg,
                        rid,
                    ),
                )
                if cur.rowcount == 1 and p.get("fold") and w_fit > 0.0:
                    applied += 1

            # Persist updated submodels & digests for every touched (population, directional_regime)
            for pop_c, dreg_c in sorted(touched_cells):
                c_rows = conn.execute(
                    """
                    SELECT obs_id, symbol, base_setup, vol_regime, decided_at, label_available_at,
                           bar_bucket, cluster_id, cluster_capped, sample_weight, target,
                           features_json, observed_mask_json
                    FROM entry_policy_v3_observations
                    WHERE engine_id=? AND economic_version=? AND population=? AND directional_regime=?
                      AND label_available_at <= ?
                    ORDER BY label_available_at ASC, id ASC
                    """,
                    (DAY_ENGINE, ENTRY_POLICY_VALUE_V3, pop_c, dreg_c, cutoff),
                ).fetchall()
                sub = fit_v3_submodel(
                    [_row_to_obs_dict(cr) for cr in c_rows],
                    population=pop_c,
                    directional_regime=dreg_c,
                    now_epoch=cutoff,
                )
                _persist_submodel(conn, sub)

            conn.execute("RELEASE mystic_entry_policy_v3")
            conn.commit()
        except Exception:
            conn.execute("ROLLBACK TO mystic_entry_policy_v3")
            conn.execute("RELEASE mystic_entry_policy_v3")
            raise
    finally:
        conn.close()

    if applied:
        invalidate_v3_cache()
        _invalidate_adaptive_caches(db_path)
        logger.info("ENTRY_POLICY_V3_LABELS_APPLIED n=%s version=%s", applied, ENTRY_POLICY_VALUE_V3)
    return applied


def rebuild_entry_policy_value_v3(
    db_path: str,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Deterministically rebuild ``ENTRY_POLICY_VALUE_V3`` from all provable retained episodes.

    Never mutates or deletes ``ENTRY_POLICY_VALUE_V1`` or ``ENTRY_POLICY_VALUE_V2``
    rows, ``policy_episodes`` history, trades, fills, or learning outcomes.
    """
    from backend.services.day_v2.entry_policy_value import ENTRY_POLICY_VALUE_V1, ENTRY_POLICY_VALUE_V2

    cutoff = float(now) if now is not None else float(datetime.now(timezone.utc).timestamp())
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        ensure_v3_schema(conn)
        conn.execute(
            "DELETE FROM entry_policy_v3_observations WHERE engine_id=? AND economic_version=?",
            (DAY_ENGINE, ENTRY_POLICY_VALUE_V3),
        )
        conn.execute(
            "DELETE FROM entry_policy_v3_models WHERE engine_id=? AND economic_version=?",
            (DAY_ENGINE, ENTRY_POLICY_VALUE_V3),
        )
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "policy_episodes" in tables:
            conn.execute(
                """
                UPDATE policy_episodes
                SET entry_policy_v3_learned=0,
                    entry_policy_v3_target=NULL,
                    entry_policy_v3_target_type=NULL,
                    entry_policy_v3_weight=NULL,
                    entry_policy_v3_cluster_id=NULL,
                    entry_policy_v3_population=NULL,
                    entry_policy_v3_dir_regime=NULL
                WHERE engine_id=?
                """,
                (DAY_ENGINE,),
            )
        conn.commit()
    finally:
        conn.close()

    invalidate_v3_cache()
    _invalidate_adaptive_caches(db_path)
    applied = apply_resolved_entry_policy_v3_labels(db_path, now=cutoff)

    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        v1_rows = 0
        v2_rows = 0
        if "adaptive_metric_state" in tables:
            v1_rows = int(
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
        v3_obs = int(
            conn.execute(
                "SELECT COUNT(*) FROM entry_policy_v3_observations WHERE engine_id=? AND economic_version=?",
                (DAY_ENGINE, ENTRY_POLICY_VALUE_V3),
            ).fetchone()[0]
        )
        model_rows = conn.execute(
            """
            SELECT population, directional_regime, obs_count, nominal_weight, effective_sample_size, model_digest
            FROM entry_policy_v3_models
            WHERE engine_id=? AND economic_version=?
            ORDER BY population ASC, directional_regime ASC
            """,
            (DAY_ENGINE, ENTRY_POLICY_VALUE_V3),
        ).fetchall()
        digests = {f"{r['population']}:{r['directional_regime']}": str(r["model_digest"]) for r in model_rows}
        total_nom = sum(float(r["nominal_weight"]) for r in model_rows)
        total_neff = sum(float(r["effective_sample_size"]) for r in model_rows)
    finally:
        conn.close()

    return {
        "applied": applied,
        "version": ENTRY_POLICY_VALUE_V3,
        "feature_version": ENTRY_POLICY_V3_FEATURE_VERSION,
        "v1_rows": v1_rows,
        "v2_rows": v2_rows,
        "v3_observations": v3_obs,
        "v3_models": len(model_rows),
        "nominal_weight": total_nom,
        "effective_sample_size": total_neff,
        "model_digests": digests,
    }


def causal_replay_v3(
    episodes: list[Mapping[str, Any] | dict[str, Any]],
    *,
    ohlcv_index: Mapping[str, tuple[list[float], list[tuple[float, float, float, float, float, float]]]] | None = None,
) -> list[dict[str, Any]]:
    """Strictly causal walk-forward replay of ``ENTRY_POLICY_VALUE_V3``.

    For every decision at ``eval_at`` (default ``decided_at``), fits ONLY labels
    satisfying ``label_available_at <= eval_at``. Zero final-model backcasting.
    """
    from backend.services.day_v2.entry_policy_value import (
        LABEL_REAL_FUNDED_EXECUTION,
        plan_causal_v2_episodes,
    )

    if not episodes:
        return []

    normalized: list[dict[str, Any]] = []
    for idx, ep in enumerate(episodes, start=1):
        copy_ep = dict(ep)
        copy_ep.setdefault("id", idx)
        copy_ep.setdefault("status", "CLOSED" if copy_ep.get("label_available_at") is not None else "OPEN")
        copy_ep.setdefault("capture_version", "EXACT_POLICY_V1")
        copy_ep.setdefault("engine_id", DAY_ENGINE)
        if "kind" not in copy_ep:
            copy_ep["kind"] = "REAL_POLICY_OUTCOME" if int(copy_ep.get("funded") or 0) == 1 else "COUNTERFACTUAL_POLICY_OUTCOME"
        if "train_eligible" not in copy_ep:
            copy_ep["train_eligible"] = 1 if copy_ep.get("eligible", True) else 0
        normalized.append(copy_ep)

    max_ts = max([0.0] + [float(e.get("decided_at") or 0.0) for e in normalized] + [float(e["label_available_at"]) for e in normalized if e.get("label_available_at") is not None])
    plan = plan_causal_v2_episodes(normalized, cutoff=max_ts)

    sym_bar_pop_setups: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for e in normalized:
        rid = int(e["id"])
        p = plan.get(rid)
        if not p or p.get("overlap_group") == "SHADOW_OF_REAL":
            continue
        sym = canon_symbol(str(e.get("symbol") or "")) or _norm_symbol(str(e.get("symbol") or ""))
        stp = str(e.get("setup") or "").strip().upper()
        pop = classify_population(stp)
        bar_b = int(float(e.get("decided_at") or 0.0) // 900.0) * 900
        sym_bar_pop_setups[(sym, bar_b, pop)].add(stp)

    # Extract causal features once per episode
    prepared: dict[int, dict[str, Any]] = {}
    for e in normalized:
        rid = int(e["id"])
        sym = _norm_symbol(str(e.get("symbol") or ""))
        stp_full = str(e.get("setup") or "").strip().upper()
        bstp = base_setup_name(stp_full)
        reg = str(e.get("regime") or "").strip().lower()
        pop = classify_population(stp_full)
        dreg, vreg = extract_directional_regime(reg)
        dec_t = float(e.get("decided_at") or 0.0)
        eval_t = float(e.get("eval_at") if e.get("eval_at") is not None else dec_t)
        bar_b = int(dec_t // 900.0) * 900
        cid = f"{sym}:{bar_b}:{pop}"

        raw_sf: Mapping[str, Any] = {}
        if isinstance(e.get("features"), Mapping):
            raw_sf = e["features"]
        elif e.get("snapshot_json"):
            try:
                sn = json.loads(str(e["snapshot_json"]))
                if isinstance(sn, dict) and isinstance(sn.get("features"), Mapping):
                    raw_sf = sn["features"]
            except json.JSONDecodeError:
                raw_sf = {}
        ep_spread = _finite_float(e.get("entry_spread"))
        if ep_spread is None and e.get("snapshot_json"):
            try:
                sn = json.loads(str(e["snapshot_json"]))
                if isinstance(sn, dict):
                    ep_spread = _finite_float(sn.get("spread"))
            except json.JSONDecodeError:
                pass

        has_explicit_state = bool(raw_sf) or bool(ohlcv_index) or (ep_spread is not None)
        xc, mc, _ = extract_v3_continuous_features(
            raw_sf,
            symbol=sym,
            decided_at=dec_t,
            entry_spread=ep_spread,
            ohlcv_index=ohlcv_index,
            include_session_from_decided_at=has_explicit_state,
        )
        p = plan.get(rid, {})
        is_real = p.get("target_type") == LABEL_REAL_FUNDED_EXECUTION
        n_sib = max(1, len(sym_bar_pop_setups[(sym, bar_b, pop)]))
        w_fit = 1.0 if is_real else (float(p.get("specific_weight") or 0.0) / float(n_sib))

        prepared[rid] = {
            "id": e.get("id"),
            "rid": rid,
            "raw": e,
            "symbol": sym,
            "setup": stp_full,
            "base_setup": bstp,
            "regime": reg,
            "population": pop,
            "directional_regime": dreg,
            "vol_regime": vreg,
            "decided_at": dec_t,
            "eval_at": eval_t,
            "bar_bucket": bar_b,
            "cluster_id": cid,
            "x_cat": build_categorical_vector(sym, bstp, vreg),
            "x_cont": xc,
            "m_cont": mc,
            "plan": p,
            "w_fit": w_fit,
        }

    # Build per-(population, directional_regime) sorted arrays of resolvable training rows
    resolvable_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rid, item in prepared.items():
        p = item["plan"]
        if p.get("fold") and item["w_fit"] > 0.0 and p.get("target") is not None:
            resolvable_by_cell[(item["population"], item["directional_regime"])].append(
                {
                    "obs_id": f"ep:{rid}",
                    "symbol": item["symbol"],
                    "base_setup": item["base_setup"],
                    "vol_regime": item["vol_regime"],
                    "decided_at": item["decided_at"],
                    "label_available_at": float(p["label_available_at"]),
                    "bar_bucket": item["bar_bucket"],
                    "cluster_id": item["cluster_id"],
                    "cluster_capped": True,
                    "sample_weight": item["w_fit"],
                    "target": float(p["target"]),
                    "x_cat": item["x_cat"],
                    "x_cont": item["x_cont"],
                    "m_cont": item["m_cont"],
                }
            )

    cell_state: dict[tuple[str, str], dict[str, Any]] = {}
    for ck, obs_list in resolvable_by_cell.items():
        obs_list.sort(key=lambda o: (o["label_available_at"], o["obs_id"]))
        cell_state[ck] = {
            "obs": obs_list,
            "avail": np.array([o["label_available_at"] for o in obs_list], dtype=float),
            "cursor": -1,
            "last_moment": None,
            "submodel": None,
        }

    decisions = sorted(prepared.values(), key=lambda d: (d["eval_at"], d["rid"]))
    scored: list[dict[str, Any]] = []
    i = 0
    total_d = len(decisions)
    while i < total_d:
        eval_ts = decisions[i]["eval_at"]
        # Collect batch of decisions sharing exact eval_ts so each touched cell fits at most once per timestamp
        j = i + 1
        while j < total_d and decisions[j]["eval_at"] == eval_ts:
            j += 1
        needed_cells = {(decisions[k]["population"], decisions[k]["directional_regime"]) for k in range(i, j)}
        for ck in needed_cells:
            cs = cell_state.get(ck)
            if cs is None:
                continue
            # Strictly causal: label_available_at <= eval_ts
            k_end = int(np.searchsorted(cs["avail"], eval_ts, side="right"))
            # Re-fit when new labels arrived OR >6h elapsed since last decay anchor
            need_refit = cs["submodel"] is None or k_end != cs["cursor"] or cs["last_moment"] is None or (eval_ts - float(cs["last_moment"])) >= 21600.0
            if need_refit:
                cs["cursor"] = k_end
                cs["last_moment"] = eval_ts
                cs["submodel"] = fit_v3_submodel(
                    cs["obs"][:k_end],
                    population=ck[0],
                    directional_regime=ck[1],
                    now_epoch=eval_ts,
                )

        for k in range(i, j):
            d = decisions[k]
            ck = (d["population"], d["directional_regime"])
            cs = cell_state.get(ck)
            sub = cs["submodel"] if (cs and cs["submodel"] is not None) else fit_v3_submodel([], population=ck[0], directional_regime=ck[1], now_epoch=eval_ts)
            view = predict_v3_submodel(
                sub,
                symbol=d["symbol"],
                setup=d["setup"],
                regime=d["regime"],
                x_cont=d["x_cont"],
                m_cont=d["m_cont"],
                eval_timestamp=eval_ts,
            )
            p = d["plan"]
            raw_ep = d["raw"]
            scored.append(
                {
                    "id": d["id"],
                    "symbol": raw_ep.get("symbol"),
                    "setup": raw_ep.get("setup"),
                    "regime": raw_ep.get("regime"),
                    "population": d["population"],
                    "directional_regime": d["directional_regime"],
                    "decided_at": d["decided_at"],
                    "eval_at": eval_ts,
                    "planning_net": raw_ep.get("planning_net"),
                    "target_net": p.get("target"),
                    "target_type": p.get("target_type"),
                    "sample_weight": d["w_fit"] if p.get("fold") else 0.0,
                    "overlap_group": p.get("overlap_group"),
                    "prediction": float(view["mean"]),
                    "expected_policy_net_bps": float(view["expected_policy_net_bps"]),
                    "n": float(view["effective_sample_size"]),
                    "effective_sample_size": float(view["effective_sample_size"]),
                    "nominal_weight": float(view["nominal_weight"]),
                    "sd": float(view["sd"]),
                    "residual_sd": float(view["residual_sd"]),
                    "prediction_se": float(view["prediction_se"]),
                    "parent": float(view["parent"]),
                    "prior": float(view["prior"]),
                    "state_tilt": float(view["state_tilt"]),
                    "feature_version": ENTRY_POLICY_V3_FEATURE_VERSION,
                    "model_digest": str(view["model_digest"]),
                    "version": ENTRY_POLICY_VALUE_V3,
                }
            )
        i = j

    return scored
