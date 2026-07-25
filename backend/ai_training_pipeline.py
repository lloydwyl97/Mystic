#!/usr/bin/env python3
"""
AI Training Data Pipeline
Collects real trade data and market signals for AI learning and model training
"""

import asyncio
import json
import logging
import os
import pickle
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from backend.config.redis_config import get_shared_redis_async, get_shared_redis_sync
from backend.config.trading_universe import TOP10_COINS, TRADING_SYMBOLS
from backend.services.market_data import market_data_service

# Optional import for learning metrics
try:
    from ai_learning_report import AILearningReport
except ImportError:
    AILearningReport = None

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.services.task_manager import task_manager

# Lazy imports for optional AI services (may not be available in all deployments)
try:
    from ai_learning_report import AILearningReport
except ImportError:
    AILearningReport = None  # type: ignore[assignment, misc]

try:
    from backend.modules.ai.ai_model_versioning import get_ai_model_versioning
except ImportError:
    get_ai_model_versioning = None  # type: ignore[assignment, misc]

from backend.config.ai_day_htf_contract import FEATURE_VERSION_DAY_HTF
from backend.config.ai_training_contract import (
    CANONICAL_TELEMETRY_CONTEXT_KEY_DAY_HTF,
    CANONICAL_TELEMETRY_CONTEXT_KEY_V3,
)
from backend.config.trade_worthiness_timing import (
    day_label_grid_seconds,
    label_horizon_bars_for_strategy,
    label_horizon_bars_for_symbol,
    max_hold_seconds_for_symbol,
    primary_label_bar_seconds_for_strategy,
)
from backend.config.training_label_economics import (
    NO_TRACTION_CHECK_FRAC,
    NO_TRACTION_MIN_MFE_PCT,
    SLIPPAGE_PCT,
    TAKER_FEE,
    TRADE_WORTHINESS_LABEL_VERSION,
    default_spread_pct,
    profit_edge_buffer_pct,
    required_edge_pct_for_strategy,
    required_edge_pct_for_training,
    traction_params_for_strategy,
)
from backend.services.ai_canonical_storage import persist_ai_feature_sample_row
from backend.services.canonical_cache import canonical_cache
from backend.services.live_strategy_contracts import per_coin_artifact_file, train_strategy_ids
from backend.services.smart_training_data_manager import smart_training_data_manager
from backend.training.trade_worthiness_label import trade_worthiness_binary_label
from backend.utils.path_helpers import ensure_model_directories

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _format_float(v: Any) -> float:
    try:
        if isinstance(v, (int, float)):
            result = float(v)
        elif isinstance(v, str):
            result = float(v.replace("%", "").replace(",", "").strip())
        else:
            result = 0.0
    except (ValueError, TypeError, AttributeError):
        return 0.0
    else:
        return result


def _decode(b: bytes | None) -> str | None:
    if b is None:
        return None

    try:
        return b.decode()
    except (UnicodeDecodeError, AttributeError, TypeError):
        return None


# RF: per-symbol StandardScaler + one shared RandomForest (see ai_signal_generator inference).
MIN_RF_TRAIN_SAMPLES = int(os.getenv("MIN_RF_TRAIN_SAMPLES", "100") or "100")
MIN_SCALER_ROWS_PER_SYMBOL = 5
# Need enough Tier-A rows for a clean validation split even when Tier B/C
# relieve train-set starvation.
MIN_RF_TIER_A_ROWS = int(os.getenv("MIN_RF_TIER_A_ROWS", "40") or "40")

# CANONICAL feature contract — kept here as integer constants to avoid circular imports.
# Source of truth (with names): backend.services.ai_decision_contract.
_FEATURE_DIM_V1 = 124
_FEATURE_DIM_V2 = 145
_FEATURE_DIM = _FEATURE_DIM_V2  # default training target

_TARGET_FEATURE_VERSION = int(os.getenv("AI_FEATURE_VERSION_TARGET", "3"))
if _TARGET_FEATURE_VERSION not in (1, 2, 3, 4, 5):
    _TARGET_FEATURE_VERSION = 3
_TARGET_FEATURE_DIM = _FEATURE_DIM_V2 if _TARGET_FEATURE_VERSION >= 2 else _FEATURE_DIM_V1


def _canonical_training_pair_symbol(raw: str) -> str:
    """Map cache base (e.g. BTC), pair (BTC/USDT), or BUS (BTCUSDT) to TRADING_SYMBOLS form."""
    s = (raw or "").strip().upper()
    if "/" in s:
        s = s.replace("/", "")
    if s.endswith("USDT"):
        return s
    try:
        idx = TOP10_COINS.index(s)
    except ValueError:
        return f"{s}USDT"
    return TRADING_SYMBOLS[idx]


LABEL_MIN_MOVE_PCT = float(os.getenv("RF_LABEL_MIN_MOVE_PCT", "0.003"))


def _trade_worthiness_label_for_matrix(
    arr: np.ndarray,
    entry_index: int,
    horizon_bars: int,
    *,
    required_edge_pct: float,
    traction_check_frac: float = NO_TRACTION_CHECK_FRAC,
    traction_min_mfe_pct: float = NO_TRACTION_MIN_MFE_PCT,
    ambiguous_max_mfe_pct: float = LABEL_MIN_MOVE_PCT,
    closes: np.ndarray | None = None,
) -> int | None:
    """Binary trade-worthiness label; None skips ambiguous flat rows.

    ``closes`` when set (e.g. day HTF rows) is the **label price series** aligned to sorted rows
    (typically 4h closes). Otherwise column 0 of ``arr`` is used (legacy day path).
    """
    series = closes if closes is not None else arr[:, 0]
    return trade_worthiness_binary_label(
        series,
        entry_index,
        horizon_bars,
        required_edge_pct=required_edge_pct,
        traction_check_frac=traction_check_frac,
        traction_min_mfe_pct=traction_min_mfe_pct,
        ambiguous_max_mfe_pct=ambiguous_max_mfe_pct,
    )


def build_xy_arrays_for_live_strategy(
    rows: list[dict[str, Any]],
    live_strategy_id: str,
    target_dim: int = _TARGET_FEATURE_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Self-supervised X/y for one live strategy (distinct horizon, edge, traction)."""
    strat = (live_strategy_id or "day").strip().lower()
    req_edge = required_edge_pct_for_strategy(strat)
    t_frac, t_mfe = traction_params_for_strategy(strat)
    amb_day = os.getenv("DAY_RF_LABEL_MIN_MOVE_PCT")
    if strat == "day" and amb_day is not None and str(amb_day).strip() != "":
        ambiguous_floor = float(amb_day)
    else:
        from backend.config.training_label_economics import ambiguous_max_mfe_pct_for_training

        ambiguous_floor = ambiguous_max_mfe_pct_for_training()

    by_sym: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not isinstance(r, dict) or "features" not in r:
            continue
        sym_raw = r.get("symbol")
        if not sym_raw:
            continue
        sym_k = _canonical_training_pair_symbol(str(sym_raw))
        by_sym[sym_k].append(r)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    sym_parts: list[np.ndarray] = []
    for sym_k, items in by_sym.items():
        items.sort(key=lambda x: (str(x.get("timestamp") or ""), str(x.get("collection_time") or "")))

        def _row_ok(it: dict[str, Any]) -> bool:
            if not isinstance(it.get("features"), (list, tuple)) or len(it["features"]) != target_dim:
                return False
            fv = int(it.get("feature_version") or 0)
            if strat == "day":
                if fv < FEATURE_VERSION_DAY_HTF:
                    return False
                try:
                    if float(it.get("label_anchor_close") or 0) <= 0:
                        return False
                except (TypeError, ValueError):
                    return False
            elif fv < _TARGET_FEATURE_VERSION:
                return False
            if target_dim == _FEATURE_DIM_V2:
                ls = (it.get("live_strategy_id") or "").strip().lower()
                if ls != strat:
                    return False
            return True

        items_ok = [it for it in items if _row_ok(it)]
        if not items_ok:
            continue
        feats = [it["features"] for it in items_ok]
        xs = np.asarray(feats, dtype=np.float64)
        n = xs.shape[0]
        horizon = label_horizon_bars_for_strategy(strat, sym_k)
        ahead = min(horizon, n - 1)
        if ahead < 2:
            continue
        closes_for_lab: np.ndarray | None
        if strat == "day":
            closes_for_lab = np.asarray(
                [float(it.get("label_anchor_close") or 0.0) for it in items_ok],
                dtype=np.float64,
            )
            if closes_for_lab.shape[0] != n or not bool(np.all(closes_for_lab > 0)):
                continue
        else:
            closes_for_lab = None
        keep_x = []
        keep_y = []
        for i in range(n - ahead):
            lab = _trade_worthiness_label_for_matrix(
                xs,
                i,
                ahead,
                required_edge_pct=req_edge,
                traction_check_frac=t_frac,
                traction_min_mfe_pct=t_mfe,
                ambiguous_max_mfe_pct=ambiguous_floor,
                closes=closes_for_lab,
            )
            if lab is None:
                continue
            keep_x.append(_zero_learning_blocked_feature_dims([float(v) for v in xs[i]]))
            keep_y.append(lab)
        if len(keep_x) < 10:
            continue
        x_parts.append(np.array(keep_x))
        y_parts.append(np.array(keep_y, dtype=np.int64))
        sym_parts.append(np.array([sym_k] * len(keep_x), dtype=object))
    if not x_parts:
        return np.array([]), np.array([]), np.array([])
    return np.vstack(x_parts), np.concatenate(y_parts), np.concatenate(sym_parts)


def build_xy_arrays_per_symbol(
    rows: list[dict[str, Any]],
    target_dim: int = _TARGET_FEATURE_DIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alias for scripts/tests: **day** contract (matches legacy horizon defaults)."""
    return build_xy_arrays_for_live_strategy(rows, "day", target_dim=target_dim)


def _infer_outcome_feature_version(row: dict[str, Any]) -> int:
    """Resolve feature version from context_json or feature vector length."""
    ctx_fv = 0
    dim_fv = 0
    try:
        cj = row.get("context_json")
        if cj:
            ctx0 = json.loads(cj) if isinstance(cj, str) else {}
            if isinstance(ctx0, dict):
                ctx_fv = int(ctx0.get("_feature_version") or ctx0.get("feature_version") or 0)
    except (json.JSONDecodeError, TypeError, ValueError):
        ctx_fv = 0
    try:
        fj = row.get("features_json")
        if fj:
            feats = json.loads(fj) if isinstance(fj, str) else fj
            if isinstance(feats, list):
                dim = len(feats)
                if dim >= _FEATURE_DIM_V2:
                    dim_fv = int(FEATURE_VERSION_DAY_HTF)
                elif dim >= _FEATURE_DIM_V1:
                    dim_fv = 1
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return max(ctx_fv, dim_fv)


def _live_strategy_from_outcome_row(strategy_id_val: str | None, context_json_val: str | None) -> str | None:
    """Parse live strategy from explicit column first, then context_json."""
    if strategy_id_val and str(strategy_id_val).strip().lower() in ("day", "day"):
        return str(strategy_id_val).strip().lower()
    if not context_json_val:
        return None
    try:
        d = json.loads(context_json_val)
        if isinstance(d, dict):
            v = d.get("_live_ai_strategy") or d.get("live_ai_strategy")
            if v is not None and str(v).strip():
                s = str(v).strip().lower()
                if s in ("day", "day"):
                    return s
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def _zero_learning_blocked_feature_dims(feats: list[float]) -> list[float]:
    """Zero proxy/unsupported dims so RF cannot learn from dishonest tape features."""
    try:
        from backend.services.day_feature_health import zero_learning_blocked_feature_dims

        return zero_learning_blocked_feature_dims(feats)
    except Exception:
        return list(feats)


# Root-cause -> training weight multiplier. Distinguishes a genuinely
# learnable entry-quality lesson (bad setup, weak volume/volatility confirmation)
# from noise the model should not blame on entry features (regime shift, a
# market-wide reversal, or an entry that was already flagged unhealthy at the
# time — training on that just teaches garbage-in). See day_outcome_attribution
# .classify_outcome_reason for the source classification.
_ATTRIBUTION_WEIGHT: dict[str, float] = {
    "BAD_SETUP": 2.0,
    "GOOD_SETUP_BAD_ENTRY": 1.8,
    "VOLATILITY_EXPANSION_AGAINST_TRADE": 1.5,
    "VOLUME_CONFIRMATION_FAILED": 1.5,
    "SETUP_HISTORY_WEAK": 1.3,
    "EXIT_TOO_LATE": 1.1,
    "EXECUTION_COST_TOO_HIGH": 0.7,
    "REGIME_SHIFT_AGAINST_TRADE": 0.6,
    "MARKET_WIDE_REVERSAL": 0.6,
    "FEATURE_HEALTH_WEAK": 0.4,
}


def _outcome_attribution_multiplier(row: dict[str, Any]) -> float | None:
    """Look up the root-cause weight for a row's outcome_attribution_reason, if present."""
    raw = row.get("score_components_json")
    if not raw:
        return None
    try:
        comps = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None
    if not isinstance(comps, dict):
        return None
    reason = str(comps.get("outcome_attribution_reason") or "").strip().upper()
    return _ATTRIBUTION_WEIGHT.get(reason)


def _outcome_exit_class_multiplier(row: dict[str, Any], y_label: int) -> float:
    """Weight NET_PROFIT wins and STALL/GIVEBACK losses higher than flat noise exits.

    Prefers the root-cause attribution weight when available (finer-grained than
    the close_reason string alone); falls back to the original exit-reason rule.
    """
    attribution_mult = _outcome_attribution_multiplier(row)
    if attribution_mult is not None:
        return attribution_mult
    reason = str(row.get("close_reason") or row.get("outcome_class") or "").upper()
    if y_label == 1 and ("NET_PROFIT" in reason or reason.startswith("TP")):
        return 2.0
    if y_label == 0 and ("STALL" in reason or "GIVEBACK" in reason):
        return 1.8
    if y_label == 0 and "TIME_STOP" in reason:
        return 1.35
    if y_label == 1:
        return 1.25
    return 1.0


def _outcome_rows_to_xy_for_strategy(
    outcome_rows: list[dict[str, Any]],
    live_strategy_id: str,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Realized outcomes filtered to one live strategy (distinct supervision).

    Logs a single ``OUTCOME_XY_FILTER`` diagnostic line per call so operators
    can verify that DAY v4 training is not silently fed v3 (day) rows or
    missing feature vectors. The counters are intentionally cheap.

    Returns X, y, symbols, exit_class_multipliers (aligned with y).
    """
    want = (live_strategy_id or "day").strip().lower()
    empty = (np.array([]), np.array([]), np.array([]), np.array([]))
    if not outcome_rows:
        logger.info(
            "OUTCOME_XY_FILTER strategy=%s total=0 eligible=0 reason=no_outcome_rows",
            want,
        )
        return empty
    Xs: list[list[float]] = []
    ys: list[int] = []
    syms: list[str] = []
    w_mults: list[float] = []
    skipped_strategy_mismatch = 0
    skipped_feature_version = 0
    skipped_missing_features = 0
    skipped_feature_dim = 0
    skipped_missing_symbol = 0
    skipped_parse_error = 0
    for r in outcome_rows:
        try:
            tag = _live_strategy_from_outcome_row(r.get("strategy_id"), r.get("context_json"))
            if want == "day":
                if tag is not None and tag != "day":
                    skipped_strategy_mismatch += 1
                    continue
            elif tag != want:
                skipped_strategy_mismatch += 1
                continue
            if want == "day":
                fv_row = _infer_outcome_feature_version(r)
                if fv_row < FEATURE_VERSION_DAY_HTF:
                    skipped_feature_version += 1
                    continue
            fj = r.get("features_json")
            if not fj:
                skipped_missing_features += 1
                continue
            feats = json.loads(fj)
            if not isinstance(feats, list) or len(feats) != target_dim:
                skipped_feature_dim += 1
                continue
            sym = _canonical_training_pair_symbol(str(r.get("symbol") or ""))
            if not sym:
                skipped_missing_symbol += 1
                continue
            Xs.append(_zero_learning_blocked_feature_dims([float(x) for x in feats]))
            y_label = int(r.get("outcome_label") or 0)
            mem_class = str(r.get("good_bad_memory_class") or "").strip().upper()
            if mem_class == "BAD":
                y_label = 0
            net_ev_entry = r.get("selected_net_expected_value")
            rank_snapshot_id = r.get("rank_snapshot_id")
            try:
                if y_label > 0 and rank_snapshot_id not in (None, "", 0) and float(net_ev_entry or 0.0) < 0.0:
                    y_label = 0
            except (TypeError, ValueError):
                pass
            ys.append(y_label)
            syms.append(sym)
            w_mults.append(_outcome_exit_class_multiplier(r, y_label))
        except (ValueError, TypeError, json.JSONDecodeError):
            skipped_parse_error += 1
            continue
    logger.info(
        "OUTCOME_XY_FILTER strategy=%s total=%d eligible=%d "
        "skipped_strategy_mismatch=%d skipped_feature_version=%d "
        "skipped_missing_features=%d skipped_feature_dim=%d "
        "skipped_missing_symbol=%d skipped_parse_error=%d target_dim=%d min_fv=%s",
        want,
        len(outcome_rows),
        len(Xs),
        skipped_strategy_mismatch,
        skipped_feature_version,
        skipped_missing_features,
        skipped_feature_dim,
        skipped_missing_symbol,
        skipped_parse_error,
        target_dim,
        FEATURE_VERSION_DAY_HTF if want == "day" else "n/a",
    )
    if not Xs:
        return empty
    return (
        np.asarray(Xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(syms, dtype=object),
        np.asarray(w_mults, dtype=np.float64),
    )


def transform_rows_per_symbol_scalers(
    X: np.ndarray,
    sym_row: np.ndarray,
    scalers_by_symbol: dict[str, StandardScaler],
    global_scaler: StandardScaler,
) -> np.ndarray:
    """[MYSTIC_CORE_TAG: DEAD_CODE] Unused; RF training fits one StandardScaler per symbol in _train_lightweight_models."""
    out = np.empty_like(X, dtype=np.float64)
    for i in range(X.shape[0]):
        s = str(sym_row[i])
        sc = scalers_by_symbol.get(s) or global_scaler
        out[i] = sc.transform(X[i : i + 1])[0]
    return out


def fit_per_symbol_scalers_for_rf(
    X_train: np.ndarray,
    sym_train: np.ndarray,
    *,
    min_rows: int = MIN_SCALER_ROWS_PER_SYMBOL,
) -> tuple[dict[str, StandardScaler], StandardScaler]:
    """[MYSTIC_CORE_TAG: DEAD_CODE] Unused abandoned multi-scaler pooling helper."""
    global_scaler = StandardScaler()
    global_scaler.fit(X_train)

    per_sym: dict[str, StandardScaler] = {}
    for s in np.unique(sym_train):
        sk = str(s)
        mask = sym_train == s
        sub = X_train[mask]
        if sub.shape[0] >= min_rows:
            sc = StandardScaler()
            sc.fit(sub)
            per_sym[sk] = sc
        else:
            per_sym[sk] = global_scaler

    scalers_out: dict[str, StandardScaler] = {}
    for sym in TRADING_SYMBOLS:
        scalers_out[sym] = per_sym.get(sym, global_scaler)
    return scalers_out, global_scaler


class AITrainingDataPipeline:
    """Collects and processes training data for AI models"""

    def __init__(self, cache: Any = None) -> None:
        self.cache = cache or canonical_cache
        self.is_running = False
        self._current_accuracy = 0.0  # Store current accuracy from enhanced models
        self._start_time = time.time()
        # Track background tasks for proper cleanup
        self._tasks: list[asyncio.Task[Any]] = []

        # Training data storage - use path helpers for consistency
        directories = ensure_model_directories()
        self.training_data_dir = directories["training_data"]
        self.model_versions_dir = directories["versions"]

        # Ensure directories exist
        Path(self.training_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_versions_dir).mkdir(parents=True, exist_ok=True)

        # Data collection settings - BALANCED for sustainable 24/7 operation
        # Indicators don't change meaningfully faster than 10 seconds
        self.collection_interval = int(os.getenv("AI_COLLECTION_INTERVAL", "60"))
        self.collection_interval_idle = int(os.getenv("AI_COLLECTION_INTERVAL_IDLE", "120"))
        self.feature_window = int(os.getenv("AI_FEATURE_WINDOW", "10000"))  # 10K features per symbol
        self.prediction_horizon = 48  # Hours to predict ahead
        self.continuous_learning_enabled = True
        self.learning_frequency = int(os.getenv("AI_LEARNING_FREQUENCY", "900"))  # Train every 15 minutes (900s)
        self.last_retrain_time = 0
        self.feature_count = _TARGET_FEATURE_DIM  # 145 on v2, 124 on v1; canonical contract
        self.feature_importance_threshold = 0.01  # Minimum importance threshold for features

        # Training data cache - BALANCED for memory efficiency
        self.training_cache: dict[str, list[dict[str, Any]]] = {}
        self.max_training_cache_size = int(os.getenv("AI_MAX_CACHE_SIZE", "100000"))  # 100K per symbol
        self.last_collection: dict[str, str] = {}
        # v3: one persisted row per (symbol_bus, strategy) primary OHLCV bucket
        self._last_primary_emit_bucket: dict[str, int] = {}

        # DAY v5: dedupe historical 4h anchor timestamps already written to training_cache
        self._day_emitted_anchor_ts: dict[str, set[int]] = defaultdict(set)
        self._last_collect_new_anchors: int = 0

        # Load existing training data from disk on startup
        self._load_existing_training_data()
        self._replay_day_training_anchor_index()

        # Model performance tracking - BALANCED for efficiency
        self.model_performance: dict[str, dict[str, Any]] = {}
        self.training_history: list[dict[str, Any]] = []
        self.max_training_history = int(os.getenv("AI_MAX_TRAINING_HISTORY", "5000"))  # 5K sessions
        self.performance_metrics = {
            "total_training_sessions": 0,
            "successful_trainings": 0,
            "failed_trainings": 0,
            "average_accuracy": 0.0,
            "best_accuracy": 0.0,
            "last_training_time": None,
            "models_trained": 0,
        }

        # Model versioning - BALANCED to keep best models, prune old ones
        self.model_versions: dict[str, list[dict[str, Any]]] = {}
        self.max_model_versions = int(os.getenv("AI_MAX_MODEL_VERSIONS", "50"))  # Keep 50 best models
        self.current_versions: dict[str, str] = {}
        self._last_prune_time = 0  # Track when we last pruned models
        self._last_model_rollback_check = 0.0
        self.model_rollback_check_interval = int(os.getenv("AI_MODEL_ROLLBACK_CHECK_SEC", "300"))

    def _load_existing_training_data(self) -> None:
        """Load existing training data from disk on startup"""
        try:
            training_data_path = Path(self.training_data_dir)
            if not training_data_path.exists():
                logger.info("No existing training data directory found")
                return

            # Load data from all *_latest.json files (legacy ``COIN_latest`` or ``BUS_day_latest`` / ``BUS_day_latest``)
            loaded_samples = 0
            for file_path in training_data_path.glob("*_latest.json"):
                try:
                    with file_path.open() as f:
                        symbol_data = json.load(f)

                    stem = file_path.stem
                    if not stem.endswith("_latest"):
                        continue
                    key = stem[: -len("_latest")]  # e.g. BTCUSDT_day or BTC (legacy)
                    if key.endswith("_day"):
                        training_pair = _canonical_training_pair_symbol(key[: -len("_day")])
                        strat_hint = "day"
                    else:
                        training_pair = _canonical_training_pair_symbol(key.replace("/", ""))
                        strat_hint = ""

                    if symbol_data and isinstance(symbol_data, list):
                        for row in symbol_data:
                            if not isinstance(row, dict):
                                continue
                            if not row.get("symbol"):
                                row["symbol"] = training_pair
                            if strat_hint and not row.get("live_strategy_id"):
                                row["live_strategy_id"] = strat_hint
                        if strat_hint == "day":
                            symbol_data = [
                                row
                                for row in symbol_data
                                if isinstance(row, dict)
                                and int(row.get("feature_version") or 0) >= FEATURE_VERSION_DAY_HTF
                                and str(row.get("live_strategy_id") or "").strip().lower() == "day"
                                and isinstance(row.get("features"), (list, tuple))
                                and len(row["features"]) == _FEATURE_DIM_V2
                                and float(row.get("label_anchor_close") or 0) > 0
                            ]
                        cache_key = f"{training_pair}_{strat_hint}" if strat_hint else training_pair
                        self.training_cache[cache_key] = symbol_data[-self.max_training_cache_size :]
                        loaded_samples += len(self.training_cache[cache_key])

                        if self.training_cache[cache_key]:
                            latest_entry = max(self.training_cache[cache_key], key=lambda x: x.get("timestamp", ""))
                            self.last_collection[cache_key] = latest_entry.get("collection_time", _now_iso())

                        logger.info("Loaded %d training samples for %s", len(self.training_cache[cache_key]), cache_key)

                except (json.JSONDecodeError, OSError, ValueError) as e:
                    logger.warning(f"Failed to load training data from {file_path}: {e}")
                    continue

            if loaded_samples > 0:
                logger.info(f"Successfully loaded {loaded_samples} total training samples from disk")
            else:
                logger.info("No training data found on disk")

        except Exception as e:
            logger.exception(f"Error loading existing training data: {e}")

    def _replay_day_training_anchor_index(self) -> None:
        """Populate _day_emitted_anchor_ts from persisted DAY v5 rows (stable across process restarts)."""
        for ck, rows in self.training_cache.items():
            if not ck.endswith("_day"):
                continue
            seen = self._day_emitted_anchor_ts[ck]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ao = row.get("label_anchor_4h_open_ms")
                if ao is None:
                    continue
                try:
                    seen.add(int(ao))
                except (TypeError, ValueError):
                    continue

    def _normalize_symbol(self, symbol: str | None) -> str | None:
        """Normalize symbol to order book key format (e.g., BTC from BTCUSDT)."""
        if not symbol:
            return None
        normalized = str(symbol).upper().replace("/", "").replace("-", "").strip()
        if normalized.endswith("USDT"):
            return normalized[:-4] or None
        if normalized.endswith("USD"):
            return normalized[:-3] or None
        return normalized or None

    async def _fetch_order_book_features(self, symbol: str | None) -> dict[str, float] | None:
        """Fetch order book microstructure: Redis ``orderbook:{BASE}`` first, then live public L2."""
        normalized = self._normalize_symbol(symbol)
        if not normalized:
            return None
        parsed: dict[str, float] | None = None
        try:
            redis_client = get_shared_redis_async()
            if redis_client is not None:
                raw = await redis_client.hgetall(f"orderbook:{normalized}")
                if raw:
                    parsed = {}
                    for field, value in raw.items():
                        try:
                            fk = field.decode() if isinstance(field, bytes) else field
                            fv = value.decode() if isinstance(value, bytes) else value
                            parsed[str(fk)] = float(fv)
                        except (TypeError, ValueError):
                            continue
        except Exception as error:
            logger.debug("Redis order book lookup failed for %s: %s", normalized, error)

        if parsed:
            return parsed

        try:
            from backend.services.order_book_service import fetch_order_book_features_live

            ccxt_sym = f"{normalized}/USDT"
            live = await fetch_order_book_features_live(ccxt_sym)
            if live:
                logger.debug("Order book live fallback used for %s", normalized)
            return live
        except Exception as ex:
            logger.debug("Live order book fallback failed for %s: %s", normalized, ex)
            return None

    async def start(self) -> None:
        """Start the training data collection process"""
        logger.info("Starting AI Training Data Pipeline with Continuous Learning")
        self.is_running = True
        self._start_time = time.time()
        try:
            # Seed pattern memory from recent closed outcomes so sizing can learn
            # from coin history even after a cold start / empty pattern tables.
            try:
                from backend.services.ai_pattern_memory import backfill_pattern_memory_from_outcomes

                bf = await asyncio.to_thread(backfill_pattern_memory_from_outcomes)
                logger.info("PATTERN_MEMORY_BACKFILL_ON_START %s", bf)
            except Exception as bf_err:
                logger.warning("PATTERN_MEMORY_BACKFILL_ON_START failed: %s", bf_err)

            # Start background tasks with proper exception handling
            task1 = await task_manager.create_task(self._collection_loop(), name="ai_training_pipeline:collection_loop")
            self._tasks.append(task1)
            task2 = await task_manager.create_task(self._continuous_learning_loop(), name="ai_training_pipeline:continuous_learning_loop")
            self._tasks.append(task2)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            logger.exception(f"Error starting training pipeline: {e}")
            self.is_running = False
            raise

    async def _collection_loop(self) -> None:
        """Background task to collect training data at regular intervals"""
        from backend.services.day_active_market_bundle import apply_day_bundle_stagger

        await apply_day_bundle_stagger("learning")
        attempt = 0
        max_attempts = 5
        backoff_seconds = 60

        while self.is_running:
            try:
                await self.collect_training_data()

                # LEARNING INGESTION (starvation fix): forward-label candidate
                # snapshots (rejected/no-trade/buy) so the learning loop gets
                # labels from decisions that never became closed trades.
                try:
                    from backend.services.ai_learning_ingestion import LABEL_BATCH_LIMIT, label_pending_snapshots

                    # Drain pending backlog in short bursts (was capping at one 400-row batch).
                    total_lbl = {"scanned": 0, "labeled": 0, "partial": 0, "unlabelable": 0}
                    for _pass in range(4):
                        label_counters = await asyncio.to_thread(label_pending_snapshots)
                        for k in total_lbl:
                            total_lbl[k] += int(label_counters.get(k, 0) or 0)
                        if int(label_counters.get("scanned", 0) or 0) < int(LABEL_BATCH_LIMIT):
                            break
                    if total_lbl.get("labeled") or total_lbl.get("partial") or total_lbl.get("scanned"):
                        logger.info(
                            "SNAPSHOT_LABELER: scanned=%d labeled=%d partial=%d unlabelable=%d",
                            total_lbl.get("scanned", 0),
                            total_lbl.get("labeled", 0),
                            total_lbl.get("partial", 0),
                            total_lbl.get("unlabelable", 0),
                        )
                except Exception as lbl_e:
                    logger.debug("snapshot labeler skipped: %s", lbl_e)

                # [MYSTIC_CORE_TAG: TELEMETRY_ONLY] Empty-features call only refreshes
                # ai_learning_stats publish; canonical retrain runs in
                # _continuous_learning_loop with the populated training_batch.
                await self._train_lightweight_models(features=[])
                attempt = 0  # Reset attempt counter on success

                sleep_sec = self.collection_interval_idle if self._last_collect_new_anchors <= 0 else self.collection_interval
                await asyncio.sleep(sleep_sec)

            except (ValueError, TypeError, AttributeError, RuntimeError, ConnectionError) as e:
                attempt += 1
                logger.exception(f"Error in training data collection (attempt {attempt}/{max_attempts}): {e}")

                if attempt >= max_attempts:
                    logger.exception(f"Failed to collect training data after {max_attempts} attempts, pausing collection")
                    attempt = 0
                    await asyncio.sleep(300)  # Longer pause after max attempts
                else:
                    # Exponential backoff
                    retry_time = backoff_seconds * attempt
                    logger.info(f"Retrying in {retry_time} seconds...")
                    await asyncio.sleep(retry_time)

    async def _continuous_learning_loop(self) -> None:
        """Run advanced continuous model retraining with all 124 features"""
        while self.is_running:
            try:
                # Collection loop owns all Binance bundle fetches; avoid duplicate collects here.
                await self._maybe_run_model_rollback_checks()

                current_time = time.time()
                time_since_last_train = current_time - self.last_retrain_time

                # Check if it's time to retrain
                if self.continuous_learning_enabled and time_since_last_train >= self.learning_frequency:
                    logger.info(
                        "[%s] Starting canonical retrain (target feature_version=%d, target_dim=%d)",
                        EXCHANGE_ID,
                        _TARGET_FEATURE_VERSION,
                        _TARGET_FEATURE_DIM,
                    )
                    try:
                        # Outcome rows are merged INSIDE _train_lightweight_models so
                        # the join to features_json + label happens at the same dim
                        # as the training matrix.

                        # Prepare batch training data from DAY cache keys only (`{BUS}_day`).
                        # Legacy `{BUS}` buckets can hold tens of thousands of fv2 samples; blindly
                        # prepending them exhausts RF batch caps before any *_day v5 rows appear.
                        all_features = []
                        day_cache_keys = sorted(k for k in self.training_cache if k.endswith("_day"))
                        total_samples = sum(len(self.training_cache.get(k, [])) for k in day_cache_keys)
                        logger.info("Total DAY-cache training samples available: %d", total_samples)

                        for _symbol in day_cache_keys:
                            data_points = self.training_cache.get(_symbol) or []
                            if len(data_points) <= 0:
                                continue
                            recent_points = data_points[-1000:] if len(data_points) > 1000 else data_points
                            all_features.extend(recent_points)

                        # Do not shuffle: labels are sequential next-step within symbol; shuffling would leak future across symbols/times.

                        # Limit to a reasonable batch size for efficient training
                        max_batch_size = 10000
                        training_batch = all_features[:max_batch_size] if len(all_features) > max_batch_size else all_features

                        if len(training_batch) >= 5:  # Lower threshold for testing
                            # Train models with the collected features
                            await self._train_lightweight_models(training_batch)

                            # Perform advanced feature analysis
                            await self._analyze_feature_importance(training_batch)

                            # Update learning progress metrics
                            self._update_learning_progress(len(training_batch))

                            # BALANCED: Keep training history within limits for memory efficiency
                            if len(self.training_history) > self.max_training_history:
                                self.training_history = self.training_history[-self.max_training_history :]
                                logger.debug(f"Trimmed training history to {self.max_training_history} entries")

                            # SMART CLEANUP: Intelligent training data management
                            # Runs once per day, keeps best learning examples, removes garbage
                            if smart_training_data_manager.should_run_cleanup():
                                logger.info("Running smart training data cleanup...")
                                cleanup_result = await smart_training_data_manager.smart_cleanup()
                                logger.info(f"Cleanup result: {cleanup_result}")

                            # MODEL PRUNING: Keep only best N models to prevent disk bloat
                            await self._prune_old_models()

                            # Stale-refresh pass: re-evaluate latest candidates for coins
                            # whose active artifacts aged out (tie-or-better promote path).
                            try:
                                from backend.services.ai_stale_model_workflow import run_stale_model_workflow

                                stale_summary = await asyncio.to_thread(run_stale_model_workflow)
                                logger.info(
                                    "STALE_MODEL_WORKFLOW: evaluated=%s promoted=%s",
                                    stale_summary.get("symbols_evaluated"),
                                    [r.get("symbol") for r in (stale_summary.get("results") or []) if r.get("promoted")],
                                )
                            except Exception as stale_err:
                                logger.warning("STALE_MODEL_WORKFLOW failed: %s", stale_err)

                            # DISK USAGE MONITORING: Emergency cleanup at 90%
                            disk_stats = smart_training_data_manager.get_disk_usage_stats()
                            if disk_stats.get("usage_percent", 0) > 90:
                                logger.warning(f"Training data disk usage HIGH: {disk_stats}")
                                loop = asyncio.get_running_loop()
                                await loop.run_in_executor(None, smart_training_data_manager.emergency_cleanup)

                            # Generate periodic learning reports
                            await self._generate_learning_report()
                        else:
                            logger.warning(f"Insufficient training data: {len(training_batch)} samples (need at least 100)")

                        self.last_retrain_time = time.time()
                    except (ValueError, TypeError, AttributeError, RuntimeError, ConnectionError) as e:
                        logger.exception(f"Error in continuous learning process: {e}")
                        logger.exception(f"[{EXCHANGE_ID}] Continuous learning error: {e}")

                # Wait before checking again - use configured learning frequency
                # Sleep for 60 seconds between checks (actual training controlled by learning_frequency)
                await asyncio.sleep(60)

            except (ValueError, TypeError, AttributeError, RuntimeError, ConnectionError) as e:
                logger.exception(f"Error in continuous learning loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying after error

    async def _maybe_run_model_rollback_checks(self) -> None:
        now = time.time()
        if (now - self._last_model_rollback_check) < max(60, self.model_rollback_check_interval):
            return
        self._last_model_rollback_check = now
        try:
            from backend.services.ai_model_promotion import maybe_rollback_underperforming_model

            rollbacks = 0
            for strat in train_strategy_ids():
                for sym in TRADING_SYMBOLS:
                    ok, reason = maybe_rollback_underperforming_model(
                        strategy_id=strat,
                        symbol=str(sym),
                        min_samples=int(os.getenv("AI_MODEL_ROLLBACK_MIN_SAMPLES", "20")),
                    )
                    if ok:
                        rollbacks += 1
                        logger.warning("MODEL_ROLLBACK_EXECUTED: strategy=%s symbol=%s reason=%s", strat, sym, reason)
            if rollbacks:
                logger.warning("MODEL_ROLLBACK_SUMMARY: %d model rollbacks executed", rollbacks)
            else:
                logger.info("MODEL_ROLLBACK_CHECK: no rollback required")
        except Exception as rollback_err:
            logger.debug("MODEL_ROLLBACK_CHECK skipped: %s", rollback_err)

    async def _analyze_feature_importance(self, features: list[dict[str, Any]]) -> None:
        """[MYSTIC_CORE_TAG: TELEMETRY_ONLY] Feature names may not match FEATURE_MAPPING; same lookahead contract as RF when per-symbol path used."""
        try:
            if len(features) < 100:
                return

            by_symbol: dict[str, list[list[float]]] = defaultdict(list)
            for feature_dict in features:
                if not isinstance(feature_dict, dict) or "features" not in feature_dict:
                    continue
                sym = feature_dict.get("symbol")
                if not sym:
                    continue
                row = feature_dict["features"]
                if isinstance(row, list) and len(row) >= 2:
                    by_symbol[str(sym)].append([float(x) for x in row])

            xs: list[list[float]] = []
            ys: list[int] = []
            req_edge = required_edge_pct_for_training()
            for sym_key, rows in by_symbol.items():
                sym_norm = _canonical_training_pair_symbol(sym_key)
                la = label_horizon_bars_for_symbol(sym_norm)
                if len(rows) < la + 1:
                    continue
                arr = np.asarray(rows, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < _FEATURE_DIM:
                    continue
                for i in range(len(arr) - la):
                    lab = _trade_worthiness_label_for_matrix(arr, i, la, required_edge_pct=req_edge)
                    if lab is None:
                        continue
                    xs.append(arr[i].tolist())
                    ys.append(lab)

            if len(xs) < 100:
                flat: list[list[float]] = []
                for feature_dict in features:
                    if isinstance(feature_dict, dict) and "features" in feature_dict:
                        row = feature_dict["features"]
                        if isinstance(row, list) and row:
                            flat.append([float(x) for x in row])
                flat_la = label_horizon_bars_for_symbol(TRADING_SYMBOLS[0])
                if len(flat) < flat_la + 10:
                    return
                arr = np.asarray(flat, dtype=float)
                if arr.ndim != 2 or arr.shape[1] < _FEATURE_DIM:
                    return
                xs = []
                ys = []
                for i in range(len(arr) - flat_la):
                    lab = _trade_worthiness_label_for_matrix(arr, i, flat_la, required_edge_pct=req_edge)
                    if lab is None:
                        continue
                    xs.append(arr[i].tolist())
                    ys.append(lab)
                if len(xs) < 100:
                    return

            X = np.asarray(xs, dtype=float)
            y = np.asarray(ys, dtype=int)

            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            model.fit(X, y)

            # Extract feature importance
            importances = model.feature_importances_

            # Store feature importance using shared Redis pool (prevents new connections)
            redis_client = get_shared_redis_async()
            if redis_client is None:
                logger.warning("Shared Redis client unavailable; skipping feature importance storage")
                return

            # Create mapping of feature index to name for better interpretability
            feature_names = [
                # Basic price features (10)
                "price",
                "high",
                "low",
                "open",
                "volume",
                "change_24h",
                "change_7d",
                "change_30d",
                "price_range",
                "typical_price",
                # Technical indicators (24)
                "ma_5",
                "ma_10",
                "ma_20",
                "ma_50",
                "ma_100",
                "ma_200",
                "ema_12",
                "ema_26",
                "ema_50",
                "rsi",
                "rsi_14",
                "stoch_k",
                "stoch_d",
                "williams_r",
                "cci",
                "macd",
                "macd_signal",
                "macd_histogram",
                "bb_upper",
                "bb_middle",
                "bb_lower",
                "bb_position",
                "bb_width",
                "obv",
                "ad_line",
                "cmf",
                "mfi",
                # Volatility indicators (10)
                "volatility",
                "atr",
                "natr",
                "keltner_upper",
                "keltner_lower",
                "donchian_upper",
                "donchian_lower",
                "parabolic_sar",
                "volatility_ratio",
                "price_volatility",
                # Momentum indicators (15)
                "roc",
                "momentum",
                "ppo",
                "trix",
                "ultimate_oscillator",
                "awesome_oscillator",
                "balance_of_power",
                "ease_of_movement",
                "mass_index",
                "vortex_vi_plus",
                "vortex_vi_minus",
                "kst",
                "tsi",
                "aroon_up",
                "aroon_down",
                # Trend indicators (10)
                "adx",
                "di_plus",
                "di_minus",
                "aroon_oscillator",
                "ichimoku_tenkan",
                "ichimoku_kijun",
                "ichimoku_senkou_a",
                "ichimoku_senkou_b",
                "psar",
                "trend_strength",
                # Volume profile (8)
                "volume_ma_5",
                "volume_ma_10",
                "volume_ma_20",
                "volume_ratio",
                "volume_price_trend",
                "negative_volume_index",
                "positive_volume_index",
                "volume_weighted_price",
                # Market sentiment (10)
                "fear_greed_index",
                "social_sentiment",
                "news_sentiment",
                "put_call_ratio",
                "vix",
                "market_cap",
                "supply",
                "circulating_supply",
                "max_supply",
                "market_dominance",
                # Time-based features (10)
                "hour",
                "day_of_week",
                "day_of_month",
                "month",
                "iso_weekday",
                "day_of_year",
                "hour_12h",
                "minute",
                "second",
                "seconds_since_midnight",
                # Advanced technical analysis (8)
                "fib_23.6",
                "fib_38.2",
                "fib_61.8",
                "pivot_point",
                "resistance_1",
                "resistance_2",
                "support_1",
                "support_2",
                # Advanced volume analysis (8)
                "volume_profile_poc",
                "volume_profile_vah",
                "volume_profile_val",
                "vwap",
                "twap",
                "volume_imbalance",
                "volume_delta",
                "order_flow",
                # Advanced market microstructure (8)
                "bid_ask_spread",
                "order_book_imbalance",
                "market_depth",
                "liquidity_score",
                "price_impact",
                "market_efficiency",
                "volatility_smile",
                "price_skewness",
            ]

            # Ensure we have the right number of feature names
            if len(feature_names) != len(importances):
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            # Create sorted importance data
            importance_data = []
            for i, imp in enumerate(importances):
                if i < len(feature_names):
                    importance_data.append({"name": feature_names[i], "importance": float(imp)})
                else:
                    importance_data.append({"name": f"feature_{i}", "importance": float(imp)})

            # Sort by importance (descending)
            importance_data.sort(key=lambda x: x["importance"], reverse=True)

            # Store in Redis (async client)
            await redis_client.set("ai_feature_importances", json.dumps(importance_data))
            await redis_client.set("ai_feature_importances_updated", _now_iso())

            # Log top 10 most important features
            top_features = importance_data[:10]
            logger.info(f"Top 10 most important features: {[f['name'] for f in top_features]}")

        except (ImportError, ValueError, TypeError, AttributeError) as e:
            logger.exception(f"Error analyzing feature importance: {e}")

    def _update_learning_progress(self, _batch_size: int) -> None:
        """Update AI learning progress metrics"""
        try:
            # Update learning progress metrics
            if AILearningReport is None:
                # Skip report generation if AILearningReport is not available
                logger.debug("AI learning report not available - skipping report generation")
                return

            # Create report generator
            report_generator = AILearningReport()  # type: ignore[misc]

            # Update metrics with new data points
            metrics = report_generator.update_metrics()

            # Log progress
            if metrics:
                progress = metrics.get("learning_progress", 0)
                total_points = metrics.get("total_data_points", 0)
                patterns = metrics.get("patterns_discovered", 0)

                logger.info(f"AI Learning Progress: {progress:.2f}% complete, {total_points:,} data points, {patterns:,} patterns")

        except (ImportError, ValueError, TypeError, AttributeError) as e:
            logger.exception(f"Error updating learning progress: {e}")

    PER_COIN_CANDIDATE_RETENTION = 10

    def _prune_per_coin_candidates(self, version_dir: Path, strat: str, sym: str) -> None:
        """Keep only the newest N candidate artifacts per (strategy, symbol).

        Candidates are written every training cycle; without retention the
        versions dir grows unbounded (observed 11k+ files / 2.1 GB).
        """
        try:
            files = sorted(
                version_dir.glob(f"{strat}_{sym}_*.pkl"),
                key=lambda p: p.name,
                reverse=True,
            )
            for stale in files[self.PER_COIN_CANDIDATE_RETENTION :]:
                stale.unlink(missing_ok=True)
        except OSError as e:
            logger.debug("PER_COIN_CANDIDATE_PRUNE failed for %s/%s: %s", strat, sym, e)

    async def _prune_old_models(self) -> None:
        """[MYSTIC_CORE_TAG: LEGACY_NOT_LIVE] Targets lightweight/ prefix globs; does not prune models/active/*_direction.pkl."""
        try:
            # Only run pruning every hour (3600 seconds)
            current_time = time.time()
            if current_time - self._last_prune_time < 3600:
                return

            self._last_prune_time = current_time

            model_dir = Path(self.model_versions_dir) / "lightweight"
            if not model_dir.exists():
                return

            # Get all model files grouped by type
            model_types = {
                "model_": [],  # RandomForest
                "gb_model_": [],  # GradientBoosting
                "xgb_model_": [],  # XGBoost
            }

            for file_path in model_dir.glob("*.pkl"):
                for prefix, file_list in model_types.items():
                    if file_path.name.startswith(prefix):
                        file_list.append(file_path)
                        break

            # For each type, keep only the latest N models
            total_deleted = 0
            for _prefix, files in model_types.items():
                if len(files) <= self.max_model_versions:
                    continue

                # Sort by modification time (newest first)
                files_sorted = sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)

                # Delete older files beyond the limit
                for file_to_delete in files_sorted[self.max_model_versions :]:
                    try:
                        file_to_delete.unlink()
                        total_deleted += 1
                    except OSError as e:
                        logger.debug(f"Could not delete {file_to_delete}: {e}")

            if total_deleted > 0:
                logger.info(f"Model pruning: deleted {total_deleted} old models, keeping {self.max_model_versions} per type")

        except Exception as e:
            logger.debug(f"Model pruning error (non-fatal): {e}")

    async def _generate_learning_report(self) -> None:
        """Generate periodic AI learning reports"""
        try:
            # Generate reports once every hour (3600 seconds)
            current_time = time.time()
            last_report_time_str = await self.cache.get("last_learning_report_time")

            if not last_report_time_str:
                last_report_time = 0
            else:
                try:
                    last_report_time = float(last_report_time_str)
                except (ValueError, TypeError):
                    last_report_time = 0

            if current_time - last_report_time >= 3600:  # Generate report every hour
                # Use optional report generator if available
                if AILearningReport is None:
                    logger.debug("AI learning report not available - skipping report generation")
                    return None

                # Create report generator
                report_generator = AILearningReport()  # type: ignore[misc]

                # Generate report
                report_path = report_generator.generate_report()

                if report_path:
                    logger.info(f"Generated AI learning report: {report_path}")

                    # Create visualization
                    viz_path = report_generator.visualize_progress()
                    if viz_path:
                        logger.info(f"Generated AI learning visualization: {viz_path}")

                # Update last report time
                await self.cache.set("last_learning_report_time", str(current_time))

        except (ImportError, ValueError, TypeError, AttributeError) as e:
            logger.exception(f"Error generating learning report: {e}")

    # Canonical per-coin artifact filename (must match ai_signal_generator.PER_COIN_MODEL_FILENAME)
    PER_COIN_MODEL_FILENAME = "{symbol}_direction.pkl"

    async def _train_lightweight_models(self, features: list[dict[str, Any]]):
        """Train one directional RF model per traded symbol and save to canonical per-coin path."""
        redis_client = get_shared_redis_async()
        if redis_client is None:
            logger.warning("Shared Redis client unavailable; skipping per-coin model training")
            return

        try:
            accuracy_key = "ai_model_accuracy"
            try:
                accuracy_raw = await redis_client.get(accuracy_key)
                if accuracy_raw:
                    self._current_accuracy = float(_decode(accuracy_raw) or "0.0")
            except (ConnectionError, OSError, AttributeError, TypeError, ValueError):
                pass

            if not features:
                self.performance_metrics["total_training_sessions"] += 1
                self.performance_metrics["last_training_time"] = _now_iso()
                try:
                    await self._publish_metrics(redis_client)
                except Exception:
                    pass
                return

            try:
                target_dim = _TARGET_FEATURE_DIM
                feature_version_used = _TARGET_FEATURE_VERSION if target_dim == _FEATURE_DIM_V2 else 1
                from backend.config.ai_primary_clock import primary_bar_seconds_for_strategy

                outcome_rows: list[dict[str, Any]] = []
                try:
                    from backend.services.ai_canonical_storage import read_recent_outcome_training_rows

                    outcome_rows = await asyncio.to_thread(read_recent_outcome_training_rows, limit=5000)
                except Exception as ob_e:
                    logger.debug("fetch ai_outcome_training_rows failed: %s", ob_e)

                self._last_outcome_rows_used = len(outcome_rows) if outcome_rows else 0

                active_dir = Path(ensure_model_directories()["active"])
                active_dir.mkdir(parents=True, exist_ok=True)
                version_dir = Path(self.model_versions_dir) / "per_coin"
                version_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

                total_trained = 0
                best_acc_global = 0.0
                acc_key = "ai_model_accuracy"

                for strat in train_strategy_ids():
                    if strat == "day":
                        rows_with_sym = []
                        for r in features:
                            if not isinstance(r, dict) or not r.get("symbol") or not isinstance(r.get("features"), (list, tuple)) or len(r["features"]) != target_dim:
                                continue
                            if int(r.get("feature_version") or 0) < FEATURE_VERSION_DAY_HTF or str(r.get("live_strategy_id") or "").strip().lower() != "day":
                                continue
                            try:
                                if float(r.get("label_anchor_close") or 0) <= 0:
                                    continue
                            except (TypeError, ValueError):
                                continue
                            rows_with_sym.append(r)
                    else:
                        rows_with_sym = [r for r in features if isinstance(r, dict) and r.get("symbol") and isinstance(r.get("features"), (list, tuple)) and len(r["features"]) == target_dim]

                    X_self, y_self, sym_self = build_xy_arrays_for_live_strategy(rows_with_sym, strat, target_dim=target_dim)
                    if len(X_self) == 0 and target_dim == _FEATURE_DIM_V2:
                        logger.warning(
                            "PER_COIN_TRAIN: strategy=%s zero v2 self-supervised rows — day/day rely on outcome merge or cache warm-up",
                            strat,
                        )

                    X_oc = y_oc = sym_oc = np.array([])
                    w_oc = np.array([])
                    if outcome_rows:
                        X_oc, y_oc, sym_oc, w_oc = _outcome_rows_to_xy_for_strategy(outcome_rows, strat, target_dim=target_dim)
                        if len(X_oc) > 0:
                            logger.info(
                                "AI_OUTCOME_TRAIN: strategy=%s merging %d realized-outcome rows (dim=%d)",
                                strat,
                                len(X_oc),
                                target_dim,
                            )

                    if len(X_oc) > 0:
                        if len(X_self) > 0:
                            X_all = np.vstack([X_self, X_oc])
                            _y_all = np.concatenate([y_self, y_oc])
                            _sym_all = np.concatenate([sym_self, sym_oc])
                        else:
                            X_all, _y_all, _sym_all = X_oc, y_oc, sym_oc
                    else:
                        X_all, _y_all, _sym_all = X_self, y_self, sym_self

                    if len(X_all) == 0:
                        logger.warning("PER_COIN_TRAIN: strategy=%s no labeled rows, skipping", strat)
                        continue

                    fv_log = int(FEATURE_VERSION_DAY_HTF) if strat == "day" else int(feature_version_used)
                    logger.info(
                        "PER_COIN_TRAIN: strategy=%s dim=%d fv=%d self_supervised=%d outcome=%d total=%d",
                        strat,
                        target_dim,
                        fv_log,
                        len(X_self),
                        len(X_oc),
                        len(X_all),
                    )

                    trained_count = 0
                    acc_sum = 0.0
                    best_acc = 0.0

                    from backend.services.ai_training_data_balance import prepare_outcome_weighted_training_arrays

                    for sym in TRADING_SYMBOLS:
                        if len(X_self) > 0:
                            ss_mask = sym_self == sym
                            X_ss = X_self[ss_mask]
                            y_ss = y_self[ss_mask]
                        else:
                            X_ss = np.empty((0, target_dim), dtype=np.float64)
                            y_ss = np.array([], dtype=np.int64)

                        if len(X_oc) > 0:
                            oc_mask = sym_oc == sym
                            X_oc_sym = X_oc[oc_mask]
                            y_oc_sym = y_oc[oc_mask]
                            w_oc_sym = w_oc[oc_mask] if len(w_oc) == len(sym_oc) else None
                        else:
                            X_oc_sym = np.empty((0, target_dim), dtype=np.float64)
                            y_oc_sym = np.array([], dtype=np.int64)
                            w_oc_sym = None

                        X_sym, y_sym, w_sym, balance_diag = prepare_outcome_weighted_training_arrays(
                            X_ss,
                            y_ss,
                            X_oc_sym,
                            y_oc_sym,
                            outcome_row_multipliers=w_oc_sym,
                        )

                        # Fetch Tier B/C BEFORE the min-sample gate so starvation
                        # relief can actually rescue symbols just under the floor
                        # (e.g. ETH 93 after balance). Validation stays Tier A-only.
                        tier_x: list = []
                        tier_y: list = []
                        tier_w: list = []
                        tier_b_n = 0
                        tier_c_n = 0
                        if strat == "day":
                            try:
                                from backend.services.ai_learning_ingestion import (
                                    MAX_TIER_BC_SHARE,
                                    TIER_B_WEIGHT,
                                    TIER_C_WEIGHT,
                                    tier_b_training_rows,
                                    tier_c_training_rows,
                                )

                                xb, yb = await asyncio.to_thread(tier_b_training_rows, strategy_id=strat, symbol=sym, feature_dim=target_dim)
                                xc, yc = await asyncio.to_thread(tier_c_training_rows, strategy_id=strat, symbol=sym, feature_dim=target_dim)
                                tier_b_n, tier_c_n = len(yb), len(yc)
                                tier_x = [*xb, *xc]
                                tier_y = [*[int(v) for v in yb], *[int(v) for v in yc]]
                                tier_w = [*([TIER_B_WEIGHT] * len(yb)), *([TIER_C_WEIGHT] * len(yc))]
                            except Exception as tier_e:
                                logger.debug("tiered train prefetch skipped for %s: %s", sym, tier_e)

                        min_samples = MIN_RF_TRAIN_SAMPLES
                        effective_n = len(X_sym) + len(tier_x)
                        if len(X_sym) < MIN_RF_TIER_A_ROWS or (len(X_sym) < min_samples and effective_n < min_samples):
                            logger.info(
                                "PER_COIN_TRAIN: [%s] %s only %d rows after balance (tier_bc=%d effective=%d need=%d tier_a_floor=%d) diag=%s, skipping",
                                strat,
                                sym,
                                len(X_sym),
                                len(tier_x),
                                effective_n,
                                min_samples,
                                MIN_RF_TIER_A_ROWS,
                                balance_diag,
                            )
                            continue

                        split_idx = int(len(X_sym) * 0.8)
                        X_train, X_val = X_sym[:split_idx], X_sym[split_idx:]
                        y_train, y_val = y_sym[:split_idx], y_sym[split_idx:]
                        w_train = w_sym[:split_idx]

                        # Merge Tier B/C into TRAIN only (val stays Tier A/outcome).
                        if tier_x:
                            try:
                                from backend.services.ai_learning_ingestion import MAX_TIER_BC_SHARE

                                max_rows = int(len(X_train) * MAX_TIER_BC_SHARE / max(1e-9, 1.0 - MAX_TIER_BC_SHARE))
                                use_x, use_y, use_w = tier_x, tier_y, tier_w
                                if len(use_x) > max_rows > 0:
                                    use_x = use_x[:max_rows]
                                    use_y = use_y[:max_rows]
                                    use_w = use_w[:max_rows]
                                if use_x:
                                    X_train = np.vstack([X_train, np.asarray(use_x, dtype=np.float64)])
                                    y_train = np.concatenate([y_train, np.asarray(use_y, dtype=np.int64)])
                                    w_train = np.concatenate([w_train, np.asarray(use_w, dtype=np.float64)])
                                    logger.info(
                                        "TIERED_TRAIN_MERGE: [%s] %s tier_b=%d tier_c=%d merged=%d train_total=%d",
                                        strat,
                                        sym,
                                        tier_b_n,
                                        tier_c_n,
                                        len(use_x),
                                        len(X_train),
                                    )
                            except Exception as tier_e:
                                logger.debug("tiered train merge skipped for %s: %s", sym, tier_e)

                        if len(X_train) < max(20, int(min_samples * 0.6)):
                            logger.info(
                                "PER_COIN_TRAIN: [%s] %s train too small after tier merge (%d), skipping",
                                strat,
                                sym,
                                len(X_train),
                            )
                            continue

                        if len(X_val) < 5:
                            logger.info(
                                "PER_COIN_TRAIN: [%s] %s val set too small (%d), skipping",
                                strat,
                                sym,
                                len(X_val),
                            )
                            continue

                        buy_n = int(np.sum(y_train == 1))
                        hold_n = int(np.sum(y_train == 0))
                        val_buy = int(np.sum(y_val == 1))
                        val_hold = int(np.sum(y_val == 0))
                        train_classes = np.unique(y_train)
                        from sklearn.utils.class_weight import compute_class_weight

                        balanced_w = compute_class_weight("balanced", classes=train_classes, y=y_train)
                        effective_weights = {int(c): round(float(w), 4) for c, w in zip(train_classes, balanced_w, strict=False)}
                        logger.info(
                            "PER_COIN_CLASS_BALANCE: [%s] %s train BUY=%d HOLD=%d val BUY=%d HOLD=%d class_weight=balanced effective=%s raw_ss=%s raw_oc=%s final_buy_rate=%s",
                            strat,
                            sym,
                            buy_n,
                            hold_n,
                            val_buy,
                            val_hold,
                            effective_weights,
                            balance_diag.get("raw_self_supervised"),
                            balance_diag.get("raw_outcome"),
                            balance_diag.get("final_buy_rate"),
                        )

                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train)
                        X_val_s = scaler.transform(X_val)

                        model = RandomForestClassifier(
                            n_estimators=50,
                            max_depth=10,
                            min_samples_split=5,
                            random_state=42,
                            n_jobs=-1,
                            class_weight="balanced",
                        )
                        model.fit(X_train_s, y_train, sample_weight=w_train)
                        acc = model.score(X_val_s, y_val)

                        # MODEL DIVERSITY: blend the RF with a HistGradientBoostingClassifier
                        # fit on the same data — a real second algorithm family (gradient-boosted
                        # trees vs RF's bagged trees), not a stub. Falls back to pure RF on thin
                        # data or fit failure. See ai_blended_classifier.py for the full rationale.
                        try:
                            from backend.services.ai_blended_classifier import build_blended_classifier

                            final_model, blend_telemetry = build_blended_classifier(
                                model, X_train_s, y_train, w_train, X_val_s, y_val
                            )
                        except Exception as blend_e:
                            logger.debug("BLENDED_CLASSIFIER_SKIPPED: [%s] %s (%s)", strat, sym, blend_e)
                            final_model, blend_telemetry = model, {
                                "rf_val_acc": round(float(acc), 4),
                                "gbm_val_acc": None,
                                "blend_w_rf": 1.0,
                                "blend_w_gbm": 0.0,
                                "blend_status": "rf_only_exception",
                            }
                        if blend_telemetry.get("blend_status") == "blended":
                            acc = float(final_model.score(X_val_s, y_val))

                        # CONFIDENCE CALIBRATION: raw model predict_proba is not a
                        # reliable win-rate estimate (never checked against actual
                        # outcomes before). Fit isotonic regression mapping raw
                        # BUY-class probability -> empirical val-set accuracy, so
                        # live "confidence" means something sizing can trust. Fits on
                        # the final (possibly blended) model's probabilities, not raw RF.
                        calibrator = None
                        try:
                            if len(X_val_s) >= 10 and len(np.unique(y_val)) > 1 and 1 in final_model.classes_:
                                from sklearn.isotonic import IsotonicRegression

                                buy_col = list(final_model.classes_).index(1)
                                raw_buy_probs = final_model.predict_proba(X_val_s)[:, buy_col]
                                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                                calibrator.fit(raw_buy_probs, y_val)
                                logger.info(
                                    "CONFIDENCE_CALIBRATION_FIT: [%s] %s val_n=%d raw_mean=%.3f cal_mean=%.3f",
                                    strat,
                                    sym,
                                    len(y_val),
                                    float(np.mean(raw_buy_probs)),
                                    float(np.mean(calibrator.predict(raw_buy_probs))),
                                )
                        except Exception as cal_e:
                            logger.debug("CONFIDENCE_CALIBRATION_SKIPPED: [%s] %s (%s)", strat, sym, cal_e)
                            calibrator = None

                        req_snap = required_edge_pct_for_strategy(strat)
                        t_frac, t_mfe = traction_params_for_strategy(strat)
                        _lah = label_horizon_bars_for_strategy(strat, sym)
                        _psec = primary_label_bar_seconds_for_strategy(strat)
                        label_max_hold_sec = int(_lah * _psec)
                        amb_move = LABEL_MIN_MOVE_PCT
                        if strat == "day":
                            amb_raw = os.getenv("DAY_RF_LABEL_MIN_MOVE_PCT")
                            if amb_raw is not None and str(amb_raw).strip() != "":
                                amb_move = float(amb_raw)

                        art_fv = int(FEATURE_VERSION_DAY_HTF) if strat == "day" else int(feature_version_used)
                        art_clock = "day_active_v5" if strat == "day" else "v3"
                        prim_sig_sec = int(day_label_grid_seconds()) if strat == "day" else int(primary_bar_seconds_for_strategy(strat))

                        artifact: dict[str, Any] = {
                            "model": final_model,
                            "scaler": scaler,
                            "confidence_calibrator": calibrator,
                            "accuracy": acc,
                            "rf_val_acc": blend_telemetry.get("rf_val_acc"),
                            "gbm_val_acc": blend_telemetry.get("gbm_val_acc"),
                            "blend_w_rf": blend_telemetry.get("blend_w_rf"),
                            "blend_w_gbm": blend_telemetry.get("blend_w_gbm"),
                            "blend_status": blend_telemetry.get("blend_status"),
                            "symbol": sym,
                            "feature_version": art_fv,
                            "feature_dim": int(target_dim),
                            "features": int(target_dim),
                            "train_samples": len(X_train),
                            "train_class_distribution": {"BUY": buy_n, "HOLD": hold_n},
                            "val_class_distribution": {"BUY": val_buy, "HOLD": val_hold},
                            "class_weight_mode": "balanced",
                            "class_effective_weights": effective_weights,
                            "training_balance": balance_diag,
                            "outcome_row_weight": balance_diag.get("outcome_row_weight"),
                            "self_supervised_row_weight": balance_diag.get("self_supervised_row_weight"),
                            "final_training_buy_rate": balance_diag.get("final_buy_rate"),
                            "trained_at": _now_iso(),
                            "label_version": TRADE_WORTHINESS_LABEL_VERSION,
                            "label_bar_seconds": int(_psec),
                            "label_max_hold_seconds": label_max_hold_sec,
                            "label_min_move_pct": amb_move,
                            "label_required_edge_pct": req_snap,
                            "label_traction_check_frac": t_frac,
                            "label_traction_min_mfe_pct": t_mfe,
                            "label_taker_fee": TAKER_FEE,
                            "label_slippage_pct": SLIPPAGE_PCT,
                            "label_spread_pct": default_spread_pct(),
                            "label_profit_edge_buffer_pct": profit_edge_buffer_pct(),
                            "live_strategy_id": strat,
                            "label_lookahead_bars": _lah,
                            "label_lookahead": _lah,
                            "primary_signal_bar_seconds": prim_sig_sec,
                            "ai_clock_contract": art_clock,
                            "day_htf_contract": "",
                        }
                        if strat == "day":
                            from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES
                            from backend.services.ai_decision_contract import CONTEXT_DIMS_DAY_FULL

                            artifact["day_htf_contract"] = "day_active_bundle_v5"
                            artifact["day_active_timeframes"] = list(DAY_ACTIVE_TIMEFRAMES)
                            artifact["context_dims_day_full"] = list(CONTEXT_DIMS_DAY_FULL)

                        coin_path = per_coin_artifact_file(active_dir, strat, sym)
                        coin_path.parent.mkdir(parents=True, exist_ok=True)
                        ver_path = version_dir / f"{strat}_{sym}_{ts}.pkl"
                        with ver_path.open("wb") as f:
                            pickle.dump(artifact, f)
                        promo_state = "promoted"
                        try:
                            from backend.services.ai_model_promotion import register_candidate_and_maybe_promote
                            from backend.services.ai_model_promotion_holdout import build_holdout_validation_metrics

                            validation_metrics = build_holdout_validation_metrics(
                                strategy_id=strat,
                                symbol_bus=sym,
                                candidate_path=ver_path,
                                active_path=coin_path if coin_path.exists() else None,
                                feature_version=art_fv,
                                feature_dim=int(target_dim),
                                rf_val_samples=len(X_val),
                            )
                            artifact["holdout_accuracy"] = validation_metrics.get("candidate_accuracy")
                            artifact["holdout_sample_count"] = validation_metrics.get("holdout_sample_count")
                            artifact["holdout_profit_after_cost"] = validation_metrics.get("candidate_profit_after_cost")
                            artifact["holdout_buy_label_count"] = validation_metrics.get("holdout_buy_label_count")
                            with ver_path.open("wb") as f:
                                pickle.dump(artifact, f)

                            promoted, promo_reason = register_candidate_and_maybe_promote(
                                strategy_id=strat,
                                symbol=sym,
                                candidate_path=ver_path,
                                active_path=coin_path,
                                validation_metrics=validation_metrics,
                            )
                            if not promoted:
                                promo_state = f"rejected:{promo_reason}"
                                logger.info(
                                    "MODEL_PROMOTION_REJECTED: strategy=%s symbol=%s reason=%s",
                                    strat,
                                    sym,
                                    promo_reason,
                                )
                        except Exception as promote_err:
                            promo_state = "fallback_direct_write"
                            logger.warning("MODEL_PROMOTION_FALLBACK: %s/%s (%s) — writing active directly", strat, sym, promote_err)
                            with coin_path.open("wb") as f:
                                pickle.dump(artifact, f)
                            shutil.copy2(coin_path, ver_path)

                        self._prune_per_coin_candidates(version_dir, strat, sym)

                        trained_count += 1
                        acc_sum += acc
                        best_acc = max(best_acc, acc)
                        best_acc_global = max(best_acc_global, acc)
                        logger.info(
                            "PER_COIN_TRAINED: [%s] %s accuracy=%.4f samples=%d candidate=%s promotion=%s active=%s",
                            strat,
                            sym,
                            acc,
                            len(X_train),
                            ver_path.name,
                            promo_state,
                            str(coin_path),
                        )

                    if trained_count > 0:
                        avg_acc = acc_sum / trained_count
                        await redis_client.set(f"{acc_key}:{strat}", str(avg_acc))
                        await redis_client.set(f"ai_model_last_trained:{strat}", _now_iso())
                        if strat == "day":
                            await redis_client.set(acc_key, str(avg_acc))
                            self._current_accuracy = avg_acc
                            self.performance_metrics["average_accuracy"] = avg_acc
                        self.performance_metrics["best_accuracy"] = max(self.performance_metrics.get("best_accuracy", 0.0), best_acc)
                        self.performance_metrics["models_trained"] = self.performance_metrics.get("models_trained", 0) + trained_count
                        logger.info(
                            "PER_COIN_TRAIN_COMPLETE: strategy=%s trained=%d/%d avg_acc=%.4f fv=%d dim=%d",
                            strat,
                            trained_count,
                            len(TRADING_SYMBOLS),
                            avg_acc,
                            feature_version_used,
                            target_dim,
                        )

                    total_trained += trained_count

                if total_trained > 0:
                    self.performance_metrics["successful_trainings"] = self.performance_metrics.get("successful_trainings", 0) + 1
                    await redis_client.set("ai_model_features", str(target_dim))
                    await redis_client.set("ai_model_feature_version", str(feature_version_used))
                    logger.info(
                        "PER_COIN_TRAIN_SESSION: total_symbol_models=%d best_acc=%.4f dim=%d",
                        total_trained,
                        best_acc_global,
                        target_dim,
                    )
                else:
                    logger.warning("PER_COIN_TRAIN: no symbols had enough data to train for any strategy")

            except ImportError as e:
                logger.exception(f"Required library not available: {e}")
            except Exception as e:
                logger.exception(f"Error during per-coin model training: {e}")
                self.performance_metrics["failed_trainings"] = self.performance_metrics.get("failed_trainings", 0) + 1

            self.performance_metrics["last_training_time"] = _now_iso()
            try:
                await self._publish_metrics(redis_client)
            except Exception:
                pass

        except (ValueError, TypeError, AttributeError, RuntimeError, ConnectionError) as e:
            logger.exception(f"Error in per-coin model training: {e}")
            self.performance_metrics["failed_trainings"] = self.performance_metrics.get("failed_trainings", 0) + 1

    async def _publish_metrics(self, redis_client) -> None:
        """Persist live training metrics to Redis for downstream consumers."""
        if redis_client is None:
            return

        try:
            max(0.0, time.time() - self._start_time)

            # Get learning metrics from report generator (same data as ai_learning_report.py)
            learning_metrics = {}
            try:
                if AILearningReport is not None:
                    report_gen = AILearningReport()
                    learning_metrics = report_gen.update_metrics() or {}
            except Exception as e:
                logger.debug(f"Could not get learning metrics: {e}")

            # Format data for downstream consumers
            training_stats = {
                # Basic training status
                "status": "active" if self.is_running else "inactive",
                "pipeline_running": self.is_running,
                "current_accuracy": float(self._current_accuracy),
                "sessions": int(self.performance_metrics.get("total_training_sessions", 0)),
                "successful_trainings": int(self.performance_metrics.get("successful_trainings", 0)),
                "failed_trainings": int(self.performance_metrics.get("failed_trainings", 0)),
                "best_accuracy": float(self.performance_metrics.get("best_accuracy", 0.0)),
                "average_accuracy": float(self.performance_metrics.get("average_accuracy", 0.0)),
                "last_training_time": self.performance_metrics.get("last_training_time"),
                "models_trained": int(self.performance_metrics.get("models_trained", 0)),
                "timestamp": _now_iso(),
                # Learning metrics
                "total_data_points": int(learning_metrics.get("total_data_points", 0)),
                "patterns_discovered": int(learning_metrics.get("patterns_discovered", 0)),
                "accuracy_improvement": float(learning_metrics.get("accuracy_improvement", 0.0)),
                "learning_progress": float(learning_metrics.get("learning_progress", 0.0)),
                "active_strategies": int(learning_metrics.get("active_strategies", 0)),
                # Training status object
                "training_status": {
                    "status": "active" if self.is_running else "inactive",
                    "current_epoch": int(self.performance_metrics.get("current_epoch", 0)),
                    "total_epochs": int(self.performance_metrics.get("total_epochs", 0)),
                },
                # Additional metadata
                "last_updated": _now_iso(),
                "source": "redis",
            }

            await redis_client.set("ai_learning_stats", json.dumps(training_stats))
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Failed to publish training metrics: {e}")

    def create_feature_vector(self, market_data: dict[str, Any], mystic_data: dict[str, Any], trade_data: dict[str, Any]) -> list[float]:
        """
        [MYSTIC_CORE_TAG: LEGACY_NOT_LIVE] Not used by RF training path; canonical 124-vector is build_feature_vector_124.

        Create a comprehensive feature vector from market, mystic and trade data
        This method extracts and normalizes 124 features for advanced AI training
        """
        # Initialize empty feature vector
        feature_vector = []

        try:
            # Helper function for safe float conversion
            def safe_float(value: Any, default: float = 0.0) -> float:
                try:
                    if isinstance(value, (int, float)):
                        result = float(value)
                    elif isinstance(value, str):
                        result = float(value.replace("%", "").replace(",", "").strip())
                    else:
                        result = default
                except (ValueError, TypeError, AttributeError):
                    return default
                else:
                    return result

            # === BASIC PRICE FEATURES (10) ===
            if market_data:
                close_price = safe_float(market_data.get("close", 0))
                open_price = safe_float(market_data.get("open", 0))
                high_price = safe_float(market_data.get("high", 0))
                low_price = safe_float(market_data.get("low", 0))
                volume = safe_float(market_data.get("volume", 0))

                feature_vector.extend(
                    [
                        close_price,  # 1. Current price
                        high_price,  # 2. High price
                        low_price,  # 3. Low price
                        open_price,  # 4. Open price
                        volume,  # 5. Volume
                        safe_float(market_data.get("change_24h", 0)),  # 6. 24h change
                        safe_float(market_data.get("change_7d", 0)),  # 7. 7d change
                        safe_float(market_data.get("change_30d", 0)),  # 8. 30d change
                        high_price - low_price,  # 9. Price range
                        (high_price + low_price + close_price) / 3,  # 10. Typical price
                    ]
                )
            else:
                feature_vector.extend([0.0] * 10)

            # === TECHNICAL INDICATORS (24) ===
            # Moving Averages
            if market_data:
                feature_vector.extend(
                    [
                        safe_float(market_data.get("ma_5", 0)),  # 11. MA5
                        safe_float(market_data.get("ma_10", 0)),  # 12. MA10
                        safe_float(market_data.get("ma_20", 0)),  # 13. MA20
                        safe_float(market_data.get("ma_50", 0)),  # 14. MA50
                        safe_float(market_data.get("ma_100", 0)),  # 15. MA100
                        safe_float(market_data.get("ma_200", 0)),  # 16. MA200
                    ]
                )

                # Exponential Moving Averages
                feature_vector.extend(
                    [
                        safe_float(market_data.get("ema_12", 0)),  # 17. EMA12
                        safe_float(market_data.get("ema_26", 0)),  # 18. EMA26
                        safe_float(market_data.get("ema_50", 0)),  # 19. EMA50
                    ]
                )

                # Oscillators
                feature_vector.extend(
                    [
                        safe_float(market_data.get("rsi", 50)),  # 20. RSI
                        safe_float(market_data.get("rsi_14", 50)),  # 21. RSI14
                        safe_float(market_data.get("stoch_k", 50)),  # 22. Stochastic %K
                        safe_float(market_data.get("stoch_d", 50)),  # 23. Stochastic %D
                        safe_float(market_data.get("williams_r", -50)),  # 24. Williams %R
                        safe_float(market_data.get("cci", 0)),  # 25. Commodity Channel Index
                    ]
                )

                # MACD Components
                feature_vector.extend(
                    [
                        safe_float(market_data.get("macd", 0)),  # 26. MACD
                        safe_float(market_data.get("macd_signal", 0)),  # 27. MACD Signal
                        safe_float(market_data.get("macd_histogram", 0)),  # 28. MACD Histogram
                    ]
                )

                # Bollinger Bands
                feature_vector.extend(
                    [
                        safe_float(market_data.get("bb_upper", 0)),  # 29. BB Upper
                        safe_float(market_data.get("bb_middle", 0)),  # 30. BB Middle
                        safe_float(market_data.get("bb_lower", 0)),  # 31. BB Lower
                        safe_float(market_data.get("bb_position", 0.5)),  # 32. BB Position
                        safe_float(market_data.get("bb_width", 0)),  # 33. BB Width
                    ]
                )

                # Volume Indicators
                feature_vector.extend(
                    [
                        safe_float(market_data.get("obv", 0)),  # 34. On-Balance Volume
                        safe_float(market_data.get("ad_line", 0)),  # 35. A/D Line
                        safe_float(market_data.get("cmf", 0)),  # 36. Chaikin Money Flow
                        safe_float(market_data.get("mfi", 50)),  # 37. Money Flow Index
                    ]
                )
            else:
                feature_vector.extend([0.0] * 24)

            # === VOLATILITY INDICATORS (10) ===
            if market_data:
                feature_vector.extend(
                    [
                        safe_float(market_data.get("volatility")),  # 38. Volatility (0 if unknown)
                        safe_float(market_data.get("atr", 0)),  # 39. Average True Range
                        safe_float(market_data.get("natr", 0)),  # 40. Normalized ATR
                        safe_float(market_data.get("keltner_upper", 0)),  # 41. Keltner Upper
                        safe_float(market_data.get("keltner_lower", 0)),  # 42. Keltner Lower
                        safe_float(market_data.get("donchian_upper", 0)),  # 43. Donchian Upper
                        safe_float(market_data.get("donchian_lower", 0)),  # 44. Donchian Lower
                        safe_float(market_data.get("parabolic_sar", 0)),  # 45. Parabolic SAR
                        safe_float(market_data.get("volatility_ratio", 1)),  # 46. Volatility Ratio
                        safe_float(market_data.get("price_volatility", 0)),  # 47. Price Volatility
                    ]
                )
            else:
                feature_vector.extend([0.0] * 10)

            # === MOMENTUM INDICATORS (15) ===
            if market_data:
                feature_vector.extend(
                    [
                        safe_float(market_data.get("roc", 0)),  # 48. Rate of Change
                        safe_float(market_data.get("momentum", 0)),  # 49. Momentum
                        safe_float(market_data.get("ppo", 0)),  # 50. Percentage Price Oscillator
                        safe_float(market_data.get("trix", 0)),  # 51. TRIX
                        safe_float(market_data.get("ultimate_oscillator", 50)),  # 52. Ultimate Oscillator
                        safe_float(market_data.get("awesome_oscillator", 0)),  # 53. Awesome Oscillator
                        safe_float(market_data.get("balance_of_power", 0)),  # 54. Balance of Power
                        safe_float(market_data.get("ease_of_movement", 0)),  # 55. Ease of Movement
                        safe_float(market_data.get("mass_index", 25)),  # 56. Mass Index
                        safe_float(market_data.get("vortex_vi_plus", 1)),  # 57. Vortex VI+
                        safe_float(market_data.get("vortex_vi_minus", 1)),  # 58. Vortex VI-
                        safe_float(market_data.get("kst", 0)),  # 59. Know Sure Thing
                        safe_float(market_data.get("tsi", 0)),  # 60. True Strength Index
                        safe_float(market_data.get("aroon_up", 50)),  # 61. Aroon Up
                        safe_float(market_data.get("aroon_down", 50)),  # 62. Aroon Down
                    ]
                )
            else:
                feature_vector.extend([0.0] * 15)

            # === TREND INDICATORS (10) ===
            if market_data:
                feature_vector.extend(
                    [
                        safe_float(market_data.get("adx", 25)),  # 63. ADX
                        safe_float(market_data.get("di_plus", 25)),  # 64. DI+
                        safe_float(market_data.get("di_minus", 25)),  # 65. DI-
                        safe_float(market_data.get("aroon_oscillator", 0)),  # 66. Aroon Oscillator
                        safe_float(market_data.get("ichimoku_tenkan", 0)),  # 67. Ichimoku Tenkan
                        safe_float(market_data.get("ichimoku_kijun", 0)),  # 68. Ichimoku Kijun
                        safe_float(market_data.get("ichimoku_senkou_a", 0)),  # 69. Ichimoku Senkou A
                        safe_float(market_data.get("ichimoku_senkou_b", 0)),  # 70. Ichimoku Senkou B
                        safe_float(market_data.get("psar", 0)),  # 71. Parabolic SAR
                        safe_float(market_data.get("trend_strength", 0)),  # 72. Trend Strength
                    ]
                )
            else:
                feature_vector.extend([0.0] * 10)

            # === VOLUME PROFILE (8) ===
            if market_data:
                feature_vector.extend(
                    [
                        safe_float(market_data.get("volume_ma_5", 0)),  # 73. Volume MA5
                        safe_float(market_data.get("volume_ma_10", 0)),  # 74. Volume MA10
                        safe_float(market_data.get("volume_ma_20", 0)),  # 75. Volume MA20
                        safe_float(market_data.get("volume_ratio", 1)),  # 76. Volume Ratio
                        safe_float(market_data.get("volume_price_trend", 0)),  # 77. Volume Price Trend
                        safe_float(market_data.get("negative_volume_index", 1000)),  # 78. Negative Volume Index
                        safe_float(market_data.get("positive_volume_index", 1000)),  # 79. Positive Volume Index
                        safe_float(market_data.get("volume_weighted_price", 0)),  # 80. Volume Weighted Price
                    ]
                )
            else:
                feature_vector.extend([0.0] * 8)

            # === MARKET SENTIMENT (10) ===
            if mystic_data:
                sentiment_data = mystic_data.get("sentiment", {})
                market_data = market_data or {}
                feature_vector.extend(
                    [
                        safe_float(sentiment_data.get("fear_greed_index", 50)),  # 81. Fear & Greed Index
                        safe_float(sentiment_data.get("social_sentiment", 0.5)),  # 82. Social Sentiment
                        safe_float(sentiment_data.get("news_sentiment", 0.5)),  # 83. News Sentiment
                        safe_float(sentiment_data.get("put_call_ratio", 1)),  # 84. Put/Call Ratio
                        safe_float(sentiment_data.get("vix", 20)),  # 85. VIX
                        safe_float(market_data.get("market_cap", 0)),  # 86. Market Cap
                        safe_float(market_data.get("supply", 0)),  # 87. Supply
                        safe_float(market_data.get("circulating_supply", 0)),  # 88. Circulating Supply
                        safe_float(market_data.get("max_supply", 0)),  # 89. Max Supply
                        safe_float(market_data.get("market_dominance", 0)),  # 90. Market Dominance
                    ]
                )
            else:
                feature_vector.extend([0.0] * 10)

            # === TIME-BASED FEATURES (10) ===
            now = datetime.now(timezone.utc)
            timestamp = market_data.get("timestamp") if market_data else None
            if timestamp:
                try:
                    if isinstance(timestamp, (int, float)):
                        dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 10000000000 else timestamp, tz=timezone.utc)
                    else:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except (ValueError, TypeError, AttributeError):
                    dt = now
            else:
                dt = now

            feature_vector.extend(
                [
                    float(dt.hour),  # 91. Hour of day
                    float(dt.weekday()),  # 92. Day of week
                    float(dt.day),  # 93. Day of month
                    float(dt.month),  # 94. Month
                    float(dt.isoweekday()),  # 95. ISO weekday
                    float(dt.timetuple().tm_yday),  # 96. Day of year
                    float(dt.hour % 12),  # 97. Hour in 12h format
                    float(dt.minute),  # 98. Minute
                    float(dt.second),  # 99. Second
                    float(dt.timestamp() % 86400),  # 100. Seconds since midnight
                ]
            )

            # === ADDITIONAL ADVANCED INDICATORS (24) ===
            # Advanced Technical Analysis
            if market_data:
                feature_vector.extend(
                    [
                        safe_float(market_data.get("fibonacci_retracement_23.6", 0)),  # 101. Fib 23.6%
                        safe_float(market_data.get("fibonacci_retracement_38.2", 0)),  # 102. Fib 38.2%
                        safe_float(market_data.get("fibonacci_retracement_61.8", 0)),  # 103. Fib 61.8%
                        safe_float(market_data.get("pivot_point", 0)),  # 104. Pivot Point
                        safe_float(market_data.get("resistance_1", 0)),  # 105. Resistance 1
                        safe_float(market_data.get("resistance_2", 0)),  # 106. Resistance 2
                        safe_float(market_data.get("support_1", 0)),  # 107. Support 1
                        safe_float(market_data.get("support_2", 0)),  # 108. Support 2
                    ]
                )
            else:
                feature_vector.extend([0.0] * 8)

            # Advanced Volume Analysis
            if market_data:
                feature_vector.extend(
                    [
                        safe_float(market_data.get("volume_profile_poc", 0)),  # 109. Volume Profile POC
                        safe_float(market_data.get("volume_profile_vah", 0)),  # 110. Volume Profile VAH
                        safe_float(market_data.get("volume_profile_val", 0)),  # 111. Volume Profile VAL
                        safe_float(market_data.get("vwap", 0)),  # 112. VWAP
                        safe_float(market_data.get("twap", 0)),  # 113. TWAP
                        safe_float(market_data.get("volume_imbalance", 0)),  # 114. Volume Imbalance
                        safe_float(market_data.get("volume_delta", 0)),  # 115. Volume Delta
                        safe_float(market_data.get("order_flow", 0)),  # 116. Order Flow
                    ]
                )
            else:
                feature_vector.extend([0.0] * 8)

            # Advanced Market Microstructure
            if trade_data:
                feature_vector.extend(
                    [
                        safe_float(trade_data.get("bid_ask_spread", 0)),  # 117. Bid-Ask Spread
                        safe_float(trade_data.get("order_book_imbalance", 0)),  # 118. Order Book Imbalance
                        safe_float(trade_data.get("market_depth", 0)),  # 119. Market Depth
                        safe_float(trade_data.get("liquidity_score", 0)),  # 120. Liquidity Score
                        safe_float(trade_data.get("price_impact", 0)),  # 121. Price Impact
                        safe_float(trade_data.get("market_efficiency", 0)),  # 122. Market Efficiency
                        safe_float(trade_data.get("volatility_smile", 0)),  # 123. Volatility Smile
                        safe_float(trade_data.get("price_skewness", trade_data.get("skewness", 0))),  # 124. Price Skewness
                    ]
                )
            else:
                feature_vector.extend([0.0] * 8)

            # Ensure all values are valid floats - VECTORIZED for performance
            feature_array = np.array(feature_vector, dtype=float)
            # Replace non-finite values with 0.0 using vectorized operations
            feature_array[~np.isfinite(feature_array)] = 0.0
            feature_vector = feature_array.tolist()

            # Ensure we have exactly 124 features
            if len(feature_vector) != 124:
                # Pad or truncate to 124 features without noisy warnings
                while len(feature_vector) < 124:
                    feature_vector.append(0.0)
                feature_vector = feature_vector[:124]
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
            logger.exception("Error creating feature vector: %s", e)
            # All Live Data, No Fallback/Hardcoded Data
            # Raise exception instead of returning fallback zero vector
            msg = f"Failed to create feature vector from live data: {e}"
            raise RuntimeError(msg) from e
        else:
            # Return the feature vector
            return feature_vector

    async def collect_training_data(self):
        """Collect training rows: day v3 (5m-primary); day v4 (1h/4h/1d/1w HTF + ctx, 4h label grid)."""
        from backend.config.ai_primary_clock import primary_bar_seconds_for_strategy
        from backend.services.ai_day_htf_features import build_day_htf_feature_vector_145
        from backend.services.ai_feature_fundamentals import merge_canonical_sentiment_payload
        from backend.services.ai_feature_v3 import build_feature_vector_v3
        from backend.services.ai_market_context import hydrate_ai_context_payload
        from backend.services.history_context_gates import (
            min_ohlcv_bars_for_signal,
            min_primary_bars_for_strategy,
            ohlcv_1m_fetch_limit_for_primary,
        )
        from backend.services.live_market_data import live_market_data_service
        from backend.services.ohlcv_resample import resample_ohlcv_to_seconds
        from backend.utils.canonical_symbol_formatter import CanonicalSymbolFormatter

        features: list[float] = []
        total_new_anchors = 0
        try:
            strategies = list(train_strategy_ids())
            if not strategies:
                strategies = ["day", "day"]
            flat = int(time.time() // 10) % max(1, (len(TOP10_COINS) * len(strategies)))
            strat = strategies[flat // len(TOP10_COINS)]
            rotation_coin = TOP10_COINS[flat % len(TOP10_COINS)]
            bases_raw = os.getenv("DAY_HISTORICAL_TRAIN_BASES", "").strip().upper()
            if strat == "day" and bases_raw:
                wanted = [b.strip() for b in bases_raw.split(",") if b.strip()]
                coin_list = [b for b in wanted if b in TOP10_COINS]
                if not coin_list:
                    coin_list = list(TOP10_COINS)
            elif strat == "day":
                coin_list = list(TOP10_COINS)
            else:
                coin_list = [rotation_coin]
            day_hist_cap = max(
                MIN_RF_TRAIN_SAMPLES,
                int(os.getenv("DAY_HISTORICAL_ROWS_PER_COLLECT", "140")),
            )
            anchor_stride = max(1, int(os.getenv("DAY_HISTORICAL_ANCHOR_STRIDE", "3")))

            if strat == "day" and not live_market_data_service:
                return []

            for current_coin in coin_list:
                ccxt_symbol = f"{current_coin}/USDT"
                training_pair = _canonical_training_pair_symbol(str(current_coin))
                cache_ck = f"{training_pair}_{strat}"
                symbol_hint = current_coin

                order_book_features = await self._fetch_order_book_features(symbol_hint)
                trade_data = order_book_features or {}

                vp = None
                try:
                    redis_client = get_shared_redis_async()
                    if redis_client:
                        raw_vp = await redis_client.hgetall(f"volume_profile:{symbol_hint}")
                        if raw_vp:
                            vp = {k.decode() if isinstance(k, bytes) else k: float(v.decode() if isinstance(v, bytes) else v) for k, v in raw_vp.items()}
                except Exception as ex:
                    logger.debug("Volume profile fetch failed for %s: %s", symbol_hint, ex)
                    vp = None

                ai_ctx_dict: dict[str, Any] = {}
                try:
                    redis_client = get_shared_redis_async()
                    if redis_client:
                        raw_ctx = await redis_client.hgetall(f"ai_context:{training_pair}")
                        if raw_ctx:
                            ai_ctx_dict = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in raw_ctx.items()}
                except Exception as ctx_e:
                    logger.debug("ai_context fetch for %s failed: %s", training_pair, ctx_e)

                try:
                    ai_ctx_dict = await hydrate_ai_context_payload(training_pair, ai_ctx_dict)
                except Exception as hy_e:
                    logger.debug("hydrate_ai_context_payload skipped: %s", hy_e)

                sentiment_payload: dict[str, Any] | None = None
                try:
                    from backend.services.ai_decision_contract import REDIS_KEY_AI_SENTIMENT

                    redis_client = get_shared_redis_async()
                    if redis_client:
                        raw_s = await redis_client.get(REDIS_KEY_AI_SENTIMENT)
                        if raw_s:
                            sdec = raw_s.decode() if isinstance(raw_s, bytes) else raw_s
                            sentiment_payload = {"fear_greed_index": float(sdec)}
                        if (not sentiment_payload or sentiment_payload.get("fear_greed_index", 0.0) == 0.0) and ai_ctx_dict:
                            raw_fg = ai_ctx_dict.get("ctx_sentiment_fear_greed")
                            if raw_fg is not None and str(raw_fg).strip() != "":
                                sentiment_payload = sentiment_payload or {}
                                sentiment_payload["fear_greed_index"] = float(raw_fg)
                except (TypeError, ValueError, AttributeError) as se:
                    logger.debug("sentiment redis for training sample failed: %s", se)

                if strat == "day":
                    from backend.services.day_active_market_bundle import (
                        async_fetch_day_active_ohlcv_bundle,
                        async_fetch_day_active_ohlcv_bundle_asof,
                        validate_day_active_bundle,
                    )

                    if not live_market_data_service:
                        continue
                    bundle_ref = await async_fetch_day_active_ohlcv_bundle(live_market_data_service, ccxt_symbol)
                    bundle_ok, miss_bundle = validate_day_active_bundle(bundle_ref)
                    if not bundle_ok:
                        logger.debug(
                            "DAY_ACTIVE_TRAIN_SKIP insufficient bundle %s missing=%s",
                            current_coin,
                            miss_bundle,
                        )
                        continue
                    oh4 = bundle_ref.get("4h") or []
                    if len(oh4) < 2:
                        continue
                    gsec = int(day_label_grid_seconds())
                    last_k = len(oh4) - 1
                    emitted_this_coin = 0
                    redis_client_lm = get_shared_redis_async()

                    tail_span_bars = int(
                        os.getenv(
                            "DAY_HISTORICAL_TAIL_4H_BARS",
                            str(min(len(oh4), day_hist_cap * anchor_stride)),
                        )
                    )
                    tail_start = max(0, len(oh4) - max(anchor_stride, tail_span_bars))
                    for k in range(tail_start, len(oh4), anchor_stride):
                        if emitted_this_coin >= day_hist_cap:
                            break
                        anchor_open = int(oh4[k][0])
                        if anchor_open in self._day_emitted_anchor_ts[cache_ck]:
                            continue

                        bar_end_exclusive = int(oh4[k + 1][0]) if k + 1 < len(oh4) else None
                        asof_et = bar_end_exclusive - 1 if bar_end_exclusive is not None else int(time.time() * 1000)
                        trimmed = await async_fetch_day_active_ohlcv_bundle_asof(
                            live_market_data_service,
                            ccxt_symbol,
                            asof_et,
                        )
                        trimmed_ok, _miss_t = validate_day_active_bundle(trimmed)
                        if not trimmed_ok:
                            continue
                        is_latest = k == last_k
                        label_anchor_close = float(oh4[k][4])
                        if label_anchor_close <= 0:
                            continue

                        tb_sent = trimmed.get("1m") or []
                        vp_use = vp if is_latest else None
                        ob_use = trade_data if is_latest else None
                        sentiment_use: dict[str, Any] | None = None
                        ai_ctx_use: dict[str, Any] = ai_ctx_dict if is_latest else {}
                        if is_latest:
                            sentiment_use = sentiment_payload
                            if redis_client_lm and tb_sent:
                                try:
                                    sentiment_use = await merge_canonical_sentiment_payload(
                                        base_symbol=CanonicalSymbolFormatter.to_base(training_pair),
                                        pair_symbol=training_pair,
                                        ctx_for_overlay=ai_ctx_dict if isinstance(ai_ctx_dict, dict) else None,
                                        redis_client=redis_client_lm,
                                        ohlcv_1m=tb_sent if isinstance(tb_sent, list) else [],
                                        existing=sentiment_payload,
                                    )
                                except Exception as fe:
                                    logger.debug(
                                        "merge_canonical_sentiment_payload (day training) skipped: %s",
                                        fe,
                                    )
                                    sentiment_use = sentiment_payload
                        features_k = build_day_htf_feature_vector_145(
                            symbol_ccxt=ccxt_symbol,
                            day_bundle=trimmed,
                            volume_profile=vp_use,
                            orderbook=ob_use,
                            sentiment=sentiment_use,
                            ai_context=ai_ctx_use,
                        )

                        cutoff_iso = datetime.fromtimestamp(anchor_open / 1000.0, tz=timezone.utc).isoformat()

                        try:
                            if is_latest and len(features_k) == _FEATURE_DIM_V2:
                                await asyncio.to_thread(
                                    lambda tp=training_pair, feat=list(features_k): persist_ai_feature_sample_row(
                                        symbol=tp,
                                        context_key=CANONICAL_TELEMETRY_CONTEXT_KEY_DAY_HTF,
                                        features=feat,
                                        feature_version=int(FEATURE_VERSION_DAY_HTF),
                                    ),
                                )
                        except (TypeError, ValueError, RuntimeError) as pe:
                            logger.debug("persist_ai_feature_sample_row (day HTF) skipped: %s", pe)

                        self._day_emitted_anchor_ts[cache_ck].add(anchor_open)
                        emitted_this_coin += 1

                        logger.debug(
                            "Collected day ACTIVE training row hist %s anchor_ms=%s latest=%s ctx_full=%s",
                            current_coin,
                            anchor_open,
                            is_latest,
                            is_latest,
                        )

                        timestamp = cutoff_iso
                        feature_data_k = {
                            "timestamp": timestamp,
                            "collection_time": cutoff_iso,
                            "features": features_k,
                            "feature_count": len(features_k),
                            "feature_version": int(FEATURE_VERSION_DAY_HTF),
                            "symbol": training_pair,
                            "live_strategy_id": strat,
                            "primary_bar_seconds": int(gsec),
                            "ai_clock_contract": "day_active_v5",
                            "label_anchor_close": label_anchor_close,
                            "label_anchor_4h_open_ms": anchor_open,
                            "day_htf_bar_counts": {tf: len(trimmed.get(tf) or []) for tf in trimmed if isinstance(tf, str) and not tf.startswith("_")},
                        }
                        symbol_cache = self.training_cache.setdefault(cache_ck, [])
                        symbol_cache.append(feature_data_k)
                        if len(symbol_cache) > self.max_training_cache_size:
                            self.training_cache[cache_ck] = symbol_cache[-self.max_training_cache_size :]

                        self.last_collection[cache_ck] = timestamp
                        features = list(features_k)

                        if len(symbol_cache) % 100 == 0:
                            try:
                                Path(self.training_data_dir).mkdir(parents=True, exist_ok=True)
                                safe_bus = training_pair.replace("/", "_")
                                latest_path = Path(self.training_data_dir) / f"{safe_bus}_{strat}_latest.json"
                                save_count = min(5000, len(symbol_cache), self.max_training_cache_size)
                                with latest_path.open("w") as f:
                                    json.dump(symbol_cache[-save_count:], f)
                                if len(symbol_cache) % 1000 == 0:
                                    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                                    archive_path = Path(self.training_data_dir) / f"{safe_bus}_{strat}_{timestamp_str}.json"
                                    with archive_path.open("w") as f:
                                        json.dump(symbol_cache[-1000:], f)
                            except (OSError, FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
                                logger.debug("Error persisting training data: %s", e)

                        self.log_training_progress(cache_ck, features_k)

                    logger.info(
                        "DAY_COLLECT_HISTORICAL %s new_anchors=%d training_cache_rows=%d",
                        training_pair,
                        emitted_this_coin,
                        len(self.training_cache.get(cache_ck, [])),
                    )
                    total_new_anchors += emitted_this_coin
                    if emitted_this_coin:
                        try:
                            Path(self.training_data_dir).mkdir(parents=True, exist_ok=True)
                            safe_bus = training_pair.replace("/", "_")
                            latest_path = Path(self.training_data_dir) / f"{safe_bus}_{strat}_latest.json"
                            sc_flush = self.training_cache.get(cache_ck, [])
                            save_count = min(5000, len(sc_flush), self.max_training_cache_size)
                            with latest_path.open("w") as f:
                                json.dump(sc_flush[-save_count:], f)
                        except (OSError, FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
                            logger.debug("Error persisting training data (day flush): %s", e)
                    continue

                # ---- non-day (legacy v3 primary) path — live strategy universe is DAY-only today ----
                psec = primary_bar_seconds_for_strategy(strat)
                min_1m = min_ohlcv_bars_for_signal()
                lim_1m = ohlcv_1m_fetch_limit_for_primary(strat)
                ohlcv_1m_nd: list[list] | None = None
                ohlcv_1d = None
                if live_market_data_service:
                    try:
                        ohlcv_1m_nd = await live_market_data_service.get_ohlcv(ccxt_symbol, "1m", lim_1m)
                        d1 = await live_market_data_service.get_ohlcv(ccxt_symbol, "1d", 40)
                        if isinstance(d1, list) and len(d1) >= 2:
                            ohlcv_1d = d1
                    except Exception as e:
                        logger.debug("Failed to get OHLCV for %s: %s", ccxt_symbol, e)
                        ohlcv_1m_nd = None

                if not ohlcv_1m_nd or len(ohlcv_1m_nd) < min_1m:
                    logger.debug(
                        "No sufficient 1m OHLCV for %s strat=%s (len=%s need=%s), skipping",
                        current_coin,
                        strat,
                        len(ohlcv_1m_nd or []),
                        min_1m,
                    )
                    continue

                ohlcv_primary = resample_ohlcv_to_seconds(ohlcv_1m_nd, psec)
                min_p = min_primary_bars_for_strategy(strat)
                if len(ohlcv_primary) < min_p:
                    logger.debug(
                        "Insufficient primary bars %s strat=%s psec=%s bars=%s need=%s",
                        current_coin,
                        strat,
                        psec,
                        len(ohlcv_primary),
                        min_p,
                    )
                    continue

                last_ts_ms_nd = int(ohlcv_primary[-1][0])
                bucket_nd = (last_ts_ms_nd // 1000 // psec) * psec
                if self._last_primary_emit_bucket.get(cache_ck) == bucket_nd:
                    continue
                self._last_primary_emit_bucket[cache_ck] = bucket_nd

                try:
                    redis_client_nd = get_shared_redis_async()
                    sentiment_payload = await merge_canonical_sentiment_payload(
                        base_symbol=CanonicalSymbolFormatter.to_base(training_pair),
                        pair_symbol=training_pair,
                        ctx_for_overlay=ai_ctx_dict if isinstance(ai_ctx_dict, dict) else None,
                        redis_client=redis_client_nd,
                        ohlcv_1m=ohlcv_primary,
                        existing=sentiment_payload,
                    )
                except Exception as fe:
                    logger.debug("merge_canonical_sentiment_payload (training) skipped: %s", fe)

                exec_tail = ohlcv_1m_nd[-150:] if len(ohlcv_1m_nd) > 20 else ohlcv_1m_nd
                features = build_feature_vector_v3(
                    symbol_ccxt=ccxt_symbol,
                    ohlcv_primary=ohlcv_primary,
                    volume_profile=vp,
                    orderbook=trade_data,
                    ohlcv_1d=ohlcv_1d,
                    sentiment=sentiment_payload,
                    ai_context=ai_ctx_dict,
                    ohlcv_exec_1m=exec_tail,
                )

                try:
                    if len(features) == _FEATURE_DIM_V2 and _TARGET_FEATURE_VERSION >= 3:
                        await asyncio.to_thread(
                            lambda tp=training_pair, feat=list(features): persist_ai_feature_sample_row(
                                symbol=tp,
                                context_key=CANONICAL_TELEMETRY_CONTEXT_KEY_V3,
                                features=feat,
                                feature_version=int(_TARGET_FEATURE_VERSION),
                            ),
                        )
                except (TypeError, ValueError, RuntimeError) as pe:
                    logger.debug("persist_ai_feature_sample_row skipped: %s", pe)

                logger.debug(
                    "Collected v3 training row %s strat=%s primary_sec=%s features=%d ctx=%s",
                    current_coin,
                    strat,
                    psec,
                    len(features),
                    bool(ai_ctx_dict),
                )

                timestamp = _now_iso()
                feature_data = {
                    "timestamp": timestamp,
                    "features": features,
                    "feature_count": len(features),
                    "feature_version": int(_TARGET_FEATURE_VERSION),
                    "collection_time": timestamp,
                    "symbol": training_pair,
                    "live_strategy_id": strat,
                    "primary_bar_seconds": int(psec),
                    "ai_clock_contract": "v3",
                }

                symbol_cache_nd = self.training_cache.setdefault(cache_ck, [])
                symbol_cache_nd.append(feature_data)
                if len(symbol_cache_nd) > self.max_training_cache_size:
                    self.training_cache[cache_ck] = symbol_cache_nd[-self.max_training_cache_size :]

                self.last_collection[cache_ck] = timestamp

                if len(symbol_cache_nd) % 100 == 0:
                    try:
                        Path(self.training_data_dir).mkdir(parents=True, exist_ok=True)
                        safe_bus = training_pair.replace("/", "_")
                        latest_path = Path(self.training_data_dir) / f"{safe_bus}_{strat}_latest.json"
                        save_count = min(5000, len(symbol_cache_nd), self.max_training_cache_size)
                        with latest_path.open("w") as f:
                            json.dump(symbol_cache_nd[-save_count:], f)
                        if len(symbol_cache_nd) % 1000 == 0:
                            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                            archive_path = Path(self.training_data_dir) / f"{safe_bus}_{strat}_{timestamp_str}.json"
                            with archive_path.open("w") as f:
                                json.dump(symbol_cache_nd[-1000:], f)
                    except (OSError, FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
                        logger.debug("Error persisting training data: %s", e)

                self.log_training_progress(cache_ck, features)
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
            logger.exception("Error in collect_training_data: %s", e)
            self._last_collect_new_anchors = 0
            return []
        else:
            self._last_collect_new_anchors = total_new_anchors
            return features

    def log_training_progress(self, symbol: str, features: list[float]) -> None:
        """Log training progress with automatic history cleanup"""
        try:
            # Only log periodically to avoid flooding
            should_log = len(self.training_cache.get(symbol, [])) % 100 == 0
            if should_log:
                feature_count = len(features)
                # VECTORIZED total samples calculation for performance
                total_samples = sum(len(samples) for samples in self.training_cache.values())
                logger.info(f"Training data collection: {symbol} - {feature_count} features, {total_samples} total samples")

            # 24/7 FIX: Trim training history to prevent unbounded memory growth
            if len(self.training_history) > self.max_training_history:
                self.training_history = self.training_history[-self.max_training_history :]

        except (ValueError, TypeError, AttributeError, KeyError) as e:
            # Non-fatal error, just log it
            logger.debug(f"Error logging training progress: {e}")


# AI training pipeline state - using dict to avoid global keyword
_ai_training_pipeline_state: dict[str, AITrainingDataPipeline | None] = {"instance": None}


def get_ai_training_pipeline(cache: Any = None) -> AITrainingDataPipeline | None:
    """
    Get or create the global AI training pipeline instance.

    Args:
        cache: Optional cache instance. If None, uses canonical_cache.

    Returns:
        AITrainingDataPipeline instance or None if initialization fails.
    """
    try:
        if _ai_training_pipeline_state["instance"] is None:
            _ai_training_pipeline_state["instance"] = AITrainingDataPipeline(cache=cache)
        return _ai_training_pipeline_state["instance"]
    except Exception as e:
        logger.warning(f"Failed to get AI training pipeline: {e}")
        return None


def get_training_stats() -> dict[str, Any]:
    """
    Get training statistics from the AI training pipeline.

    Returns:
        Dictionary containing training statistics.
    """
    try:
        pipeline = get_ai_training_pipeline()
        if pipeline is None:
            return {
                "status": "unavailable",
                "error": "AI training pipeline not available",
                "timestamp": _now_iso(),
            }

        # Get training status from pipeline
        status = pipeline.get_training_status() if hasattr(pipeline, "get_training_status") else {}

        return {
            "status": "ok",
            "pipeline_running": getattr(pipeline, "is_running", False),
            "current_accuracy": getattr(pipeline, "_current_accuracy", 0.0),
            "training_status": status,
            "timestamp": _now_iso(),
        }
    except Exception as e:
        logger.warning(f"Failed to get training stats: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": _now_iso(),
        }


# Auto-start pipeline when module is imported (opt-out via env)
# DISABLED: Now started explicitly in app_factory.py lifespan for proper event loop
try:
    if os.getenv("AUTO_START_AI_TRAINING", "0") == "1":
        import asyncio

        _pipeline = get_ai_training_pipeline()
        if _pipeline is not None and not getattr(_pipeline, "is_running", False):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_pipeline.start())
            else:
                loop.run_until_complete(_pipeline.start())
except Exception as _auto_exc:  # pragma: no cover - defensive autostart guard
    logger.debug(f"Auto-start of AI training pipeline skipped: {_auto_exc}")
