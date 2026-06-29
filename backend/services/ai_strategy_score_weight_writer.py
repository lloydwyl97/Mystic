"""
Learn soft ranking weights from closed trades → ai_strategy_score_weights.

Ranking only: never blocks symbols or forces trades. Populates the table
consumed by portfolio_engine.compute_adaptive_score_delta().
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

logger = logging.getLogger(__name__)

# Must match portfolio_engine._ADAPTIVE_COMPONENT_BOUNDS keys and ranges.
COMPONENT_BOUNDS: dict[str, tuple[float, float]] = {
    "model_probability": (4.0, 80.0),
    "buy_margin": (20.0, 260.0),
    "relative_strength": (2.0, 30.0),
    "volume": (2.0, 24.0),
    "trend": (2.0, 26.0),
    "momentum": (2.0, 24.0),
    "spread_penalty": (-5000.0, -200.0),
    "slippage_penalty": (-5000.0, -200.0),
    "chop_penalty": (-40.0, -2.0),
    "memory_bonus": (1.0, 30.0),
    "memory_penalty": (-30.0, -1.0),
    "symbol_expectancy": (2.0, 80.0),
    "net_expected_value": (900.0, 5000.0),
}

PRIMARY_COMPONENTS: tuple[str, ...] = (
    "model_probability",
    "buy_margin",
    "relative_strength",
    "volume",
    "trend",
    "symbol_expectancy",
    "net_expected_value",
    "chop_penalty",
    "spread_penalty",
)

MIN_BUCKET_SAMPLES = 3
LEARN_ALPHA = 0.18
BACKFILL_LIMIT = 400
DB_BUSY_TIMEOUT_SEC = 30.0
DB_LOCK_RETRIES = 6


def _connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=DB_BUSY_TIMEOUT_SEC)
    conn.execute(f"PRAGMA busy_timeout={int(DB_BUSY_TIMEOUT_SEC * 1000)}")
    return conn


def _normalize_symbol_bus(sym: str) -> str:
    s = (sym or "").strip().upper().replace("/", "")
    return s


def _normalize_regime(value: Any) -> str:
    s = str(value or "unknown").strip().lower()
    return s or "unknown"


def _opt(d: dict[str, Any], key: str) -> float | None:
    raw = d.get(key)
    if raw in (None, ""):
        return None
    try:
        v = float(raw)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def extract_scaled_components(explain: dict[str, Any]) -> dict[str, float]:
    """Map buy explainability / decision_data into adaptive bound coordinates."""
    dd = dict(explain or {})
    out: dict[str, float] = {}
    bmin, bmax = COMPONENT_BOUNDS["model_probability"]
    conf = _opt(dd, "ai_confidence") or _opt(dd, "confidence") or _opt(dd, "winner_probability")
    if conf is not None:
        out["model_probability"] = bmin + max(0.0, min(1.0, conf)) * (bmax - bmin)

    bmin, bmax = COMPONENT_BOUNDS["buy_margin"]
    bm = _opt(dd, "entry_buy_margin") or _opt(dd, "buy_margin")
    if bm is not None:
        out["buy_margin"] = bmin + max(0.0, min(1.0, bm / 0.20)) * (bmax - bmin)

    bmin, bmax = COMPONENT_BOUNDS["relative_strength"]
    rs = _opt(dd, "signal_ctx_rs_btc") or _opt(dd, "ctx_rs_btc")
    if rs is not None:
        mid = (bmin + bmax) / 2.0
        span = (bmax - bmin) / 2.0
        out["relative_strength"] = mid + max(-1.0, min(1.0, rs / 5.0)) * span

    bmin, bmax = COMPONENT_BOUNDS["volume"]
    vol = _opt(dd, "relative_volume") or _opt(dd, "volume_expansion")
    if vol is not None:
        out["volume"] = bmin + max(0.0, min(1.0, vol / 3.0)) * (bmax - bmin)

    bmin, bmax = COMPONENT_BOUNDS["trend"]
    ts = _opt(dd, "trend_score")
    if ts is not None:
        out["trend"] = bmin + max(0.0, min(1.0, ts)) * (bmax - bmin)

    bmin, bmax = COMPONENT_BOUNDS["symbol_expectancy"]
    ce = _opt(dd, "coin_expectancy") or _opt(dd, "symbol_trust_expectancy")
    if ce is not None:
        mid = (bmin + bmax) / 2.0
        span = (bmax - bmin) / 2.0
        out["symbol_expectancy"] = mid + max(-1.0, min(1.0, ce)) * span

    bmin, bmax = COMPONENT_BOUNDS["net_expected_value"]
    ev = _opt(dd, "selected_net_expected_value") or _opt(dd, "net_expected_value")
    if ev is not None:
        out["net_expected_value"] = bmin + max(0.0, min(1.0, ev / 0.005)) * (bmax - bmin)

    bmin, bmax = COMPONENT_BOUNDS["chop_penalty"]
    cs = _opt(dd, "chop_score")
    if cs is not None:
        out["chop_penalty"] = bmax - max(0.0, min(1.0, cs)) * (bmax - bmin)

    bmin, bmax = COMPONENT_BOUNDS["spread_penalty"]
    sp = _opt(dd, "entry_spread_pct") or _opt(dd, "signal_spread_pct") or _opt(dd, "spread_pct")
    if sp is not None:
        out["spread_penalty"] = bmax - max(0.0, min(1.0, sp / 0.01)) * (bmax - bmin)

    fh = _opt(dd, "feature_health_score")
    if fh is not None:
        bmin, bmax = COMPONENT_BOUNDS["trend"]
        out["trend"] = out.get("trend", bmin + fh * (bmax - bmin))

    return out


def _regime_from_explain(explain: dict[str, Any]) -> str:
    regime = _normalize_regime(
        explain.get("day_route_regime")
        or explain.get("signal_regime_label")
        or explain.get("regime")
        or explain.get("price_structure_regime")
    )
    setup = str(explain.get("setup_type") or explain.get("entry_thesis") or "NO_CLEAR_THESIS").strip().upper()
    return setup_regime_bucket(regime, setup)


def setup_regime_bucket(regime: str, setup_thesis: str) -> str:
    """Composite learning bucket stored in ai_strategy_score_weights.regime."""
    r = _normalize_regime(regime)
    setup = str(setup_thesis or "NO_CLEAR_THESIS").strip().upper() or "NO_CLEAR_THESIS"
    return f"{r}::{setup}"


def _target_weight(
    component: str,
    win_vals: list[float],
    loss_vals: list[float],
) -> float:
    bounds = COMPONENT_BOUNDS.get(component)
    if not bounds:
        return 0.0
    bmin, bmax = bounds
    mid = (bmin + bmax) / 2.0
    if not win_vals and not loss_vals:
        return mid
    win_mean = sum(win_vals) / len(win_vals) if win_vals else mid
    loss_mean = sum(loss_vals) / len(loss_vals) if loss_vals else mid
    if win_mean > loss_mean + 1e-9:
        return mid + 0.32 * (bmax - mid)
    if win_mean < loss_mean - 1e-9:
        return mid - 0.32 * (mid - bmin)
    return mid


def _upsert_weight(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    symbol_bus: str,
    regime: str,
    component_name: str,
    target: float,
    sample_count: int,
    good_count: int,
    bad_count: int,
    net_expectancy: float,
) -> None:
    bounds = COMPONENT_BOUNDS.get(component_name)
    if not bounds:
        return
    bmin, bmax = bounds
    target = max(bmin, min(bmax, float(target)))
    now = datetime.now(timezone.utc).isoformat()
    row = conn.execute(
        """
        SELECT weight FROM ai_strategy_score_weights
        WHERE LOWER(strategy_id)=LOWER(?) AND UPPER(symbol)=UPPER(?) AND LOWER(regime)=LOWER(?) AND component_name=?
        """,
        (strategy_id, symbol_bus, regime, component_name),
    ).fetchone()
    if row:
        prev = float(row[0] or 0.0)
        weight = (1.0 - LEARN_ALPHA) * prev + LEARN_ALPHA * target
    else:
        weight = target
    weight = max(bmin, min(bmax, weight))
    conn.execute(
        """
        INSERT INTO ai_strategy_score_weights (
            strategy_id, symbol, regime, component_name, weight, previous_weight,
            sample_count, good_count, bad_count, net_expectancy, last_adjusted_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(strategy_id, symbol, regime, component_name) DO UPDATE SET
            previous_weight=ai_strategy_score_weights.weight,
            weight=excluded.weight,
            sample_count=excluded.sample_count,
            good_count=excluded.good_count,
            bad_count=excluded.bad_count,
            net_expectancy=excluded.net_expectancy,
            last_adjusted_at=excluded.last_adjusted_at,
            updated_at=excluded.updated_at
        """,
        (
            strategy_id,
            symbol_bus,
            regime,
            component_name,
            weight,
            float(row[0]) if row else 0.0,
            sample_count,
            good_count,
            bad_count,
            net_expectancy,
            now,
            now,
        ),
    )


def recompute_bucket_weights(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    symbol_bus: str,
    regime: str,
    samples: list[dict[str, Any]],
) -> int:
    """Recompute all primary component weights for one bucket. Returns rows touched."""
    if len(samples) < MIN_BUCKET_SAMPLES:
        return 0
    good = [s for s in samples if s.get("is_good")]
    bad = [s for s in samples if not s.get("is_good")]
    pnls = [float(s.get("net_pnl_pct") or 0.0) for s in samples]
    net_exp = sum(pnls) / len(pnls) if pnls else 0.0
    touched = 0
    for comp in PRIMARY_COMPONENTS:
        win_vals = [float(s["components"][comp]) for s in good if comp in s.get("components", {})]
        loss_vals = [float(s["components"][comp]) for s in bad if comp in s.get("components", {})]
        if not win_vals and not loss_vals:
            continue
        target = _target_weight(comp, win_vals, loss_vals)
        _upsert_weight(
            conn,
            strategy_id=strategy_id,
            symbol_bus=symbol_bus,
            regime=regime,
            component_name=comp,
            target=target,
            sample_count=len(samples),
            good_count=len(good),
            bad_count=len(bad),
            net_expectancy=net_exp,
        )
        touched += 1
    return touched


def _load_training_samples(
    conn: sqlite3.Connection,
    *,
    strategy_id: str,
    symbol_bus: str,
    limit: int = 80,
) -> list[dict[str, Any]]:
    """Load recent outcome rows for symbol/strategy (all regimes).

    Regime is only the write bucket key on close; filtering samples by regime
    starved bull buckets when historical rows were tagged bear.
    """
    sym_slash = f"{symbol_bus.replace('USDT', '')}/USDT" if symbol_bus.endswith("USDT") else symbol_bus
    rows = conn.execute(
        """
        SELECT good_bad_memory_class, net_pnl_pct, score_components_json, context_json
        FROM ai_outcome_training_rows
        WHERE (UPPER(REPLACE(symbol,'/',''))=UPPER(?) OR UPPER(symbol)=UPPER(?))
          AND LOWER(COALESCE(strategy_id,'day'))=LOWER(?)
        ORDER BY id DESC LIMIT ?
        """,
        (symbol_bus, sym_slash, strategy_id, limit),
    ).fetchall()
    samples: list[dict[str, Any]] = []
    for gb, net_pnl, sc_raw, ctx_raw in rows:
        explain: dict[str, Any] = {}
        try:
            if sc_raw:
                explain.update(json.loads(sc_raw) if isinstance(sc_raw, str) else dict(sc_raw))
        except Exception:
            pass
        try:
            if ctx_raw:
                ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else dict(ctx_raw)
                explain.setdefault("day_route_regime", ctx.get("market_regime") or ctx.get("regime"))
        except Exception:
            pass
        comps = extract_scaled_components(explain)
        if len(comps) < 2:
            continue
        is_good = str(gb or "").upper() == "GOOD"
        from backend.services.day_feature_health import entry_feature_health_pass

        if is_good and not entry_feature_health_pass(explain):
            continue
        samples.append({"is_good": is_good, "net_pnl_pct": net_pnl, "components": comps})
    return samples


def propagate_adaptive_score_weights_for_close(
    *,
    symbol: str,
    strategy_id: str = "day",
    explainability: dict[str, Any] | None = None,
    db_path: str = DATABASE_PATH,
) -> int:
    """After a trade close, refresh learned weights for symbol/strategy/regime."""
    ensure_ai_canonical_tables(db_path)
    sym_bus = _normalize_symbol_bus(symbol)
    sid = (strategy_id or "day").strip().lower()
    regime = _regime_from_explain(explainability or {})
    from backend.services.day_feature_health import entry_feature_health_pass

    ex = explainability or {}
    is_good = str(ex.get("good_bad_memory_class") or "").upper() == "GOOD"
    if is_good and not entry_feature_health_pass(ex):
        logger.info(
            "ADAPTIVE_WEIGHTS_SKIP good close with entry feature_health_pass=false symbol=%s regime=%s",
            sym_bus,
            regime,
        )
        return 0
    last_exc: Exception | None = None
    for attempt in range(DB_LOCK_RETRIES):
        try:
            with _connect_db(db_path) as conn:
                samples = _load_training_samples(conn, strategy_id=sid, symbol_bus=sym_bus)
                if len(samples) < MIN_BUCKET_SAMPLES:
                    return 0
                touched = recompute_bucket_weights(
                    conn, strategy_id=sid, symbol_bus=sym_bus, regime=regime, samples=samples
                )
                conn.commit()
                if touched:
                    logger.info(
                        "ADAPTIVE_WEIGHTS_UPDATED symbol=%s strategy=%s regime=%s components=%d samples=%d",
                        sym_bus,
                        sid,
                        regime,
                        touched,
                        len(samples),
                    )
                return touched
        except sqlite3.OperationalError as exc:
            last_exc = exc
            if "locked" in str(exc).lower() and attempt + 1 < DB_LOCK_RETRIES:
                time.sleep(0.05 * (2**attempt))
                continue
            logger.warning("propagate_adaptive_score_weights_for_close failed symbol=%s: %s", symbol, exc)
            return 0
        except Exception as exc:
            logger.warning("propagate_adaptive_score_weights_for_close failed symbol=%s: %s", symbol, exc)
            return 0
    if last_exc is not None:
        logger.warning("propagate_adaptive_score_weights_for_close failed symbol=%s: %s", symbol, last_exc)
    return 0


def backfill_adaptive_score_weights_from_outcomes(db_path: str = DATABASE_PATH, *, limit: int = BACKFILL_LIMIT) -> dict[str, Any]:
    """Backfill weights from paper_trades buy explainability joined to sells."""
    ensure_ai_canonical_tables(db_path)
    stats = {"pairs_processed": 0, "buckets_updated": 0, "weight_rows": 0}
    bucket_samples: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    with _connect_db(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sells = conn.execute(
            """
            SELECT s.id, s.symbol, s.pnl, s.pnl_pct, s.explainability_json, s.strategy_id,
                   (
                     SELECT b.explainability_json FROM paper_trades b
                     WHERE b.side='BUY' AND b.symbol=s.symbol AND b.id < s.id
                     ORDER BY b.id DESC LIMIT 1
                   ) AS buy_explain
            FROM paper_trades s
            WHERE s.side='SELL' AND COALESCE(s.is_synthetic,0)=0
            ORDER BY s.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()

        for sell in sells:
            buy_explain: dict[str, Any] = {}
            try:
                buy_explain = json.loads(sell["buy_explain"] or "{}")
            except Exception:
                buy_explain = {}
            if not buy_explain:
                try:
                    sell_ex = json.loads(sell["explainability_json"] or "{}")
                    buy_explain = sell_ex
                except Exception:
                    buy_explain = {}
            if not buy_explain:
                continue
            sym_bus = _normalize_symbol_bus(str(sell["symbol"]))
            sid = str(sell["strategy_id"] or buy_explain.get("live_ai_strategy") or "day").strip().lower()
            regime = _regime_from_explain(buy_explain)
            comps = extract_scaled_components(buy_explain)
            if len(comps) < 2:
                continue
            pnl = float(sell["pnl"] or 0.0)
            is_good = pnl >= 0.0
            key = (sid, sym_bus, regime)
            bucket_samples.setdefault(key, []).append({"is_good": is_good, "net_pnl_pct": float(sell["pnl_pct"] or 0.0), "components": comps})
            stats["pairs_processed"] += 1

        for (sid, sym_bus, regime), samples in bucket_samples.items():
            if len(samples) < MIN_BUCKET_SAMPLES:
                continue
            touched = recompute_bucket_weights(conn, strategy_id=sid, symbol_bus=sym_bus, regime=regime, samples=samples[-80:])
            if touched:
                stats["buckets_updated"] += 1
                stats["weight_rows"] += touched

        stats["weight_rows_total"] = conn.execute("SELECT COUNT(*) FROM ai_strategy_score_weights").fetchone()[0]
        conn.commit()

    return stats


def adaptive_weights_row_count(db_path: str = DATABASE_PATH) -> int:
    try:
        with _connect_db(db_path) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM ai_strategy_score_weights").fetchone()[0] or 0)
    except Exception:
        return 0
