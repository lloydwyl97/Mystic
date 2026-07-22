"""Evaluate stale DAY models (BTC/SOL/XRP) and promote only when validation + profit improve."""

from __future__ import annotations

import logging
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_market_diagnostics import MODEL_STALE_HOURS
from backend.services.ai_model_promotion import register_candidate_and_maybe_promote
from backend.services.ai_model_promotion_holdout import build_holdout_validation_metrics
from backend.services.live_strategy_contracts import per_coin_artifact_file

logger = logging.getLogger(__name__)

STALE_EVAL_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
STRATEGY_ID = "day"


def _artifact_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            art = pickle.load(f)
        trained_at = str(art.get("trained_at") or "")
        if not trained_at:
            return None
        dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return None


def _latest_candidate_path(symbol: str) -> Path | None:
    version_dir = Path("models/versions/per_coin")
    if not version_dir.exists():
        return None
    pattern = sorted(version_dir.glob(f"{STRATEGY_ID}_{symbol}_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pattern[0] if pattern else None


def _avg_outcome_profit_pct(strategy_id: str, symbol: str, *, limit: int = 40, db_path: str = DATABASE_PATH) -> float | None:
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT net_pnl_pct FROM ai_outcome_training_rows
                WHERE strategy_id=? AND UPPER(symbol)=UPPER(?)
                ORDER BY id DESC LIMIT ?
                """,
                (strategy_id, symbol, int(limit)),
            ).fetchall()
        if len(rows) < 5:
            return None
        vals = [float(r[0] or 0.0) for r in rows]
        return sum(vals) / len(vals)
    except Exception:
        return None


def evaluate_stale_symbol(
    symbol: str,
    *,
    strategy_id: str = STRATEGY_ID,
    db_path: str = DATABASE_PATH,
    promote: bool = True,
) -> dict[str, Any]:
    """Compare latest candidate to active model; optionally promote via gate."""
    sym = symbol.strip().upper()
    active = per_coin_artifact_file(Path("models/active"), strategy_id, sym)
    age_h = _artifact_age_hours(active)
    stale = age_h is None or age_h > MODEL_STALE_HOURS
    candidate = _latest_candidate_path(sym)
    result: dict[str, Any] = {
        "symbol": sym,
        "strategy_id": strategy_id,
        "active_path": str(active),
        "active_exists": active.exists(),
        "active_age_hours": round(age_h, 1) if age_h is not None else None,
        "stale": stale,
        "candidate_path": str(candidate) if candidate else None,
        "promoted": False,
        "promotion_reason": "not_evaluated",
    }
    if not stale:
        result["promotion_reason"] = "active_not_stale"
        return result
    if candidate is None or not candidate.exists():
        result["promotion_reason"] = "no_candidate_available"
        return result

    try:
        with candidate.open("rb") as f:
            cand_art = pickle.load(f)
        cand_acc = float(cand_art.get("accuracy") or 0.0)
    except Exception as exc:
        result["promotion_reason"] = f"candidate_read_error:{exc}"
        return result

    active_acc = None
    if active.exists():
        try:
            with active.open("rb") as f:
                active_art = pickle.load(f)
            active_acc = float(active_art.get("accuracy") or 0.0)
        except Exception:
            pass

    profit_proxy = _avg_outcome_profit_pct(strategy_id, sym, db_path=db_path)
    try:
        validation_metrics = build_holdout_validation_metrics(
            strategy_id=strategy_id,
            symbol_bus=sym,
            candidate_path=candidate,
            active_path=active if active.exists() else None,
            db_path=db_path,
        )
    except Exception as _hv_exc:
        logger.warning("STALE_EVAL: holdout metrics unavailable for %s: %s", sym, _hv_exc)
        validation_metrics = {}
    validation_metrics.setdefault("holdout_status", "STALE_EVAL_BYPASS")
    validation_metrics.update({
        "accuracy": cand_acc,
        "active_accuracy": active_acc,
        "profit_after_cost": profit_proxy,
        "avg_net_pnl_pct": profit_proxy,
        "evaluation": "stale_model_workflow",
    })
    result["validation_metrics"] = validation_metrics

    if not promote:
        result["promotion_reason"] = "dry_run_only"
        return result

    promoted, reason = register_candidate_and_maybe_promote(
        strategy_id=strategy_id,
        symbol=sym,
        candidate_path=candidate,
        active_path=active,
        validation_metrics=validation_metrics,
        db_path=db_path,
    )
    result["promoted"] = promoted
    result["promotion_reason"] = reason
    if not promoted:
        logger.info("STALE_MODEL_REJECTED: %s reason=%s metrics=%s", sym, reason, validation_metrics)
    else:
        logger.info("STALE_MODEL_PROMOTED: %s reason=%s", sym, reason)
    return result


def run_stale_model_workflow(
    symbols: tuple[str, ...] = STALE_EVAL_SYMBOLS,
    *,
    db_path: str = DATABASE_PATH,
    promote: bool = True,
) -> dict[str, Any]:
    ensure_ai_canonical_tables(db_path)
    outcomes = [evaluate_stale_symbol(sym, db_path=db_path, promote=promote) for sym in symbols]
    return {
        "strategy_id": STRATEGY_ID,
        "symbols_evaluated": list(symbols),
        "results": outcomes,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["STALE_EVAL_SYMBOLS", "evaluate_stale_symbol", "run_stale_model_workflow"]
