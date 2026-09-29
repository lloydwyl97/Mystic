"""Engine / setup / dust provenance for learning rows.

One definition shared by the live close writer and the offline backfill so
both label a trade the same way:

- engine_id: the lot's owner (DAY_V2, SCALP_V2, or LEGACY_DAY_LIVE when unset).
- strategy: "scalp" for SCALP_V2, otherwise "day". Never "day" for SCALP.
- setup: the entry setup, or "UNKNOWN" when no source records it.
- is_dust: a dust close (DUST_WRITEOFF, or a lot that was already DUST_PENDING
  before the sell began; a residual left by a real sell does not count).
  Dust is an inventory artefact, not a strategy outcome, so label_strategy
  becomes "dust" and strategy learners (strategy_id='day'/'scalp') skip it.
"""

from __future__ import annotations

import contextlib
import sqlite3
from typing import Any

SCALP_ENGINE = "SCALP_V2"
DAY_ENGINE = "DAY_V2"
LEGACY_ENGINE = "LEGACY_DAY_LIVE"
UNKNOWN_SETUP = "UNKNOWN"
DUST_LABEL = "dust"
DAY_LEARNING_ENGINES = frozenset({DAY_ENGINE, LEGACY_ENGINE})
_NO_STRATEGY_LEARNING = frozenset({"MANUAL_UNMATCHED", "HUMAN_MANUAL_SELL", "DUST_WRITEOFF"})


def day_strategy_learning_allowed(engine_id: str, close_reason: str | None, *, is_dust: bool = False) -> bool:
    """DAY bandit, DAY attribution, and DAY setup memory.

    SCALP_V2 keeps its own learning rows and never updates those DAY structures.
    Dust, unmatched closes, and accounting corrections update none of them.
    """
    if is_dust:
        return False
    reason = str(close_reason or "").strip().upper()
    if reason in _NO_STRATEGY_LEARNING or reason.startswith("FALSE_"):
        return False
    return str(engine_id or "").strip().upper() in DAY_LEARNING_ENGINES


def engine_of(position: Any) -> str:
    return str(getattr(position, "engine_id", "") or "").strip().upper() or LEGACY_ENGINE


def strategy_for_engine(engine_id: str) -> str:
    return "scalp" if str(engine_id or "").upper() == SCALP_ENGINE else "day"


def is_dust_close(close_reason: str | None, position_status: str | None = None) -> bool:
    return str(close_reason or "").upper() == "DUST_WRITEOFF" or str(position_status or "").upper() == "DUST_PENDING"


def capture_close_provenance(position: Any, *, exit_trigger: str, sell_qty: float) -> dict[str, Any]:
    """Snapshot taken before the sell path mutates the lot.

    The sell leaves a residual marked DUST_PENDING; reading status afterwards
    would label a real strategy exit as a dust event.
    """
    engine_id = engine_of(position)
    status = str(getattr(position, "status", "") or "").upper()
    return {
        "engine_id": engine_id,
        "strategy": strategy_for_engine(engine_id),
        "original_trade_id": str(getattr(position, "trade_id", "") or ""),
        "exit_trigger": str(exit_trigger or ""),
        "pre_close_status": status,
        "pre_close_qty": float(getattr(position, "quantity", 0.0) or 0.0),
        "sell_qty": float(sell_qty or 0.0),
        "residual_qty": None,
        "is_strategy_close": status != "DUST_PENDING",
    }


def close_provenance_of(position: Any) -> dict[str, Any] | None:
    prov = getattr(position, "_close_provenance", None)
    if not isinstance(prov, dict):
        return None
    if prov.get("original_trade_id") != str(getattr(position, "trade_id", "") or ""):
        return None
    return prov


def close_is_dust(position: Any, close_reason: str | None) -> bool:
    if str(close_reason or "").upper() == "DUST_WRITEOFF":
        return True
    prov = close_provenance_of(position)
    if prov is not None:
        return not bool(prov.get("is_strategy_close"))
    return is_dust_close(close_reason, getattr(position, "status", ""))


def _lookup_setup(db_path: str, engine_id: str, *, trade_id: str, opportunity_id: str) -> str:
    queries: list[tuple[str, tuple]] = []
    if engine_id == SCALP_ENGINE and opportunity_id:
        queries.append(("SELECT setup_family FROM scalp_v2_opportunities WHERE opportunity_id=? ORDER BY id DESC LIMIT 1", (opportunity_id,)))
    if engine_id != SCALP_ENGINE and trade_id:
        queries.append(("SELECT setup FROM day_v2_opportunity_state WHERE trade_id=? ORDER BY id DESC LIMIT 1", (trade_id,)))
    if not queries:
        return ""
    with contextlib.suppress(Exception), sqlite3.connect(db_path, timeout=5) as conn:
        for sql, params in queries:
            with contextlib.suppress(sqlite3.OperationalError):
                row = conn.execute(sql, params).fetchone()
                if row and str(row[0] or "").strip():
                    return str(row[0]).strip()
    return ""


def resolve_setup(db_path: str, position: Any, explain: dict | None = None) -> str:
    ex = explain or {}
    engine_id = engine_of(position)

    def _lookup() -> str:
        return _lookup_setup(
            db_path,
            engine_id,
            trade_id=str(getattr(position, "trade_id", "") or ""),
            opportunity_id=str(getattr(position, "scalp_opportunity_id", "") or ""),
        )

    own = str(getattr(position, "entry_thesis", "") or "").strip()
    explained = next((str(v).strip() for v in (ex.get("setup_type_canonical"), ex.get("setup_type"), ex.get("setup_name")) if str(v or "").strip()), "")
    # SCALP owns its setup record; DAY explainability is not a SCALP source.
    chain = (own, _lookup, explained) if engine_id == SCALP_ENGINE else (explained, own, _lookup)
    for item in chain:
        value = item() if callable(item) else item
        if value:
            return value
    return UNKNOWN_SETUP


def learning_provenance(db_path: str, position: Any, close_reason: str | None, explain: dict | None = None) -> dict[str, Any]:
    engine_id = engine_of(position)
    strategy = strategy_for_engine(engine_id)
    dust = close_is_dust(position, close_reason)
    out: dict[str, Any] = {
        "engine_id": engine_id,
        "trade_id": str(getattr(position, "trade_id", "") or ""),
        "strategy": strategy,
        "setup": resolve_setup(db_path, position, explain),
        "is_dust": dust,
        "label_strategy": DUST_LABEL if dust else strategy,
    }
    prov = close_provenance_of(position)
    if prov is not None:
        residual = prov.get("residual_qty")
        if residual is None:
            residual = float(getattr(position, "quantity", 0.0) or 0.0)
        out.update(
            {
                "original_trade_id": prov.get("original_trade_id"),
                "exit_trigger": prov.get("exit_trigger"),
                "pre_close_status": prov.get("pre_close_status"),
                "sell_qty": prov.get("sell_qty"),
                "residual_qty": float(residual),
                "is_strategy_close": bool(prov.get("is_strategy_close")) and not dust,
            }
        )
    return out
