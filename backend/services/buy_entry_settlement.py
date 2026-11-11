"""A flat position is not a finished entry while its buy can still reveal base.

Cash, fees, and the sell fill are booked when the sell happens. The one
terminal learning label for the trade id waits until the buy is terminal,
its fee is proven, and no canonical base remains.
"""

from __future__ import annotations

import json
import time
from typing import Any

_UNSETTLED = {"PENDING", "ACKED", "OPEN", "PARTIAL", "WORKING", "FEE_PENDING"}
_LEARN_KEYS = (
    "engine",
    "symbol",
    "setup",
    "regime",
    "strategy_version",
    "net_pct",
    "mfe_pct",
    "mae_pct",
    "hold_min",
    "continuation",
    "version_current",
    "is_dust",
    "entered_at",
    "unrealized_marks",
    "opportunity_id",
    "exit_reason",
    "candidate_id",
    "close_key",
)


def _num(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def entry_is_settled(db_path: str, trade_id: str) -> bool:
    """True when this trade has no buy that can still fill or reveal base.

    A sell with no strategy buy intent is already settled. Legacy and paper
    closes keep their current learning path.
    """
    if not str(trade_id or "") or not db_path:
        return True
    from backend.services.strategy_order_intent import intent_by_trade

    intent = intent_by_trade(db_path, str(trade_id), "BUY")
    if intent is None:
        return True
    return str(intent.get("status") or "") not in _UNSETTLED


def unsettled_buy_blocks_engine(db_path: str, symbol: str, engine_id: str) -> bool:
    """The same engine must not open a second entry while this one is unfinished."""
    if not db_path or not symbol or not engine_id:
        return False
    from backend.services.strategy_order_intent import _connect
    from backend.utils.symbols import normalize_symbol

    want = normalize_symbol(symbol)
    engine = str(engine_id or "")
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT symbol, engine_id, status FROM strategy_order_intents
            WHERE side='BUY' AND status IN ('PENDING', 'ACKED', 'OPEN', 'PARTIAL', 'WORKING', 'FEE_PENDING')
            """
        ).fetchall()
    finally:
        conn.close()
    return any(normalize_symbol(str(row["symbol"] or "")) == want and str(row["engine_id"] or "") == engine for row in rows)


def canonical_booked_qty(db_path: str, trade_id: str) -> float:
    """Net base already applied to this buy. Not the venue's gross executedQty."""
    from backend.services.strategy_order_intent import _connect

    conn = _connect(db_path)
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "paper_trades" in tables:
            paper = conn.execute(
                "SELECT quantity FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' LIMIT 1",
                (str(trade_id),),
            ).fetchone()
            if paper is not None:
                return _num(paper[0])
        intent = conn.execute(
            "SELECT filled_qty FROM strategy_order_intents WHERE trade_id=? AND side='BUY' ORDER BY created_at DESC LIMIT 1",
            (str(trade_id),),
        ).fetchone()
    finally:
        conn.close()
    return _num(intent[0]) if intent is not None else 0.0


def open_position_qty(db_path: str, trade_id: str) -> float:
    from backend.services.strategy_order_intent import _connect

    conn = _connect(db_path)
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "portfolio_engine_positions" not in tables:
            return 0.0
        row = conn.execute(
            "SELECT quantity FROM portfolio_engine_positions WHERE trade_id=? AND IFNULL(quantity, 0) > 0 LIMIT 1",
            (str(trade_id),),
        ).fetchone()
    except Exception:
        return 0.0
    finally:
        conn.close()
    return _num(row[0]) if row is not None else 0.0


def flat_buy_needs_recovery(intent: dict[str, Any], position_qty: float) -> bool:
    """A sold lot whose buy can still add or reveal base is not left to the two-minute pass alone."""
    if _num(position_qty) > 1e-8:
        return False
    status = str(intent.get("status") or "")
    filled = _num(intent.get("filled_qty"))
    venue = _num(intent.get("venue_executed_qty"))
    order_qty = _num(intent.get("quantity"))
    if filled <= 0 and venue <= 0:
        return False
    if status in {"PARTIAL", "WORKING"} and order_qty > filled + 1e-8:
        return True
    booked = _num(intent.get("booked_qty"))
    if booked <= 0:
        booked = filled
    return status == "FEE_PENDING" and venue > booked + 1e-8


def remember_unsettled_learn(db_path: str, trade_id: str, learn: dict[str, Any], facts: dict[str, Any]) -> None:
    """Keep the final label until the buy can no longer change the trade."""
    if not trade_id:
        return
    from backend.services.strategy_order_intent import _connect

    payload = {"learn": {key: learn.get(key) for key in _LEARN_KEYS}, "facts": facts}
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT deferred_close_json FROM strategy_order_intents WHERE trade_id=? AND side='BUY' ORDER BY created_at DESC LIMIT 1",
            (str(trade_id),),
        ).fetchone()
        if row is None:
            return
        try:
            current = json.loads(row[0] or "[]")
        except json.JSONDecodeError:
            current = []
        if not isinstance(current, list):
            current = []
        current.append(payload)
        conn.execute(
            "UPDATE strategy_order_intents SET deferred_close_json=?, updated_at=? WHERE trade_id=? AND side='BUY'",
            (json.dumps(current, default=str), time.time(), str(trade_id)),
        )
        conn.commit()
    finally:
        conn.close()


def peek_unsettled_learns(db_path: str, trade_id: str) -> list[dict[str, Any]]:
    from backend.services.strategy_order_intent import _connect

    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT deferred_close_json FROM strategy_order_intents WHERE trade_id=? AND side='BUY' ORDER BY created_at DESC LIMIT 1",
            (str(trade_id),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return []
    try:
        current = json.loads(row[0] or "[]")
    except json.JSONDecodeError:
        return []
    return [item for item in current if isinstance(item, dict)] if isinstance(current, list) else []


def clear_unsettled_learns(db_path: str, trade_id: str) -> None:
    from backend.services.strategy_order_intent import _connect

    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE strategy_order_intents SET deferred_close_json='[]', updated_at=? WHERE trade_id=? AND side='BUY'",
            (time.time(), str(trade_id)),
        )
        conn.commit()
    finally:
        conn.close()


def deferred_trade_ids(db_path: str) -> list[str]:
    from backend.services.strategy_order_intent import _connect

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT trade_id FROM strategy_order_intents
            WHERE side='BUY' AND COALESCE(deferred_close_json, '[]') NOT IN ('', '[]')
            """
        ).fetchall()
    except Exception:
        return []
    finally:
        conn.close()
    return [str(row[0]) for row in rows if row[0]]


def merge_terminal_learn(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """One market label for every sell chunk of the same entry."""
    learns = [dict(item.get("learn") or {}) for item in chunks if isinstance(item.get("learn"), dict)]
    facts = [dict(item.get("facts") or {}) for item in chunks]
    if not learns:
        return {}
    out = dict(learns[0])
    pnl = sum(_num(item.get("net_profit_usd")) for item in facts)
    cost = sum(_num(item.get("entry_price")) * _num(item.get("quantity")) for item in facts)
    if cost > 0:
        out["net_pct"] = pnl / cost
    entry = _num(facts[0].get("entry_price"))
    highs = [_num(item.get("highest_price")) for item in facts if _num(item.get("highest_price")) > 0]
    lows = [_num(item.get("lowest_price")) for item in facts if _num(item.get("lowest_price")) > 0]
    if entry > 0 and highs:
        out["mfe_pct"] = (max(highs) - entry) / entry
    if entry > 0 and lows:
        out["mae_pct"] = (entry - min(lows)) / entry
    start = _num(facts[0].get("entry_timestamp") or learns[0].get("entered_at"))
    end = max((_num(item.get("exit_timestamp")) for item in facts), default=0.0)
    if start > 0 and end > start:
        out["hold_min"] = (end - start) / 60.0
        out["entered_at"] = start
    out["exit_reason"] = learns[-1].get("exit_reason") or out.get("exit_reason")
    out["close_key"] = learns[-1].get("close_key") or out.get("close_key")
    return {key: out.get(key) for key in _LEARN_KEYS}


def completed_trade_row(extra: object) -> bool:
    """True when a learning row is a finished entry, not an interim sell chunk.

    Rows written before this field existed have neither key. They stay in
    historical reports. An explicit ``entry_settled`` false, or
    ``strategy_terminal`` false, is an execution record.
    """
    if not isinstance(extra, dict):
        return True
    if "entry_settled" not in extra and "strategy_terminal" not in extra:
        return True
    if extra.get("entry_settled") is False:
        return False
    if "strategy_terminal" in extra:
        return bool(extra.get("strategy_terminal"))
    return True


def completed_trade_predicate(column: str = "extra_json") -> str:
    """SQL for completed_trade_row. Invalid or pre-field JSON stays included."""
    return (
        f"({column} IS NULL OR TRIM(COALESCE({column}, '')) = '' OR json_valid({column}) = 0 "
        f"OR (COALESCE(json_extract({column}, '$.entry_settled'), 1) != 0 "
        f"AND COALESCE(json_extract({column}, '$.strategy_terminal'), 1) != 0))"
    )


def settlement_rows(db_path: str, *, now: float | None = None) -> list[dict[str, Any]]:
    """Buys that still own the lifecycle even when the local position is flat."""
    from backend.services.strategy_order_intent import _connect
    from backend.utils.symbols import normalize_symbol

    stamp = float(now if now is not None else time.time())
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT engine_id, symbol, trade_id, status, quantity, filled_qty, venue_executed_qty,
                   created_at, deferred_close_json
            FROM strategy_order_intents
            WHERE side='BUY' AND status IN ('PENDING', 'ACKED', 'OPEN', 'PARTIAL', 'WORKING', 'FEE_PENDING')
            """
        ).fetchall()
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        paper: dict[str, float] = {}
        positions: dict[str, float] = {}
        if "paper_trades" in tables:
            for item in conn.execute("SELECT trade_id, quantity FROM paper_trades WHERE UPPER(side)='BUY' AND trade_id != ''"):
                paper[str(item[0])] = _num(item[1])
        if "portfolio_engine_positions" in tables:
            for item in conn.execute("SELECT trade_id, quantity FROM portfolio_engine_positions WHERE IFNULL(quantity, 0) > 0 AND trade_id != ''"):
                positions[str(item[0])] = _num(item[1])
    except Exception:
        return []
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for row in rows:
        trade_id = str(row["trade_id"] or "")
        status = str(row["status"] or "")
        booked = paper.get(trade_id, _num(row["filled_qty"]))
        venue = _num(row["venue_executed_qty"])
        position_qty = positions.get(trade_id, 0.0)
        try:
            pending = json.loads(row["deferred_close_json"] or "[]")
        except json.JSONDecodeError:
            pending = []
        learning_pending = isinstance(pending, list) and len(pending) > 0
        created = _num(row["created_at"])
        out.append(
            {
                "engine": str(row["engine_id"] or ""),
                "symbol": normalize_symbol(str(row["symbol"] or "")),
                "trade_id": trade_id,
                "entry_settled": False,
                "buy_terminal": status == "FEE_PENDING",
                "buy_fee_complete": False,
                "executed_qty": venue,
                "canonical_booked_buy_qty": booked,
                "unbooked_executed_qty": max(0.0, venue - booked),
                "position_qty": position_qty,
                "trade_finalized": False,
                "terminal_learning_pending": learning_pending,
                "age_sec": None if created <= 0 else round(max(0.0, stamp - created), 3),
            }
        )
    return out
