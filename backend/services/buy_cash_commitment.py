"""Account-wide BUY quote that is still able to spend USDT.

Ledger cash is not spendable until this quote is subtracted. A pre-submit
reservation and the live order for the same engine and symbol are one hold:
the larger of the two, not the sum. A working order keeps its remaining
quantity at its limit, plus any executed quote that is not booked yet.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from backend.services.day_entry_spendable import money
from backend.utils.position_keys import POSITION_KEY_SEP

WORKING_BUY_STATUSES = ("PENDING", "ACKED", "OPEN", "PARTIAL", "WORKING")
UNBOOKED_BUY_STATUSES = ("FILLED", "FEE_PENDING")

# A protected-limit attempt returned a result that cannot still spend quote.
RELEASE_OUTCOMES = {
    "EXPIRED_UNFILLED": "EXPIRED",
    "TIMEOUT_CANCELED_UNFILLED": "CANCELED",
    "EXCHANGE_REJECTED": "REJECTED",
    "RATE_LIMITED": "NO_VENUE_ORDER",
}


def buy_failure_keeps_quote(outcome: str) -> bool:
    """Lost and unknown acknowledgements keep the quote. A proven dead order does not."""
    return str(outcome or "") not in RELEASE_OUTCOMES


def buy_failure_intent_status(outcome: str) -> str:
    return RELEASE_OUTCOMES.get(str(outcome or ""), "")


def _symbol(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("-", "/")
    if "/" not in raw and raw.endswith("USDT") and len(raw) > 4:
        raw = raw[:-4] + "/USDT"
    return raw


def _engine(sleeve: str = "", engine_id: str = "") -> str:
    if str(engine_id or "") == "SCALP_V2" or str(sleeve or "") == "SCALP_V2":
        return "SCALP_V2"
    return "DAY_V2"


def _split_key(key: str) -> tuple[str, str]:
    sleeve, sep, symbol = str(key or "").rpartition(POSITION_KEY_SEP)
    if not sep:
        return "", _symbol(key)
    return str(sleeve or ""), _symbol(symbol)


@dataclass
class _Hold:
    engine: str
    symbol: str
    decision_id: str
    reservation_id: str
    notional: Decimal
    created_at: float


@dataclass
class _IntentQuote:
    engine: str
    symbol: str
    decision_id: str
    client_order_id: str
    status: str
    filled_qty: Decimal
    remaining_quote: Decimal
    executed_quote: Decimal
    created_at: float
    trade_id: str = ""
    order_qty: Decimal = Decimal("0")

    @property
    def open_quote(self) -> Decimal:
        return self.remaining_quote + self.executed_quote


def intent_quote_parts(row: dict[str, Any]) -> tuple[Decimal, Decimal]:
    """Return (still-live limit quote, executed quote not yet booked).

    The live remainder is limit times unfilled quantity, so a partial order
    cannot spend more than its original limit. Executed quote uses the larger
    of the fill notional and the cash figure when the venue named a quote fee.
    A base-asset or other-asset fee is not turned into extra USDT.
    """
    status = str(row.get("status") or "")
    unbooked = status in UNBOOKED_BUY_STATUSES
    if status not in WORKING_BUY_STATUSES and not unbooked:
        return Decimal("0"), Decimal("0")
    if str(row.get("side") or "BUY").upper() != "BUY":
        return Decimal("0"), Decimal("0")
    limit_px = money(row.get("limit_price"))
    qty = money(row.get("quantity"))
    filled = money(row.get("filled_qty"))
    if limit_px <= 0 or qty <= 0:
        return Decimal("0"), Decimal("0")
    if filled < 0:
        filled = Decimal("0")
    filled = min(filled, qty)
    remaining = Decimal("0") if unbooked else qty - filled
    executed = money(row.get("filled_cost"))
    cash = money(row.get("filled_cash"))
    if executed <= 0 and filled > 0:
        executed = filled * limit_px
    executed = max(executed, cash)
    return remaining * limit_px, executed


def _connect(db_path: str) -> sqlite3.Connection | None:
    if not db_path or str(db_path) == ":memory:":
        return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
    except sqlite3.Error:
        return None
    conn.row_factory = sqlite3.Row
    return conn


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _load_reservations(db_path: str, *, exclude_reservation_id: str, now: float, working_decisions: set[str]) -> list[_Hold]:
    conn = _connect(db_path)
    if conn is None:
        return []
    try:
        if "day_entry_reservations" not in _tables(conn):
            return []
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(day_entry_reservations)")}
        exact = "notional_exact" if "notional_exact" in columns else "NULL"
        rows = conn.execute(
            f"""
            SELECT reservation_id, decision_id, symbol, notional_usd, {exact} AS notional_exact, sleeve,
                   created_at, expires_at
            FROM day_entry_reservations
            WHERE status='ACTIVE'
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out: list[_Hold] = []
    skip = str(exclude_reservation_id or "")
    for row in rows:
        rid = str(row["reservation_id"] or "")
        if skip and rid == skip:
            continue
        decision = str(row["decision_id"] or "")
        try:
            expires = float(row["expires_at"] or 0)
        except (TypeError, ValueError):
            expires = 0.0
        if expires and expires <= now and decision not in working_decisions:
            continue
        notional = money(row["notional_exact"] or row["notional_usd"] or 0)
        if notional <= 0:
            continue
        try:
            created = float(row["created_at"] or now)
        except (TypeError, ValueError):
            created = now
        out.append(
            _Hold(
                engine=_engine(str(row["sleeve"] or "")),
                symbol=_symbol(str(row["symbol"] or "")),
                decision_id=decision,
                reservation_id=rid,
                notional=notional,
                created_at=created,
            )
        )
    return out


def _load_intents(db_path: str) -> list[_IntentQuote]:
    conn = _connect(db_path)
    if conn is None:
        return []
    try:
        if "strategy_order_intents" not in _tables(conn):
            return []
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(strategy_order_intents)")}
        cash = "filled_cash" if "filled_cash" in columns else "0"
        cost = "filled_cost" if "filled_cost" in columns else "0"
        rows = conn.execute(
            f"""
            SELECT client_order_id, engine_id, decision_id, symbol, side, status,
                   quantity, limit_price, filled_qty, {cost} AS filled_cost, {cash} AS filled_cash, created_at,
                   trade_id
            FROM strategy_order_intents
            WHERE side='BUY' AND (
                status IN ({", ".join("?" for _ in WORKING_BUY_STATUSES)})
                OR (status IN ('FILLED', 'FEE_PENDING') AND filled_qty > 0)
            )
            """,
            WORKING_BUY_STATUSES,
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out: list[_IntentQuote] = []
    for row in rows:
        remaining, executed = intent_quote_parts(dict(row))
        if remaining + executed <= 0:
            continue
        try:
            created = float(row["created_at"] or 0)
        except (TypeError, ValueError):
            created = 0.0
        out.append(
            _IntentQuote(
                engine=_engine("", str(row["engine_id"] or "")),
                symbol=_symbol(str(row["symbol"] or "")),
                decision_id=str(row["decision_id"] or ""),
                client_order_id=str(row["client_order_id"] or ""),
                status=str(row["status"] or ""),
                filled_qty=money(row["filled_qty"]),
                remaining_quote=remaining,
                executed_quote=executed,
                created_at=created,
                trade_id=str(row["trade_id"] or ""),
                order_qty=money(row["quantity"]),
            )
        )
    return out


def _memory_holds(memory_reservations: dict[str, Any] | None) -> list[_Hold]:
    out: list[_Hold] = []
    now = time.time()
    for key, row in (memory_reservations or {}).items():
        sleeve, symbol = _split_key(str(key))
        data = row or {}
        sleeve = str(data.get("sleeve") or sleeve)
        notional = money(data.get("notional") or 0)
        if notional <= 0 or not symbol:
            continue
        try:
            created = float(data.get("ts") or now)
        except (TypeError, ValueError):
            created = now
        out.append(
            _Hold(
                engine=_engine(sleeve, str(data.get("engine_id") or "")),
                symbol=symbol,
                decision_id=str(data.get("decision_id") or ""),
                reservation_id=str(data.get("reservation_id") or ""),
                notional=notional,
                created_at=created,
            )
        )
    return out


def _extra_symbol_quote(pending_orders: dict[str, Any] | None, trailing_arms: dict[str, Any] | None, covered: set[str]) -> list[tuple[str, Decimal, float]]:
    """In-flight orders and armed watches that have no reservation or intent yet."""
    out: list[tuple[str, Decimal, float]] = []
    for order in (pending_orders or {}).values():
        if str(getattr(order, "side", "") or "").upper() != "BUY":
            continue
        symbol = _symbol(str(getattr(order, "symbol", "") or ""))
        if not symbol or symbol in covered:
            continue
        quote = money(getattr(order, "remaining_qty", 0) or 0) * money(getattr(order, "price", 0) or 0)
        if quote > 0:
            out.append((symbol, quote, time.time()))
    for symbol, arm in (trailing_arms or {}).items():
        norm = _symbol(str(symbol))
        if not norm or norm in covered:
            continue
        quote = money(getattr(arm, "quantity", 0) or 0) * money(getattr(arm, "last_ask", 0) or getattr(arm, "initial_ask", 0) or 0)
        if quote > 0:
            out.append((norm, quote, time.time()))
    return out


@dataclass
class BuyCashView:
    pending_buy_quote: Decimal
    working_buy_quote: Decimal
    executed_unbooked_buy_quote: Decimal
    committed_quote: Decimal
    reservation_count: int
    oldest_created_at: float
    working_buy_count: int = 0
    working_buys: list[dict[str, Any]] | None = None
    unresolved_buy_count: int = 0
    oldest_unresolved_buy_age_sec: float | None = None
    partial_buy_count: int = 0
    ambiguous_buy_count: int = 0
    oldest_ambiguous_buy_age_sec: float | None = None
    custody_pending_count: int = 0
    potential_unprotected_symbols: list[str] | None = None

    @property
    def oldest_age_sec(self) -> float | None:
        if self.oldest_created_at <= 0 or self.reservation_count <= 0:
            return None
        return max(0.0, time.time() - self.oldest_created_at)

    def available(self, ledger_cash: object) -> Decimal:
        free = money(ledger_cash) - self.committed_quote
        if free < 0:
            return Decimal("0")
        return free

    def as_dict(self, ledger_cash: object, *, cash_sync_held: bool) -> dict[str, Any]:
        def num(value: Decimal) -> float:
            return float(value)

        return {
            "ledger_cash": num(money(ledger_cash)),
            "pending_buy_quote": num(self.pending_buy_quote),
            "working_buy_quote": num(self.working_buy_quote),
            "executed_unbooked_buy_quote": num(self.executed_unbooked_buy_quote),
            "available_to_buy": num(self.available(ledger_cash)),
            "buy_cash_reservation_count": self.reservation_count,
            "oldest_buy_reservation_age_sec": None if self.oldest_age_sec is None else round(self.oldest_age_sec, 3),
            "cash_sync_held": bool(cash_sync_held),
            "working_buy_count": self.working_buy_count,
            "working_buys": list(self.working_buys or []),
            "unresolved_buy_count": self.unresolved_buy_count,
            "oldest_unresolved_buy_age_sec": self.oldest_unresolved_buy_age_sec,
            "partial_buy_count": self.partial_buy_count,
            "ambiguous_buy_count": self.ambiguous_buy_count,
            "oldest_ambiguous_buy_age_sec": self.oldest_ambiguous_buy_age_sec,
            "custody_pending_count": self.custody_pending_count,
            "potential_unprotected_symbols": list(self.potential_unprotected_symbols or []),
        }


def _local_buy_books(db_path: str) -> dict[str, tuple[float | None, float | None]]:
    """trade_id -> (paper BUY quantity, open position quantity).

    Paper quantity is the last cumulative fill booked onto that buy. The
    position quantity is what exit protection can see right now.
    """
    conn = _connect(db_path)
    if conn is None:
        return {}
    out: dict[str, tuple[float | None, float | None]] = {}
    try:
        tables = _tables(conn)
        if "paper_trades" in tables:
            for row in conn.execute("SELECT trade_id, quantity FROM paper_trades WHERE UPPER(side)='BUY' AND trade_id != ''"):
                out[str(row["trade_id"])] = (float(row["quantity"] or 0.0), None)
        if "portfolio_engine_positions" in tables:
            for row in conn.execute("SELECT trade_id, quantity FROM portfolio_engine_positions WHERE IFNULL(quantity, 0) > 0 AND trade_id != ''"):
                trade_id = str(row["trade_id"])
                paper_qty = out.get(trade_id, (None, None))[0]
                out[trade_id] = (paper_qty, float(row["quantity"] or 0.0))
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    return out


def buy_cash_view(
    db_path: str,
    *,
    memory_reservations: dict[str, Any] | None = None,
    pending_orders: dict[str, Any] | None = None,
    trailing_arms: dict[str, Any] | None = None,
    exclude_symbol: str = "",
    exclude_decision_id: str = "",
    exclude_engine: str = "",
    exclude_reservation_id: str = "",
    now: float | None = None,
) -> BuyCashView:
    """Quote a new buy must not spend.

    ``exclude_engine`` plus ``exclude_symbol`` drops this order's own pre-claim.
    Another engine on that symbol, and a different decision that is still
    working, stay in the hold.
    """
    stamp = float(now if now is not None else time.time())
    intents = _load_intents(db_path)
    working_decisions = {item.decision_id for item in intents if item.decision_id and item.remaining_quote > 0}
    reservations = _load_reservations(
        db_path,
        exclude_reservation_id=exclude_reservation_id,
        now=stamp,
        working_decisions=working_decisions,
    )
    reservations.extend(_memory_holds(memory_reservations))
    skip_decision = str(exclude_decision_id or "")
    skip_symbol = _symbol(exclude_symbol) if exclude_symbol else ""
    skip_engine = str(exclude_engine or "")

    groups: dict[tuple[str, str], dict[str, Any]] = {}

    def group(engine: str, symbol: str) -> dict[str, Any]:
        return groups.setdefault((engine, symbol), {"reservations": [], "intents": []})

    by_identity: dict[tuple[str, str, str], _Hold] = {}
    for hold in reservations:
        if skip_decision and hold.decision_id == skip_decision:
            continue
        identity = (hold.engine, hold.symbol, hold.decision_id or hold.reservation_id or f"anon-{id(hold)}")
        previous = by_identity.get(identity)
        if previous is None or hold.notional > previous.notional:
            by_identity[identity] = hold
    for hold in by_identity.values():
        group(hold.engine, hold.symbol)["reservations"].append(hold)
    for item in intents:
        if skip_decision and item.decision_id == skip_decision:
            continue
        group(item.engine, item.symbol)["intents"].append(item)

    pending = Decimal("0")
    working = Decimal("0")
    executed = Decimal("0")
    count = 0
    oldest = 0.0

    def note_age(created: float) -> None:
        nonlocal oldest
        if created > 0 and (oldest <= 0 or created < oldest):
            oldest = created

    for (engine, symbol), bucket in groups.items():
        res_rows: list[_Hold] = bucket["reservations"]
        intent_rows: list[_IntentQuote] = bucket["intents"]
        own_group = bool(skip_engine and skip_symbol and engine == _engine(skip_engine) and symbol == skip_symbol)
        if own_group and not intent_rows:
            continue
        intent_open = sum((item.open_quote for item in intent_rows), Decimal("0"))
        intent_executed = sum((item.executed_quote for item in intent_rows), Decimal("0"))
        intent_working = sum((item.remaining_quote for item in intent_rows), Decimal("0"))
        res_max = max((item.notional for item in res_rows), default=Decimal("0"))
        if intent_rows:
            chosen = max(intent_open, res_max)
            executed += intent_executed
            working += intent_working
            if chosen > intent_open:
                pending += chosen - intent_open
            count += max(len(intent_rows), 1 if res_rows else 0)
        else:
            chosen = res_max
            pending += chosen
            count += 1 if res_rows else 0
        if chosen <= 0:
            continue
        for item in res_rows:
            note_age(item.created_at)
        for item in intent_rows:
            note_age(item.created_at)

    covered = {symbol for (_engine_id, symbol) in groups}
    for symbol, quote, created in _extra_symbol_quote(pending_orders, trailing_arms, covered):
        if skip_symbol and symbol == skip_symbol and not skip_engine:
            continue
        pending += quote
        count += 1
        note_age(created)

    committed = pending + working + executed
    unresolved = [item for item in intents if item.remaining_quote > 0 or item.status == "FEE_PENDING"]
    ambiguous = [item for item in intents if item.status in {"PENDING", "ACKED"} and item.filled_qty <= 0]
    oldest_unresolved = min((item.created_at for item in unresolved if item.created_at > 0), default=0.0)
    oldest_ambiguous = min((item.created_at for item in ambiguous if item.created_at > 0), default=0.0)
    books = _local_buy_books(db_path)
    working_rows = []
    for item in intents:
        if item.remaining_quote <= 0:
            continue
        paper_qty, position_qty = books.get(item.trade_id, (None, None))
        booked = paper_qty if paper_qty is not None else position_qty
        venue_qty = float(item.filled_qty)
        unbooked = max(0.0, venue_qty - float(booked)) if booked is not None else 0.0
        working_rows.append(
            {
                "engine": item.engine,
                "symbol": item.symbol,
                "client_order_id": item.client_order_id,
                "status": item.status,
                "reserved_quote": float(item.open_quote),
                "filled_qty": venue_qty,
                "order_qty": float(item.order_qty),
                "venue_cumulative_filled": venue_qty,
                "locally_booked_filled": None if booked is None else float(booked),
                "unbooked_fill_qty": unbooked,
                "working_remainder": max(0.0, float(item.order_qty) - venue_qty),
                "position_qty": None if position_qty is None else float(position_qty),
                "age_sec": None if item.created_at <= 0 else round(max(0.0, stamp - item.created_at), 3),
            }
        )
    return BuyCashView(
        pending_buy_quote=pending,
        working_buy_quote=working,
        executed_unbooked_buy_quote=executed,
        committed_quote=committed,
        reservation_count=count,
        oldest_created_at=oldest,
        working_buy_count=len(working_rows),
        working_buys=working_rows,
        unresolved_buy_count=len(unresolved),
        oldest_unresolved_buy_age_sec=None if oldest_unresolved <= 0 else round(max(0.0, stamp - oldest_unresolved), 3),
        partial_buy_count=sum(1 for item in intents if item.status == "PARTIAL"),
        ambiguous_buy_count=len(ambiguous),
        oldest_ambiguous_buy_age_sec=None if oldest_ambiguous <= 0 else round(max(0.0, stamp - oldest_ambiguous), 3),
        custody_pending_count=sum(1 for item in intents if item.status == "FEE_PENDING"),
        potential_unprotected_symbols=sorted({item.symbol for item in intents if item.status in {"PENDING", "ACKED", "FEE_PENDING"} and item.symbol}),
    )


def committed_buy_quote(
    db_path: str,
    *,
    memory_reservations: dict[str, Any] | None = None,
    pending_orders: dict[str, Any] | None = None,
    trailing_arms: dict[str, Any] | None = None,
    exclude_symbol: str = "",
    exclude_decision_id: str = "",
    exclude_engine: str = "",
    exclude_reservation_id: str = "",
) -> Decimal:
    return buy_cash_view(
        db_path,
        memory_reservations=memory_reservations,
        pending_orders=pending_orders,
        trailing_arms=trailing_arms,
        exclude_symbol=exclude_symbol,
        exclude_decision_id=exclude_decision_id,
        exclude_engine=exclude_engine,
        exclude_reservation_id=exclude_reservation_id,
    ).committed_quote


def has_unbooked_buy_commitment(db_path: str) -> bool:
    return committed_buy_quote(db_path) > 0


def foreign_working_buy(db_path: str, engine_id: str, symbol: str, decision_id: str = "") -> bool:
    """True when this engine already has a different live BUY on the symbol."""
    want_engine = _engine("", engine_id)
    want_symbol = _symbol(symbol)
    skip = str(decision_id or "")
    return any(item.engine == want_engine and item.symbol == want_symbol and item.decision_id != skip for item in _load_intents(db_path))
