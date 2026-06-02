"""
Trade Store
Persistent storage for orders and fills using SQLite (SQLAlchemy ORM).

Quick Test Checklist:
- Single exchange constant only (EXCHANGE_ID = "binance_us"); no other exchange strings.
- Symbols normalized to CCXT BASE/QUOTE via _to_ccxt_symbol().
- No unreachable code after returns.
- Logging-free and ASCII-safe.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from backend.services.db import get_engine, get_sessionmaker

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# Try shared helper; fallback to local normalization.
try:
    # e.g. backend/utils/symbols.py should provide to_ccxt_symbol
    from backend.utils.symbols import to_ccxt_symbol as _to_ccxt_symbol  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover

    def _to_ccxt_symbol(sym: str) -> str:
        s = (sym or "").strip().upper()
        if not s:
            return s
        if "/" in s:
            base, quote = s.split("/", 1)
            return f"{base}/{quote}"
        if "-" in s:
            base, quote = s.split("-", 1)
            if quote == "USD":
                quote = "USDT"
            return f"{base}/{quote}"
        if s.endswith("USDT"):
            return f"{s[:-4]}/USDT"
        if s.endswith("USD"):
            return f"{s[:-3]}/USDT"
        return f"{s}/USDT"


ENGINE = get_engine()
SessionLocal = get_sessionmaker(ENGINE)
Base = declarative_base()


class OrderRow(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(128), index=True)
    exchange_order_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    exchange: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))  # store lower-case: "buy" | "sell"
    order_type: Mapped[str] = mapped_column(String(16))  # store upper-case: "MARKET" | "LIMIT" | ...
    amount: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), index=True)  # lower-case: "submitted", "open", "executed", "cancelled"
    submitted_ts: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    ack_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_ts: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    extra: Mapped[str | None] = mapped_column(Text, nullable=True)


class FillRow(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(128), index=True)
    exchange_order_id: Mapped[str] = mapped_column(String(128), index=True)
    trade_id: Mapped[str] = mapped_column(String(128), index=True)
    exchange: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    side: Mapped[str] = mapped_column(String(8))  # lower-case
    price: Mapped[float] = mapped_column(Float)
    qty: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    fee_currency: Mapped[str | None] = mapped_column(String(12), nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


# Helpful indexes
Index("ix_orders_client_exch", OrderRow.client_order_id, OrderRow.exchange)
Index("ix_fills_trade_order", FillRow.trade_id, FillRow.exchange_order_id)


def init_db() -> None:
    Base.metadata.create_all(ENGINE)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def record_order_submitted(
    *,
    client_order_id: str | None,
    _exchange: str,
    symbol: str,
    side: str,
    order_type: str,
    amount: float,
    price: float,
    status: str = "submitted",
    extra: str | None = None,
) -> int:
    """Insert a new order row (normalizes exchange, symbol, side, type, status)."""
    # Enforce single exchange id and CCXT symbols.
    sym_ccxt = _to_ccxt_symbol(symbol)
    side_norm = str(side).lower()
    type_norm = str(order_type).upper()
    status_norm = str(status).lower()
    with SessionLocal() as s:
        row = OrderRow(
            client_order_id=(client_order_id or ""),
            exchange=EXCHANGE_ID,
            symbol=sym_ccxt,
            side=side_norm,
            order_type=type_norm,
            amount=float(amount),
            price=float(price),
            status=status_norm,
            submitted_ts=_now(),
            extra=(extra or ""),
        )
        s.add(row)
        s.commit()
        return int(row.id)


def record_order_ack(
    *,
    db_id: int | None = None,
    client_order_id: str | None = None,
    exchange_order_id: str | None = None,
    _exchange: str,
    status: str,
) -> None:
    """Acknowledge/attach exchange order id and mark status on the most recent matching order."""
    status_norm = str(status).lower()
    with SessionLocal() as s:
        q = s.query(OrderRow)
        if db_id is not None:
            q = q.filter(OrderRow.id == int(db_id))
        elif client_order_id:
            q = q.filter(OrderRow.client_order_id == client_order_id)
        else:
            return
        row = q.order_by(OrderRow.submitted_ts.desc()).first()
        if not row:
            return
        if exchange_order_id:
            row.exchange_order_id = exchange_order_id
        # Force canonical exchange id
        row.exchange = EXCHANGE_ID
        row.status = status_norm
        row.ack_ts = _now()
        row.updated_ts = _now()
        s.commit()


def update_order_status(*, exchange_order_id: str, status: str) -> None:
    """Update status by exchange_order_id (latest match)."""
    status_norm = str(status).lower()
    with SessionLocal() as s:
        row = s.query(OrderRow).filter(OrderRow.exchange_order_id == exchange_order_id).order_by(OrderRow.submitted_ts.desc()).first()
        if not row:
            return
        row.status = status_norm
        row.updated_ts = _now()
        s.commit()


def record_fill(
    *,
    client_order_id: str | None,
    exchange_order_id: str | None,
    trade_id: str,
    _exchange: str,
    symbol: str,
    side: str,
    price: float,
    qty: float,
    fee: float = 0.0,
    fee_currency: str | None = None,
    ts: datetime | None = None,
) -> None:
    """Insert a fill row (normalizes exchange, symbol, side)."""
    sym_ccxt = _to_ccxt_symbol(symbol)
    side_norm = str(side).lower()
    with SessionLocal() as s:
        fill = FillRow(
            client_order_id=(client_order_id or ""),
            exchange_order_id=(exchange_order_id or ""),
            trade_id=trade_id,
            exchange=EXCHANGE_ID,
            symbol=sym_ccxt,
            side=side_norm,
            price=float(price),
            qty=float(qty),
            fee=float(fee),
            fee_currency=(fee_currency or None),
            ts=(ts or _now()),
        )
        s.add(fill)
        s.commit()
