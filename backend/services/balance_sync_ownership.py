"""Exchange-balance comparison quantities.

The balance sync compares one physical exchange quantity per asset with the
sum of every legitimate ownership component. A later row for the same asset
must add to the earlier row. Engine lots stay separate in the report.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

QTY_TOLERANCE = 0.01
# The unit tolerance alone hides BTC/ETH gaps worth hundreds of dollars; a priced gap above this is drift.
NOTIONAL_TOLERANCE_USD = 1.0
_CASH_ASSETS = frozenset({"USDT", "USD", "USDC", "BUSD"})


@dataclass
class AssetOwnership:
    total: float = 0.0
    components: list[tuple[str, str, float]] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)

    def add(self, engine: str, status: str, quantity: float, ref: str = "") -> None:
        qty = float(quantity or 0.0)
        if qty <= 0:
            return
        self.components.append((str(engine or ""), str(status or "ACTIVE"), qty))
        self.refs.append(str(ref or ""))
        self.total += qty

    def report(self) -> str:
        return ", ".join(f"{engine or 'UNSCOPED'} {status}{f'[{ref}]' if ref else ''}={qty:.8f}" for (engine, status, qty), ref in zip(self.components, self.refs, strict=True))


def asset_code(symbol: str) -> str:
    raw = str(symbol or "").strip().upper()
    if "/" in raw:
        return raw.split("/", 1)[0]
    compact = raw.replace("-", "").replace("_", "")
    if compact.endswith("USDT") and len(compact) > 4:
        return compact[:-4]
    return compact


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def load_asset_ownership(conn: sqlite3.Connection) -> dict[str, AssetOwnership]:
    """Sum DAY, SCALP, dust, held dust, and protected inventory by asset.

    Reads the ownership tables only. The exchange balance is not an input.
    """
    books: dict[str, AssetOwnership] = {}

    def book(asset: str) -> AssetOwnership:
        return books.setdefault(asset, AssetOwnership())

    if _table_exists(conn, "portfolio_engine_positions"):
        cols = _columns(conn, "portfolio_engine_positions")
        engine = "COALESCE(engine_id, '')" if "engine_id" in cols else "''"
        status = "COALESCE(status, 'ACTIVE')" if "status" in cols else "'ACTIVE'"
        for engine_id, symbol, quantity, status_value in conn.execute(f"SELECT {engine}, symbol, quantity, {status} FROM portfolio_engine_positions WHERE quantity > 0"):
            code = asset_code(str(symbol or ""))
            if code and code not in _CASH_ASSETS:
                book(code).add(str(engine_id or ""), str(status_value or "ACTIVE"), float(quantity or 0.0))

    if _table_exists(conn, "engine_strategy_dust"):
        ref_col = "source_trade_id" if "source_trade_id" in _columns(conn, "engine_strategy_dust") else "''"
        for engine_id, symbol, quantity, source_trade_id in conn.execute(
            f"SELECT engine_id, symbol, quantity, {ref_col} FROM engine_strategy_dust WHERE status='HELD' AND quantity > 0 ORDER BY rowid"
        ):
            code = asset_code(str(symbol or ""))
            if code and code not in _CASH_ASSETS:
                book(code).add(str(engine_id or ""), "HELD_DUST", float(quantity or 0.0), ref=str(source_trade_id or "").rsplit("_", 1)[-1])

    if _table_exists(conn, "protected_external_inventory"):
        for symbol, quantity in conn.execute("SELECT symbol, quantity FROM protected_external_inventory WHERE quantity > 0"):
            code = asset_code(str(symbol or ""))
            if code and code not in _CASH_ASSETS:
                book(code).add("PROTECTED", "PROTECTED", float(quantity or 0.0))

    return books


def material(quantity: float, price: float = 0.0, *, tolerance: float = QTY_TOLERANCE, notional_tolerance: float = NOTIONAL_TOLERANCE_USD) -> bool:
    """Above the unit tolerance, or worth more than the notional tolerance at ``price``.

    Without a price only the unit tolerance applies.
    """
    qty = abs(float(quantity or 0.0))
    if qty > float(tolerance):
        return True
    px = float(price or 0.0)
    return px > 0 and qty * px > float(notional_tolerance)


def quantity_drift(
    exchange_qty: float,
    owned_qty: float,
    *,
    tolerance: float = QTY_TOLERANCE,
    price: float = 0.0,
    notional_tolerance: float = NOTIONAL_TOLERANCE_USD,
) -> bool:
    diff = float(exchange_qty or 0.0) - float(owned_qty or 0.0)
    return material(diff, price, tolerance=tolerance, notional_tolerance=notional_tolerance)
