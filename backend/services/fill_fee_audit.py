"""
Fill-fee accounting audit — verification only, not a strategy mode.

On every paper/live SELL, records expected vs actual fee, spread/slippage, and net PnL.
Config fee rates are the source of truth unless the exchange reports an actual fill fee.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_economics import (
    MAKER_FEE,
    ORDERBOOK_HALF_SPREAD_ESTIMATE,
    SLIPPAGE_BUFFER,
    TAKER_FEE,
    get_trading_economics_display,
)
from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

TABLE = "portfolio_engine_fill_fee_audit"


def ensure_fill_fee_audit_table(db_path: str | None = None) -> None:
    path = db_path or DATABASE_PATH
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                trade_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL DEFAULT 'SELL',
                mode TEXT NOT NULL,
                mark_price REAL,
                fill_price REAL,
                entry_price REAL,
                quantity REAL,
                expected_fee_usd REAL,
                actual_fee_usd REAL,
                expected_spread_slippage_usd REAL,
                realized_slippage_usd REAL,
                net_pnl_after_actual_fees REAL,
                fee_delta_usd REAL,
                slippage_delta_usd REAL,
                expected_fee_rate REAL,
                actual_fee_rate REAL,
                fee_from_exchange INTEGER NOT NULL DEFAULT 0,
                audit_json TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _extract_exchange_fee_usd(live_order: dict[str, Any] | None) -> float | None:
    if not live_order or not isinstance(live_order, dict):
        return None
    for key in ("fee", "fees"):
        raw = live_order.get(key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, dict):
            try:
                cost = float(raw.get("cost") or 0.0)
                if cost > 0:
                    return cost
            except (TypeError, ValueError):
                pass
        if isinstance(raw, list):
            total = 0.0
            for item in raw:
                if isinstance(item, dict):
                    try:
                        total += float(item.get("cost") or 0.0)
                    except (TypeError, ValueError):
                        pass
            if total > 0:
                return total
    return None


def build_sell_fill_fee_audit(
    *,
    symbol: str,
    trade_id: str,
    mode: str,
    entry_price: float,
    mark_price: float,
    fill_price: float,
    quantity: float,
    sell_fee_rate: float,
    entry_fee_usd: float,
    net_pnl_after_fees: float,
    live_order: dict[str, Any] | None = None,
    half_spread_pct: float | None = None,
) -> dict[str, Any]:
    """Build expected vs actual accounting row for a completed SELL."""
    hs = float(half_spread_pct if half_spread_pct is not None else ORDERBOOK_HALF_SPREAD_ESTIMATE)
    notional = max(0.0, float(quantity) * float(fill_price))
    expected_fee = notional * float(sell_fee_rate)
    exchange_fee = _extract_exchange_fee_usd(live_order)
    fee_from_exchange = exchange_fee is not None
    actual_fee = float(exchange_fee) if fee_from_exchange else expected_fee
    actual_fee_rate = (actual_fee / notional) if notional > 0 else float(sell_fee_rate)

    # Sell: adverse move vs mark → slippage/spread impact (USD)
    realized_slip = max(0.0, (float(mark_price) - float(fill_price)) * float(quantity))
    expected_slip = float(mark_price) * hs * float(quantity) + float(mark_price) * SLIPPAGE_BUFFER * float(quantity)

    fee_delta = actual_fee - expected_fee
    slip_delta = realized_slip - expected_slip

    audit = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trade_id": trade_id,
        "symbol": symbol,
        "side": "SELL",
        "mode": mode,
        "mark_price": round(float(mark_price), 8),
        "fill_price": round(float(fill_price), 8),
        "entry_price": round(float(entry_price), 8),
        "quantity": float(quantity),
        "expected_fee_usd": round(expected_fee, 6),
        "actual_fee_usd": round(actual_fee, 6),
        "expected_spread_slippage_usd": round(expected_slip, 6),
        "realized_slippage_usd": round(realized_slip, 6),
        "net_pnl_after_actual_fees": round(float(net_pnl_after_fees), 6),
        "fee_delta_usd": round(fee_delta, 6),
        "slippage_delta_usd": round(slip_delta, 6),
        "expected_fee_rate": float(sell_fee_rate),
        "actual_fee_rate": round(actual_fee_rate, 8),
        "fee_from_exchange": fee_from_exchange,
        "entry_fee_usd": round(float(entry_fee_usd), 6),
        "economics_config": get_trading_economics_display(),
        "note": ("Accounting verification only. Config fee rates apply unless exchange fill fee is present."),
    }
    return audit


def persist_fill_fee_audit(audit: dict[str, Any], db_path: str | None = None) -> None:
    path = db_path or DATABASE_PATH
    ensure_fill_fee_audit_table(path)
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.execute(
            f"""
            INSERT INTO {TABLE} (
                ts, trade_id, symbol, side, mode,
                mark_price, fill_price, entry_price, quantity,
                expected_fee_usd, actual_fee_usd,
                expected_spread_slippage_usd, realized_slippage_usd,
                net_pnl_after_actual_fees, fee_delta_usd, slippage_delta_usd,
                expected_fee_rate, actual_fee_rate, fee_from_exchange, audit_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                audit["ts"],
                audit["trade_id"],
                audit["symbol"],
                audit.get("side", "SELL"),
                audit["mode"],
                audit.get("mark_price"),
                audit.get("fill_price"),
                audit.get("entry_price"),
                audit.get("quantity"),
                audit.get("expected_fee_usd"),
                audit.get("actual_fee_usd"),
                audit.get("expected_spread_slippage_usd"),
                audit.get("realized_slippage_usd"),
                audit.get("net_pnl_after_actual_fees"),
                audit.get("fee_delta_usd"),
                audit.get("slippage_delta_usd"),
                audit.get("expected_fee_rate"),
                audit.get("actual_fee_rate"),
                1 if audit.get("fee_from_exchange") else 0,
                json.dumps(audit, default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_sell_fill_fee_audit(
    *,
    symbol: str,
    trade_id: str,
    mode: str,
    entry_price: float,
    mark_price: float,
    fill_price: float,
    quantity: float,
    sell_fee_rate: float,
    entry_fee_usd: float,
    net_pnl_after_fees: float,
    live_order: dict[str, Any] | None = None,
    half_spread_pct: float | None = None,
) -> dict[str, Any]:
    """Log + persist fill-fee audit for a completed SELL."""
    audit = build_sell_fill_fee_audit(
        symbol=symbol,
        trade_id=trade_id,
        mode=mode,
        entry_price=entry_price,
        mark_price=mark_price,
        fill_price=fill_price,
        quantity=quantity,
        sell_fee_rate=sell_fee_rate,
        entry_fee_usd=entry_fee_usd,
        net_pnl_after_fees=net_pnl_after_fees,
        live_order=live_order,
        half_spread_pct=half_spread_pct,
    )
    logger.info("FILL_FEE_AUDIT %s", json.dumps(audit, separators=(",", ":"), default=str))
    try:
        persist_fill_fee_audit(audit)
    except Exception:
        logger.exception("fill_fee_audit persist failed trade_id=%s", trade_id)
    return audit


def bnb_fee_discount_status() -> dict[str, Any]:
    """Report whether BNB fee discount is configured (Mystic does not enable it by default)."""
    import os

    enabled_env = os.getenv("BINANCE_US_BNB_FEE_DISCOUNT", "").strip().lower()
    enabled = enabled_env in ("1", "true", "yes", "on")
    return {
        "bnb_fee_discount_enabled": enabled,
        "bnb_fee_discount_env": "BINANCE_US_BNB_FEE_DISCOUNT",
        "config_fee_override_from_account": False,
        "note": ("Mystic uses trading_economics MAKER_FEE/TAKER_FEE unless exchange fill reports actual fee. BNB discount is not wired unless BINANCE_US_BNB_FEE_DISCOUNT=true."),
    }


def config_fee_override_locations() -> list[str]:
    """Document where actual exchange fees can override config (accounting only)."""
    return [
        "fill_fee_audit.record_sell_fill_fee_audit: actual_fee from live_order fee when present",
        "portfolio_engine._recover_manual_sell_fill_from_exchange: fee_usd from trade history",
        "live_recovered_close_writer: fee from exchange fill",
    ]
