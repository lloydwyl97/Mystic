"""Live fill economics — Binance commissions and live-mode PnL isolation.

Paper cost models are unchanged. Historical paper rows are not rewritten.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from backend.utils.symbols import normalize_symbol

# coin_performance field classification (audit 2026-08-24).
# A = shared learning signal (intentionally mixed paper/live history)
# B = live-only safety / account-state authority when DAY is LIVE
# C = currently contaminated if a B field is computed from mixed history
COIN_PERFORMANCE_FIELD_CLASS: dict[str, str] = {
    "win_rate_20": "A",
    "sizing_multiplier": "A",
    "expectancy": "A",
    "avg_win": "A",
    "avg_loss": "A",
    "profit_factor": "A",  # learning; live PF *pause* is B
    "trades_last_30d": "A",
    "stop_loss_hits_10": "B",  # C until live rolling last-N is used
    "pause_until": "B",  # C if inherited from paper loss_heavy / 24h / PF pause
    "trades_24h": "B",  # pause Rule 1 authority when LIVE
    "pnl_24h": "B",  # pause Rule 1 authority when LIVE
}


@dataclass(frozen=True)
class LiveCommission:
    """Asset-aware exchange commission for one live fill."""

    usd: float
    items: tuple[dict[str, Any], ...] = ()
    fee_from_exchange: bool = False
    base_qty_reduction: float = 0.0
    quote_commission_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_commission_usd": round(float(self.usd), 8),
            "quote_commission_usd": round(float(self.quote_commission_usd), 8),
            "fee_from_exchange": bool(self.fee_from_exchange),
            "base_qty_reduction": float(self.base_qty_reduction),
            "items": list(self.items),
        }


def _base_asset(symbol: str) -> str:
    ns = normalize_symbol(symbol).replace("/", "")
    if ns.endswith("USDT"):
        return ns[:-4]
    if ns.endswith("USD"):
        return ns[:-3]
    return ns


def _as_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _iter_fee_items(order: dict[str, Any] | None) -> list[tuple[float, str]]:
    if not order or not isinstance(order, dict):
        return []
    out: list[tuple[float, str]] = []

    def _add(amount: Any, asset: Any) -> None:
        qty = _as_float(amount)
        if qty is None or qty == 0:
            return
        out.append((abs(qty), str(asset or "").upper()))

    raw_fee = order.get("fee")
    if isinstance(raw_fee, dict):
        _add(raw_fee.get("cost") if raw_fee.get("cost") is not None else raw_fee.get("amount"), raw_fee.get("currency") or raw_fee.get("asset"))
    elif isinstance(raw_fee, (int, float)) and float(raw_fee) != 0:
        _add(raw_fee, order.get("feeAsset") or order.get("commissionAsset") or "USDT")

    raw_fees = order.get("fees")
    if isinstance(raw_fees, list):
        for item in raw_fees:
            if isinstance(item, dict):
                _add(item.get("cost") if item.get("cost") is not None else item.get("amount"), item.get("currency") or item.get("asset"))

    for trade in order.get("trades") or []:
        if not isinstance(trade, dict):
            continue
        tfee = trade.get("fee")
        if isinstance(tfee, dict):
            _add(tfee.get("cost") if tfee.get("cost") is not None else tfee.get("amount"), tfee.get("currency") or tfee.get("asset"))
        _add(trade.get("commission"), trade.get("commissionAsset"))

    _add(order.get("commission"), order.get("commissionAsset"))
    info = order.get("info")
    if isinstance(info, dict):
        _add(info.get("commission"), info.get("commissionAsset"))
        for fill in info.get("fills") or []:
            if isinstance(fill, dict):
                _add(fill.get("commission"), fill.get("commissionAsset"))
    return out


def extract_live_commission(
    order: dict[str, Any] | None,
    *,
    symbol: str,
    fill_price: float,
) -> LiveCommission:
    """Convert Binance/CCXT fee fields into quote-USD and optional base-qty reduction."""
    items = _iter_fee_items(order)
    if not items:
        return LiveCommission(usd=0.0, fee_from_exchange=False)

    base = _base_asset(symbol)
    usd = 0.0
    quote_usd = 0.0
    base_qty = 0.0
    detail: list[dict[str, Any]] = []
    px = float(fill_price or 0.0)
    for amount, asset in items:
        if not asset or asset in {"USDT", "USD", "BUSD"}:
            usd += amount
            quote_usd += amount
            conv = amount
        elif asset == base:
            base_qty += amount
            conv = amount * px if px > 0 else 0.0
            usd += conv
        else:
            # Unknown asset (e.g. BNB). Keep the raw item; do not invent a FX rate.
            conv = 0.0
        detail.append({"amount": amount, "asset": asset, "usd": round(conv, 8)})
    return LiveCommission(
        usd=float(usd),
        items=tuple(detail),
        fee_from_exchange=True,
        base_qty_reduction=float(base_qty),
        quote_commission_usd=float(quote_usd),
    )


def apply_live_buy_economics(
    *,
    filled_qty: float,
    fill_price: float,
    modeled_fee: float,
    commission: LiveCommission,
) -> tuple[float, float, float]:
    """Return (received_qty, entry_fee_usd, cash_debit). Exchange fees win when present."""
    filled = max(0.0, float(filled_qty))
    px = float(fill_price or 0.0)
    if commission.fee_from_exchange:
        received = max(0.0, filled - float(commission.base_qty_reduction or 0.0))
        fee = float(commission.usd)
        cash = filled * px + float(commission.quote_commission_usd or 0.0)
        return received, fee, cash
    return filled, float(modeled_fee), filled * px + float(modeled_fee)


def apply_live_sell_economics(
    *,
    quantity: float,
    fill_price: float,
    modeled_fee: float,
    commission: LiveCommission,
) -> tuple[float, float]:
    """Return (fee_usd, proceeds). Do not also subtract modeled slippage."""
    qty = float(quantity)
    px = float(fill_price or 0.0)
    if commission.fee_from_exchange:
        fee = float(commission.usd)
    else:
        fee = float(modeled_fee)
    return fee, (qty * px) - fee


def live_round_trip_net(
    *,
    quantity: float,
    entry_price: float,
    exit_price: float,
    entry_commission_usd: float,
    exit_commission_usd: float,
) -> dict[str, float]:
    """Gross is fill-to-fill. Net subtracts actual exchange commissions only."""
    qty = float(quantity)
    gross = qty * (float(exit_price) - float(entry_price))
    entry_fee = max(0.0, float(entry_commission_usd or 0.0))
    exit_fee = max(0.0, float(exit_commission_usd or 0.0))
    return {
        "gross_price_pnl": round(gross, 8),
        "exchange_commission_usd": round(entry_fee + exit_fee, 8),
        "net_realized_pnl": round(gross - entry_fee - exit_fee, 8),
    }


def sum_realized_pnl_by_mode(
    db_path: str,
    *,
    mode: str,
) -> float:
    """Sum SELL pnl for one paper_trades.mode. Does not rewrite rows."""
    wanted = str(mode or "").strip().lower()
    if not wanted:
        return 0.0
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            expr = _sell_pnl_expr(conn)
            row = conn.execute(
                f"""
                SELECT COALESCE(SUM({expr}), 0)
                FROM paper_trades
                WHERE UPPER(side)='SELL'
                  AND pnl IS NOT NULL
                  AND LOWER(COALESCE(mode, '')) = ?
                  AND COALESCE(is_synthetic, 0) = 0
                  AND COALESCE(exit_type, '') NOT IN (
                    'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR', 'RESEARCH_RESET_EXIT'
                  )
                """,
                (wanted,),
            ).fetchone()
        return float((row or (0.0,))[0] or 0.0)
    except sqlite3.Error:
        return 0.0


def recent_sell_pnls(
    db_path: str,
    symbol: str,
    *,
    limit: int = 10,
    mode: str | None = None,
) -> list[float]:
    """Most recent completed SELL pnls for a symbol, optionally mode-filtered."""
    alt = normalize_symbol(symbol).replace("/", "").upper()
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            expr = _sell_pnl_expr(conn)
            sql_params: list[Any] = [alt]
            mode_sql = ""
            if mode:
                mode_sql = " AND LOWER(COALESCE(mode, '')) = ?"
                sql_params.append(str(mode).strip().lower())
            sql_params.append(int(limit))
            rows = conn.execute(
                f"""
                SELECT {expr}
                FROM paper_trades
                WHERE UPPER(side)='SELL'
                  AND pnl IS NOT NULL
                  AND COALESCE(is_synthetic, 0) = 0
                  AND REPLACE(REPLACE(UPPER(symbol), '/', ''), '_', '') = ?
                  {mode_sql}
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                sql_params,
            ).fetchall()
        return [float(r[0]) for r in rows if r and r[0] is not None]
    except sqlite3.Error:
        return []


def _sell_pnl_expr(conn: sqlite3.Connection) -> str:
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    except sqlite3.Error:
        return "pnl"
    if "pnl_usd_net" in cols:
        return "COALESCE(pnl_usd_net, pnl)"
    return "pnl"


def live_risk_loss_hits(*, is_live_day: bool, sticky_hits: int, live_pnls: list[float], window: int = 10) -> int:
    """LIVE uses true rolling last-N. Paper keeps existing sticky counter."""
    if is_live_day:
        return rolling_loss_count(live_pnls, window)
    return int(sticky_hits)


def live_closes_24h(db_path: str, symbol: str) -> tuple[int, float]:
    """Live-mode SELL count and pnl sum in the last 24h."""
    alt = normalize_symbol(symbol).replace("/", "").upper()
    try:
        with sqlite3.connect(db_path, timeout=5) as conn:
            expr = _sell_pnl_expr(conn)
            row = conn.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM({expr}), 0)
                FROM paper_trades
                WHERE UPPER(side)='SELL'
                  AND pnl IS NOT NULL
                  AND COALESCE(is_synthetic, 0) = 0
                  AND LOWER(COALESCE(mode, '')) = 'live'
                  AND REPLACE(REPLACE(UPPER(symbol), '/', ''), '_', '') = ?
                  AND timestamp >= datetime('now', '-24 hours')
                """,
                (alt,),
            ).fetchone()
        return int((row or (0, 0.0))[0] or 0), float((row or (0, 0.0))[1] or 0.0)
    except sqlite3.Error:
        return 0, 0.0


def rolling_loss_count(pnls: list[float], window: int = 10) -> int:
    """True last-N losses. Wins in the window reduce the count."""
    slice_ = list(pnls)[: max(1, int(window))]
    return sum(1 for p in slice_ if float(p) < 0)
