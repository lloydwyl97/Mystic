"""Canonical failsafe equity: complete fresh exchange net-liquidatable equity.

The 52.66-hour audit showed ACCOUNT_FAILSAFE pausing entries on a $166-$177
figure while free USDT cash was ~$220. That input omitted cash and/or marked
assets and treated an incomplete ledger/P&L total as account equity.

Failsafe may trip only on one complete value:

    free USDT + locked USDT + net-liquidatable value of every exchange asset

Reservations do not reduce equity. Stale or incomplete snapshots cannot be
compared as if they were complete equity. The 10% principal trip itself is
unchanged.
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any

from backend.config.execution_cost_model import TAKER_COMMISSION_PCT
from backend.services.circuit_breaker_service import ACCOUNT_FAILSAFE_EQUITY_FRACTION, account_failsafe_tripped
from backend.services.day_entry_spendable import money
from backend.services.live_exchange_equity import QUOTE_ASSETS, mark_dust_asset, sell_fee_pct

logger = logging.getLogger(__name__)

STALE_AFTER_SEC = 90.0
MIN_BINANCE_ORDER_ID_DIGITS = 8


def is_real_binance_order_id(order_id: object) -> bool:
    raw = str(order_id or "").strip()
    return raw.isdigit() and len(raw) >= MIN_BINANCE_ORDER_ID_DIGITS and int(raw) > 0


def _asset_code(row: dict[str, Any]) -> str:
    return str(row.get("asset") or row.get("code") or "").strip().upper()


def _free_locked(row: dict[str, Any]) -> tuple[Decimal, Decimal]:
    free = money(row.get("free") if row.get("free") is not None else row.get("available") or 0)
    locked = money(row.get("locked") if row.get("locked") is not None else row.get("used") or 0)
    return free, locked


def build_canonical_nle(
    *,
    balances: list[dict[str, Any]] | None,
    bids: dict[str, Any] | None,
    fee_pct: object | None = None,
    as_of_epoch: float | None = None,
    now_epoch: float | None = None,
    reservations: object = 0,
    source: str = "exchange",
) -> dict[str, Any]:
    """Value every nonzero exchange asset once. Reservations are ignored."""
    _ = reservations  # must never reduce equity
    rate = money(fee_pct) if fee_pct is not None else sell_fee_pct()
    if rate <= 0:
        rate = money(TAKER_COMMISSION_PCT)
    now = float(now_epoch if now_epoch is not None else time.time())
    as_of = float(as_of_epoch if as_of_epoch is not None else now)
    age = max(0.0, now - as_of)
    rows = list(balances or [])
    bid_map = {str(k).upper(): money(v) for k, v in (bids or {}).items()}
    missing_bids: list[str] = []
    assets: list[dict[str, Any]] = []
    cash_free = Decimal("0")
    cash_locked = Decimal("0")
    asset_gross = Decimal("0")
    asset_fee = Decimal("0")
    seen_nonzero = 0
    for row in rows:
        asset = _asset_code(row)
        if not asset:
            continue
        free, locked = _free_locked(row)
        qty = free + locked
        if qty <= 0:
            continue
        seen_nonzero += 1
        if asset in QUOTE_ASSETS:
            cash_free += free
            cash_locked += locked
            assets.append(
                {
                    "asset": asset,
                    "free": str(free),
                    "locked": str(locked),
                    "quantity": str(qty),
                    "bid": "1",
                    "gross": str(qty),
                    "estimated_sell_fee": "0",
                    "net_liquidatable": str(qty),
                    "role": "quote_cash",
                }
            )
            continue
        bid = bid_map.get(f"{asset}USDT") or bid_map.get(asset)
        if bid is None or bid <= 0:
            missing_bids.append(asset)
            assets.append(
                {
                    "asset": asset,
                    "free": str(free),
                    "locked": str(locked),
                    "quantity": str(qty),
                    "bid": None,
                    "gross": None,
                    "estimated_sell_fee": None,
                    "net_liquidatable": None,
                    "role": "unpriced",
                }
            )
            continue
        marked = mark_dust_asset(
            symbol=f"{asset}/USDT",
            asset=asset,
            quantity=qty,
            executable_bid=bid,
            fee_pct=rate,
        )
        asset_gross += money(marked["dust_market_value"])
        asset_fee += money(marked["estimated_liquidation_cost"])
        assets.append(
            {
                "asset": asset,
                "free": str(free),
                "locked": str(locked),
                "quantity": str(qty),
                "bid": str(bid),
                "gross": marked["dust_market_value"],
                "estimated_sell_fee": marked["estimated_liquidation_cost"],
                "net_liquidatable": marked["net_liquidatable_value"],
                "role": "base_asset",
            }
        )
    cash = cash_free + cash_locked
    complete = bool(rows) and not missing_bids
    stale = age > STALE_AFTER_SEC
    nle = cash + asset_gross - asset_fee if complete else None
    return {
        "source": source,
        "as_of_epoch": as_of,
        "age_sec": age,
        "stale": stale,
        "complete": complete,
        "usable": bool(complete and not stale and nle is not None),
        "missing_bids": missing_bids,
        "nonzero_asset_count": seen_nonzero,
        "cash_free_usdt": str(cash_free),
        "cash_locked_usdt": str(cash_locked),
        "cash_usdt": str(cash),
        "asset_gross": str(asset_gross),
        "estimated_liquidation_cost": str(asset_fee),
        "net_liquidatable_equity": str(nle) if nle is not None else None,
        "reservations_ignored": str(money(reservations)),
        "assets": assets,
        "fee_pct": str(rate),
    }


def decide_account_failsafe(
    snapshot: dict[str, Any] | None,
    principal: object,
) -> dict[str, Any]:
    """Trip only on a complete fresh NLE. Incomplete/stale cannot fire."""
    prin = money(principal)
    snap = dict(snapshot or {})
    usable = bool(snap.get("usable"))
    nle = money(snap.get("net_liquidatable_equity")) if snap.get("net_liquidatable_equity") is not None else None
    threshold = prin * money(ACCOUNT_FAILSAFE_EQUITY_FRACTION) if prin > 0 else None
    if not usable or nle is None:
        reason = "incomplete_or_stale_nle_cannot_compare"
        if not snap:
            reason = "missing_nle_snapshot_cannot_compare"
        elif snap.get("stale"):
            reason = "stale_nle_cannot_compare"
        elif not snap.get("complete"):
            reason = "incomplete_nle_cannot_compare"
        logger.warning(
            "ACCOUNT_FAILSAFE_SKIPPED reason=%s cash=%s complete=%s stale=%s missing_bids=%s",
            reason,
            snap.get("cash_usdt"),
            snap.get("complete"),
            snap.get("stale"),
            snap.get("missing_bids"),
        )
        return {
            "tripped": False,
            "usable": False,
            "reason": reason,
            "nle": str(nle) if nle is not None else None,
            "principal": str(prin),
            "threshold": str(threshold) if threshold is not None else None,
            "snapshot": snap,
        }
    tripped = account_failsafe_tripped(float(nle), float(prin))
    logger.info(
        "ACCOUNT_FAILSAFE_VALUATION tripped=%s nle=%s cash_free=%s cash_locked=%s asset_gross=%s liq_cost=%s principal=%s threshold=%s assets=%s",
        tripped,
        nle,
        snap.get("cash_free_usdt"),
        snap.get("cash_locked_usdt"),
        snap.get("asset_gross"),
        snap.get("estimated_liquidation_cost"),
        prin,
        threshold,
        [(a.get("asset"), a.get("quantity"), a.get("bid"), a.get("net_liquidatable")) for a in snap.get("assets") or []],
    )
    return {
        "tripped": bool(tripped),
        "usable": True,
        "reason": "ACCOUNT_FAILSAFE" if tripped else "nle_above_failsafe",
        "nle": str(nle),
        "principal": str(prin),
        "threshold": str(threshold) if threshold is not None else None,
        "snapshot": snap,
    }
