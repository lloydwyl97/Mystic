"""Transaction-level cash and NLE bridge. Decimal only. No invented baselines."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from backend.services.day_entry_spendable import money
from backend.services.live_exchange_equity import mark_dust_asset, sell_fee_pct


def quote_commission(fill: dict[str, Any]) -> Decimal:
    asset = str(fill.get("commissionAsset") or fill.get("fee_asset") or fill.get("fee_ccy") or "").upper()
    commission = money(fill.get("commission") if fill.get("commission") is not None else fill.get("fee") or fill.get("fee_cost") or 0)
    if commission <= 0:
        return Decimal("0")
    if asset in {"USDT", "USD", "BUSD", "USDC", "QUOTE"}:
        return commission
    return Decimal("0")


def base_commission_quote(fill: dict[str, Any], *, bid: object) -> Decimal:
    asset = str(fill.get("commissionAsset") or fill.get("fee_asset") or fill.get("fee_ccy") or "").upper()
    commission = money(fill.get("commission") if fill.get("commission") is not None else fill.get("fee") or fill.get("fee_cost") or 0)
    symbol = str(fill.get("symbol") or "").upper().replace("/", "")
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    if commission <= 0 or asset in {"", "USDT", "USD", "BUSD", "USDC", "QUOTE"}:
        return Decimal("0")
    if asset != base:
        return Decimal("0")
    return commission * money(bid)


def fill_quote_notional(fill: dict[str, Any]) -> Decimal:
    qty = money(fill.get("qty") if fill.get("qty") is not None else fill.get("quantity") or fill.get("executedQty") or 0)
    px = money(fill.get("price") or 0)
    quote = money(fill.get("quoteQty") if fill.get("quoteQty") is not None else fill.get("cost") or 0)
    if quote > 0:
        return quote
    return qty * px


def is_buy(fill: dict[str, Any]) -> bool:
    side = str(fill.get("side") or fill.get("isBuyer") or "").upper()
    if side in {"BUY", "TRUE", "1"}:
        return True
    if side in {"SELL", "FALSE", "0"}:
        return False
    return bool(fill.get("isBuyer"))


def dust_nle(coins: list[dict[str, Any]] | None, *, fee_pct: object | None = None) -> Decimal:
    total = Decimal("0")
    rate = money(fee_pct) if fee_pct is not None else sell_fee_pct()
    for row in coins or []:
        marked = mark_dust_asset(
            symbol=str(row.get("symbol") or ""),
            asset=str(row.get("asset") or ""),
            quantity=row.get("quantity") or 0,
            executable_bid=row.get("executable_bid") or row.get("bid") or 0,
            fee_pct=rate,
        )
        total += money(marked["net_liquidatable_value"])
    return total


def bridge_account(
    *,
    opening_cash: object,
    fills: list[dict[str, Any]],
    opening_dust: list[dict[str, Any]] | None = None,
    closing_dust: list[dict[str, Any]] | None = None,
    opening_position_marks: list[dict[str, Any]] | None = None,
    closing_position_marks: list[dict[str, Any]] | None = None,
    deposits: object = 0,
    withdrawals: object = 0,
    conversions: object = 0,
    rebates: object = 0,
    fee_pct: object | None = None,
    bids_for_base_fees: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """opening cash + sells - buys - quote fees +/- dust +/- marks +/- transfers = ending NLE."""
    cash = money(opening_cash)
    sell_credit = Decimal("0")
    buy_debit = Decimal("0")
    quote_fees = Decimal("0")
    base_fee_quote = Decimal("0")
    bid_map = {str(k).upper(): money(v) for k, v in (bids_for_base_fees or {}).items()}
    for fill in fills or []:
        notional = fill_quote_notional(fill)
        qfee = quote_commission(fill)
        symbol = str(fill.get("symbol") or "").upper().replace("/", "")
        base = symbol[:-4] if symbol.endswith("USDT") else symbol
        bfee = base_commission_quote(fill, bid=bid_map.get(f"{base}USDT") or bid_map.get(base) or fill.get("price") or 0)
        if is_buy(fill):
            buy_debit += notional
        else:
            sell_credit += notional
        quote_fees += qfee
        base_fee_quote += bfee
    open_dust = dust_nle(opening_dust, fee_pct=fee_pct)
    close_dust = dust_nle(closing_dust, fee_pct=fee_pct)
    dust_delta = close_dust - open_dust

    def _mark_sum(rows: list[dict[str, Any]] | None) -> Decimal:
        total = Decimal("0")
        for row in rows or []:
            total += money(row.get("net_liquidatable") or row.get("market_value") or 0)
        return total

    open_pos = _mark_sum(opening_position_marks)
    close_pos = _mark_sum(closing_position_marks)
    mark_delta = close_pos - open_pos
    xfer = money(deposits) + money(rebates) + money(conversions) - money(withdrawals)
    # Base-asset commissions reduce inventory; dust/mark deltas already carry
    # that quantity. Subtracting them again would double-count the fee.
    ending_cash = cash + sell_credit - buy_debit - quote_fees + xfer
    opening_nle = cash + open_dust + open_pos
    ending_nle = ending_cash + close_dust + close_pos
    reconstructed_nle = opening_nle + (ending_cash - cash) + dust_delta + mark_delta
    return {
        "opening_cash": str(cash),
        "opening_nle": str(opening_nle),
        "sell_credit": str(sell_credit),
        "buy_debit": str(buy_debit),
        "quote_commission": str(quote_fees),
        "base_commission_quote": str(base_fee_quote),
        "base_commission_in_dust_delta": True,
        "opening_dust_nle": str(open_dust),
        "closing_dust_nle": str(close_dust),
        "dust_delta": str(dust_delta),
        "opening_position_nle": str(open_pos),
        "closing_position_nle": str(close_pos),
        "position_mark_delta": str(mark_delta),
        "transfers": str(xfer),
        "ending_cash": str(ending_cash),
        "ending_nle": str(ending_nle),
        "reconstructed_nle": str(reconstructed_nle),
        "bridge_residual": str(ending_nle - reconstructed_nle),
        "identity": "opening_nle + (sells - buys - quote_fees + transfers) + dust_delta + mark_delta",
    }
