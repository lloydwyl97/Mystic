"""Binance.US actual-fee audit from authenticated fills.

Authority is the exchange, not config. Pulls GET /api/v3/account (commission
rates) and GET /api/v3/myTrades for the DAY top-four, converts every
commission into USDT at the fill timestamp, and reports realized maker/taker
fee basis points per side, per symbol, and round-trip.

Run on the host whose IP is whitelisted for the API key (Ocean):
    sudo -u mystic venv/bin/python3 scripts/audit_binance_us_actual_fees.py
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

BASE = "https://api.binance.us"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
STABLES = {"USDT", "USD", "USDC", "BUSD"}


def _creds() -> tuple[str, str]:
    key = os.getenv("BINANCE_US_API_KEY") or os.getenv("BINANCE_API_KEY") or ""
    secret = os.getenv("BINANCE_US_SECRET_KEY") or os.getenv("BINANCE_SECRET") or ""
    if not key or not secret:
        raise SystemExit("BINANCE_US_API_KEY / BINANCE_US_SECRET_KEY not set")
    return key, secret


def _get(path: str, params: dict[str, Any] | None = None, *, signed: bool = False) -> Any:
    key, secret = _creds()
    p = dict(params or {})
    headers = {"X-MBX-APIKEY": key}
    if signed:
        p["timestamp"] = int(time.time() * 1000)
        p.setdefault("recvWindow", 10000)
        query = urllib.parse.urlencode(p)
        query += "&signature=" + hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    else:
        query = urllib.parse.urlencode(p)
    url = f"{BASE}{path}?{query}" if query else f"{BASE}{path}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


_PRICE_CACHE: dict[tuple[str, int], float] = {}


def price_at(asset: str, ts_ms: int) -> float:
    """USDT price of `asset` at the minute containing `ts_ms` (1.0 for stables)."""
    a = asset.upper()
    if a in STABLES:
        return 1.0
    minute = ts_ms - (ts_ms % 60000)
    ck = (a, minute)
    if ck in _PRICE_CACHE:
        return _PRICE_CACHE[ck]
    px = 0.0
    for quote in ("USDT", "USD"):
        try:
            kl = _get(
                "/api/v3/klines",
                {"symbol": f"{a}{quote}", "interval": "1m", "startTime": minute, "limit": 1},
            )
        except Exception:
            continue
        if kl:
            px = (float(kl[0][2]) + float(kl[0][3])) / 2.0
            break
    _PRICE_CACHE[ck] = px
    return px


def fetch_fills(symbol: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    from_id = 0
    while True:
        batch = _get("/api/v3/myTrades", {"symbol": symbol, "limit": 1000, "fromId": from_id}, signed=True)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        from_id = max(int(t["id"]) for t in batch) + 1
        time.sleep(0.3)
    return out


def enrich(symbol: str, fills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in fills:
        quote_qty = float(f["quoteQty"])
        commission = float(f["commission"])
        asset = str(f["commissionAsset"]).upper()
        px = price_at(asset, int(f["time"]))
        commission_usdt = commission * px
        rows.append(
            {
                "symbol": symbol,
                "orderId": int(f["orderId"]),
                "tradeId": int(f["id"]),
                "side": "BUY" if f["isBuyer"] else "SELL",
                "price": float(f["price"]),
                "qty": float(f["qty"]),
                "quoteQty": quote_qty,
                "commission": commission,
                "commissionAsset": asset,
                "commission_usdt": commission_usdt,
                "isMaker": bool(f["isMaker"]),
                "time": int(f["time"]),
                "effective_fee_bps": (commission_usdt / quote_qty * 10000.0) if quote_qty > 0 else 0.0,
            }
        )
    rows.sort(key=lambda r: r["time"])
    return rows


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"fills": 0, "notional_usdt": 0.0, "commission_usdt": 0.0, "fee_bps": 0.0}
    notional = sum(r["quoteQty"] for r in rows)
    commission = sum(r["commission_usdt"] for r in rows)
    return {
        "fills": len(rows),
        "notional_usdt": round(notional, 4),
        "commission_usdt": round(commission, 8),
        "fee_bps": round(commission / notional * 10000.0, 4) if notional > 0 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    account = _get("/api/v3/account", signed=True)
    rates = account.get("commissionRates", {})
    balances = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in account.get("balances", [])}
    bnb_balance = balances.get("BNB", 0.0)

    all_rows: list[dict[str, Any]] = []
    per_symbol: dict[str, list[dict[str, Any]]] = {}
    for sym in SYMBOLS:
        rows = enrich(sym, fetch_fills(sym))
        per_symbol[sym] = rows
        all_rows.extend(rows)

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        liq = "maker" if r["isMaker"] else "taker"
        leg = "entry" if r["side"] == "BUY" else "exit"
        by_bucket[f"{liq}_{leg}"].append(r)

    entries = [r for r in all_rows if r["side"] == "BUY"]
    exits = [r for r in all_rows if r["side"] == "SELL"]
    fee_assets = sorted({r["commissionAsset"] for r in all_rows})

    report = {
        "account_commission_rates": rates,
        "account_maker_commission_bps": account.get("makerCommission"),
        "account_taker_commission_bps": account.get("takerCommission"),
        "bnb_balance": bnb_balance,
        "paying_fees_with_bnb": bool(bnb_balance > 0 and "BNB" in fee_assets),
        "commission_assets_observed": fee_assets,
        "tier0_taker_applied": float(rates.get("taker", 0) or 0) <= 0.00010001,
        "totals": _agg(all_rows),
        "entry_all": _agg(entries),
        "exit_all": _agg(exits),
        "maker_entry": _agg(by_bucket["maker_entry"]),
        "taker_entry": _agg(by_bucket["taker_entry"]),
        "maker_exit": _agg(by_bucket["maker_exit"]),
        "taker_exit": _agg(by_bucket["taker_exit"]),
        "maker_fill_share": round(sum(1 for r in all_rows if r["isMaker"]) / len(all_rows), 6) if all_rows else 0.0,
        "round_trip_commission_bps": round(_agg(entries)["fee_bps"] + _agg(exits)["fee_bps"], 4),
        "by_symbol": {
            s: {
                "all": _agg(rows),
                "entry": _agg([r for r in rows if r["side"] == "BUY"]),
                "exit": _agg([r for r in rows if r["side"] == "SELL"]),
                "maker_fills": sum(1 for r in rows if r["isMaker"]),
                "taker_fills": sum(1 for r in rows if not r["isMaker"]),
                "round_trip_commission_bps": round(
                    _agg([r for r in rows if r["side"] == "BUY"])["fee_bps"] + _agg([r for r in rows if r["side"] == "SELL"])["fee_bps"],
                    4,
                ),
            }
            for s, rows in per_symbol.items()
        },
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"report": report, "fills": all_rows}, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out} ({len(all_rows)} fills)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
