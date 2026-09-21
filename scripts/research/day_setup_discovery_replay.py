#!/usr/bin/env python3
"""Replay Ocean trailing-buy arms vs early-trend / structured-pullback."""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from backend.config.day_setup_discovery import EARLY_TREND, STRUCTURED_PULLBACK
from backend.config.execution_cost_model import honest_all_in_rt_pct
from backend.services.day_setup_discovery import classify_setup, market_structure
from backend.services.day_trailing_buy import formulas_for_symbol, observe_book
from backend.services.day_trailing_buy_store import TRAIL_LOW, WAIT_DIP

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
INTENTS = Path("/tmp/tb_all_intents.json")
DB = Path("/tmp/sidecar_setup_1m.db")
OUT = Path("/tmp/day_setup_discovery_replay.json")


def _api(s: str) -> str:
    return str(s or "").replace("/", "").replace("-", "").upper()


def _fetch(sym: str, start_ms: int, end_ms: int) -> list:
    rows = []
    cur = start_ms
    while cur < end_ms:
        url = f"https://api.binance.us/api/v3/klines?symbol={sym}&interval=1m&startTime={cur}&endTime={end_ms}&limit=1000"
        with urllib.request.urlopen(url, timeout=20) as resp:
            chunk = json.loads(resp.read().decode())
        if not chunk:
            break
        rows.extend(chunk)
        cur = int(chunk[-1][0]) + 60_000
        time.sleep(0.2)
    return rows


def ensure_bars() -> dict[str, list]:
    intents = json.loads(INTENTS.read_text())
    lo = int(min(r["arm_ts"] for r in intents) - 6 * 3600) * 1000
    hi = int(max(r["arm_ts"] for r in intents) + 6 * 3600) * 1000
    conn = sqlite3.connect(str(DB))
    conn.execute("CREATE TABLE IF NOT EXISTS feature_ohlcv (symbol TEXT, interval TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_kl ON feature_ohlcv(symbol, ts)")
    out: dict[str, list] = {}
    for sym in SYMBOLS:
        have = conn.execute(
            "SELECT COUNT(*) FROM feature_ohlcv WHERE symbol=? AND interval='1m' AND ts>=? AND ts<=?",
            (sym, lo // 1000, hi // 1000),
        ).fetchone()[0]
        if have < 200:
            raw = _fetch(sym, lo, hi)
            conn.execute("DELETE FROM feature_ohlcv WHERE symbol=? AND interval='1m'", (sym,))
            conn.executemany(
                "INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?,?)",
                [(sym, "1m", int(r[0] // 1000), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])) for r in raw],
            )
            conn.commit()
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM feature_ohlcv WHERE symbol=? AND interval='1m' ORDER BY ts",
            (sym,),
        ).fetchall()
        out[sym] = [(int(t), float(o), float(h), float(lo_), float(c), float(v)) for t, o, h, lo_, c, v in rows]
    conn.close()
    return out


def _asof(bars, ts):
    return [b for b in bars if b[0] <= ts]


def baseline_fill(bars, arm_ts, arm_ask, symbol):
    fwd = [b for b in bars if arm_ts <= b[0] <= arm_ts + 900]
    formulas = formulas_for_symbol(symbol, arm_spread_bps=0.0)
    intent = {
        "status": WAIT_DIP,
        "arm_ask": arm_ask,
        "min_dip_bps": formulas["min_dip_bps"],
        "rebound_bps": formulas["rebound_bps"],
        "required_improvement_bps": formulas["required_improvement_bps"],
        "lowest_ask": 0.0,
        "lowest_ask_ts": 0.0,
        "expires_at": arm_ts + 900,
    }
    for ts, _o, high, low, _close, _v in fwd:
        d = observe_book(intent, ask=low, now=float(ts), book_fresh=True)
        if d.action == "trail" or d.status == TRAIL_LOW:
            intent["status"] = TRAIL_LOW
            intent["lowest_ask"] = d.lowest_ask or low
            intent["lowest_ask_ts"] = d.lowest_ask_ts or ts
        d2 = observe_book(intent, ask=high, now=float(ts) + 1, book_fresh=True)
        if d2.action in {"expire", "cancel"}:
            return None, d2.reason
        if d2.action == "submit":
            fill = min(high, max(low, d2.current_ask))
            return fill, "REBOUND_CONFIRMED"
        if d.action in {"expire", "cancel"}:
            return None, d.reason
    return None, "TIMEOUT"


def equity(rows):
    if not rows:
        return {"n": 0, "fills": 0, "expectancy_bps": 0, "usd": 0, "pf": None, "win_rate": 0, "dd": 0, "med_ext": None, "mfe": 0, "mae": 0}
    nets = [r["net"] for r in rows]
    usd = [r["usd"] for r in rows]
    wins = [u for u in usd if u > 0]
    losses = [abs(u) for u in usd if u <= 0]
    eq = peak = dd = 0.0
    for u in usd:
        eq += u
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    ext = sorted(r["ext"] for r in rows if r.get("ext") is not None)
    return {
        "n": len(rows),
        "fills": len(rows),
        "expectancy_bps": round(sum(nets) / len(nets) * 1e4, 3),
        "usd": round(sum(usd), 4),
        "pf": None if not losses else round(sum(wins) / sum(losses), 3) if sum(losses) else None,
        "win_rate": round(sum(1 for n in nets if n > 0) / len(nets), 3),
        "dd": round(dd, 4),
        "med_ext": round(ext[len(ext) // 2], 3) if ext else None,
        "mfe": round(sum(r["mfe"] for r in rows) / len(rows) * 1e4, 3),
        "mae": round(sum(r["mae"] for r in rows) / len(rows) * 1e4, 3),
    }


def post_fill_path(bars, fill_ts, fill, symbol, notional):
    cost = honest_all_in_rt_pct(symbol)
    fwd = [b for b in bars if fill_ts <= b[0] <= fill_ts + 14 * 24 * 3600]
    if not fwd:
        return None
    mfe = mae = 0.0
    exit_px = fill
    for _ts, _o, high, low, close, _v in fwd[: 12 * 60]:
        mfe = max(mfe, high / fill - 1.0)
        mae = min(mae, low / fill - 1.0)
        exit_px = close
    net = exit_px / fill - 1.0 - cost
    return {"net": net, "usd": net * notional, "mfe": mfe - cost, "mae": mae - cost}


def main() -> int:
    bars = ensure_bars()
    intents = json.loads(INTENTS.read_text())
    late = []
    buckets = {k: [] for k in ("baseline", "early", "pullback", "combined")}
    classes = defaultdict(int)
    for it in intents:
        sym = _api(it["symbol"])
        if sym not in bars:
            continue
        ts = int(it["arm_ts"])
        ask = float(it["arm_ask"])
        atr = float(it.get("atr") or 0.0)
        notion = float(it.get("notional_usd") or 50.0)
        asof = _asof(bars[sym], ts)
        ms = market_structure(asof, ts=ts, atr=atr, ask=ask)
        clf = classify_setup(asof, symbol=sym, ts=ts, atr=atr, ask=ask)
        classes[str(clf["setup_class"])] += 1
        late.append(
            {
                "intent_id": it["intent_id"],
                "symbol": sym,
                "setup": it.get("setup"),
                "status": it.get("status"),
                "cancel_reason": it.get("cancel_reason"),
                "arm_ask": ask,
                "created_at": it.get("created_at"),
                "arm_ts": ts,
                "setup_class": clf["setup_class"],
                "class_reason": clf["reason"],
                **{
                    k: ms[k]
                    for k in (
                        "ret_5",
                        "ret_15",
                        "ret_30",
                        "ret_60",
                        "ret_240",
                        "dist_low_15_bps",
                        "dist_low_60_bps",
                        "dist_low_240_bps",
                        "dist_high_15_bps",
                        "dist_high_60_bps",
                        "dist_high_240_bps",
                        "range_4h_pct",
                        "atr_bps",
                    )
                },
            }
        )
        fill, _why = baseline_fill(bars[sym], ts, ask, sym)
        if fill:
            path = post_fill_path(bars[sym], ts, fill, sym, notion)
            if path:
                path.update({"symbol": sym, "ext": ms["range_4h_pct"], "setup": it.get("setup")})
                buckets["baseline"].append(path)
        if clf["setup_class"] == EARLY_TREND:
            path = post_fill_path(bars[sym], ts, ask, sym, notion)
            if path:
                path.update({"symbol": sym, "ext": ms["range_4h_pct"], "setup": EARLY_TREND})
                buckets["early"].append(path)
                buckets["combined"].append(path)
        if clf["setup_class"] == STRUCTURED_PULLBACK:
            req = float(clf["pullback_bps_req"])
            rec = float(clf["reclaim_bps_req"])
            low = ask * (1.0 - req / 1e4)
            trig = low * (1.0 + rec / 1e4)
            hit = None
            for bts, _o, high, lo, _c, _v in bars[sym]:
                if bts < ts or bts > ts + 900:
                    continue
                if lo <= low and high >= trig:
                    hit = min(high, max(lo, trig))
                    fill_ts = bts
                    break
            if hit:
                path = post_fill_path(bars[sym], fill_ts, hit, sym, notion)
                if path:
                    path.update({"symbol": sym, "ext": ms["range_4h_pct"], "setup": STRUCTURED_PULLBACK})
                    buckets["pullback"].append(path)
                    buckets["combined"].append(path)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "intents": len(intents),
        "filled_live": sum(1 for r in intents if r.get("status") == "FILLED"),
        "class_counts": dict(classes),
        "late_medians": {},
        "arms": {k: equity(v) for k, v in buckets.items()},
        "by_coin": {},
    }
    for key in ("range_4h_pct", "dist_high_240_bps", "ret_15", "ret_60", "ret_240", "atr_bps"):
        xs = sorted(float(r[key]) for r in late if r.get(key) is not None)
        report["late_medians"][key] = round(xs[len(xs) // 2], 4) if xs else None
    for arm, rows in buckets.items():
        report["by_coin"][arm] = {s: equity([x for x in rows if x["symbol"] == s]) for s in SYMBOLS}
    # walk-forward: first/second half by arm time
    mid = sorted(r["arm_ts"] for r in late)[len(late) // 2] if late else 0
    report["oos"] = {
        k: {
            "first": equity([x for x, src in zip(v, late, strict=False) if src["arm_ts"] < mid]),
            "second": equity([x for x, src in zip(v, late, strict=False) if src["arm_ts"] >= mid]),
        }
        for k, v in buckets.items()
    }
    report["eth_sol_examples"] = [r for r in late if r["intent_id"] in {"tb38b5ac8f26a849fc", "tb07444b2bcd784900"} or r["symbol"] in {"ETHUSDT", "SOLUSDT"}][:6]
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("intents", "filled_live", "class_counts", "late_medians", "arms")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
