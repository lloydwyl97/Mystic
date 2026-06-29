#!/usr/bin/env python3
"""Phase 3l — deterministic replay of Phase 3f exits through fixed exit logic (read-only)."""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot
from backend.services.binance_scalp.orderbook_book import walk_buy_notional, walk_sell_qty
from backend.services.binance_scalp.protected_preflight import run_scalp_preflight

DB = REPO / "mystic_trading.db"
PHASE3F_REPORT = Path("/tmp/scalp_phase3f/final_report.json")
PHASE3G_AUDIT = Path("/tmp/scalp_phase3g_audit.json")
OUT_DEFAULT = Path("/tmp/scalp_phase3l_replay.json")
SOAK_START = "2026-06-07 00:12:04"
LOOP_INTERVAL_SEC = 5.0
KLINE_PROXY_WARNING = "1m kline HIGH used as best_bid proxy — optimistic; live depth not persisted per tick"


@dataclass(frozen=True)
class ReplayResult:
    trade_id: str
    symbol: str
    entry_time: str
    original_exit_reason: str
    original_pnl_usd: float
    entry_price: float
    qty: float
    entry_buy_impact_pct: float
    spread_at_entry_pct: float
    target_bid_required_new: float
    best_executable_bid_proxy: float
    old_gate_missed_target: bool
    new_exit_reason: str
    new_exit_time_utc: str | None
    replay_pnl_usd: float | None
    stale_would_still_happen: bool
    old_exit_time_utc: str | None
    bid_source: str
    ticks_simulated: int
    max_new_net_pct: float
    max_old_net_pct: float
    phase3g_classification: str | None

    def as_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "entry_time": self.entry_time,
            "original_exit_reason": self.original_exit_reason,
            "original_pnl_usd": self.original_pnl_usd,
            "entry_price": self.entry_price,
            "qty": self.qty,
            "entry_buy_impact_pct": self.entry_buy_impact_pct,
            "spread_at_entry_pct": self.spread_at_entry_pct,
            "target_bid_required_new": self.target_bid_required_new,
            "best_executable_bid_proxy": self.best_executable_bid_proxy,
            "old_gate_missed_target": self.old_gate_missed_target,
            "new_exit_reason": self.new_exit_reason,
            "new_exit_time_utc": self.new_exit_time_utc,
            "replay_pnl_usd": self.replay_pnl_usd,
            "stale_would_still_happen": self.stale_would_still_happen,
            "old_exit_time_utc": self.old_exit_time_utc,
            "bid_source": self.bid_source,
            "ticks_simulated": self.ticks_simulated,
            "max_new_net_pct": self.max_new_net_pct,
            "max_old_net_pct": self.max_old_net_pct,
            "phase3g_classification": self.phase3g_classification,
        }


def _parse_ts(s: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(str(s)[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(s)


def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            rows = json.loads(resp.read().decode())
    except Exception:
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "20", url],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        rows = json.loads(proc.stdout)
    if isinstance(rows, dict):
        return []
    return [
        {
            "open_time_ms": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
        }
        for r in rows
    ]


def _load_trade_context(buy: sqlite3.Row, sell: sqlite3.Row, conn: sqlite3.Connection) -> dict:
    sell_diag = json.loads(sell["diagnostics_json"] or "{}")
    sell_pf = sell_diag.get("preflight", {})
    pos = conn.execute(
        "SELECT diagnostics_json FROM scalp_paper_positions WHERE trade_id=?",
        (buy["trade_id"],),
    ).fetchone()
    pos_raw = json.loads(pos["diagnostics_json"] or "{}") if pos else {}
    buy_pf = (pos_raw or {}).get("entry_preflight") or json.loads(buy["diagnostics_json"] or "{}").get("preflight", {})
    spread_e = float(buy_pf.get("spread_pct", sell_pf.get("spread_pct", 0)))
    buy_i = float(buy_pf.get("buy_impact_pct", 0))
    sell_i = float(buy_pf.get("sell_impact_pct", 0))
    return {
        "spread_at_entry": spread_e,
        "entry_buy_impact_pct": buy_i,
        "sell_impact_at_entry": sell_i,
        "sell_pf_at_actual_exit": sell_pf,
    }


def _snapshot_from_bid(
    symbol: str,
    best_bid: float,
    spread_pct: float,
    qty: float,
) -> MarketSnapshot:
    """Synthetic depth: flat book at bid with depth for qty walk."""
    best_ask = best_bid * (1.0 + spread_pct) if best_bid > 0 else 0.0
    mid = (best_bid + best_ask) / 2.0 if best_bid > 0 else 0.0
    depth_qty = max(qty * 20.0, 1.0)
    bids = [[best_bid, depth_qty]]
    asks = [[best_ask, depth_qty]]
    return MarketSnapshot(
        symbol=symbol,
        symbol_bus=symbol,
        best_bid=best_bid,
        best_ask=best_ask,
        mid=mid,
        spread_pct=spread_pct,
        bids=bids,
        asks=asks,
        redis_spread_pct=None,
        order_book_imbalance=0.0,
        book_source="kline_bid_proxy_replay",
        orderbook_age_sec=0.0,
    )


def _target_bid_new(
    entry: float,
    entry_buy_impact: float,
    sell_impact: float,
    econ: ScalpEconomics,
) -> float:
    """Minimum sell fill for executable_exit_net_pct >= target."""
    if entry <= 0:
        return 0.0
    fixed_costs = econ.entry_fee_pct() + econ.exit_fee_pct() + econ.slippage_buffer_pct * 2.0 + entry_buy_impact + sell_impact
    return entry * (1.0 + econ.net_profit_target_pct + fixed_costs)


def _old_sell_gate(
    snap: MarketSnapshot,
    econ: ScalpEconomics,
    *,
    entry_price: float,
    qty: float,
    notional_usd: float,
) -> tuple[bool, float, float]:
    """Pre-3g exit gate: roundtrip_cost_pct incl. spread + current-book buy impact."""
    buy_walk = walk_buy_notional(snap.asks, notional_usd, snap.best_ask)
    buy_impact = buy_walk.impact_pct
    sw = walk_sell_qty(snap.bids, qty, snap.best_bid)
    if not sw.depth_sufficient:
        return False, -1.0, 0.0
    sell_fill = sw.expected_avg_fill if sw.expected_avg_fill > 0 else snap.best_bid
    gross = (sell_fill - entry_price) / entry_price if entry_price > 0 else -1.0
    costs = econ.roundtrip_cost_pct(snap.spread_pct, buy_impact, sw.impact_pct)
    net = gross - costs
    passed = net >= econ.net_profit_target_pct
    return passed, net, sell_fill


def _new_sell_gate(
    snap: MarketSnapshot,
    econ: ScalpEconomics,
    config,
    *,
    entry_price: float,
    qty: float,
    entry_buy_impact_pct: float,
) -> tuple[bool, float, float]:
    pf = run_scalp_preflight(
        snap,
        econ,
        config,
        side="SELL",
        entry_price=entry_price,
        entry_buy_impact_pct=entry_buy_impact_pct,
        quantity=qty,
        check_paper_enabled=False,
    )
    sell_fill = pf.expected_avg_fill if pf.expected_avg_fill > 0 else pf.limit_sell_price
    return pf.passed, pf.expected_net_edge_pct, sell_fill


def _replay_pnl_usd(
    entry: float,
    exit_price: float,
    qty: float,
    econ: ScalpEconomics,
) -> float:
    notional = qty * exit_price
    fee = notional * econ.taker_fee_pct
    slip = notional * econ.slippage_buffer_pct
    return (exit_price - entry) * qty - fee - slip - (entry * qty * econ.taker_fee_pct)


def _bid_for_tick(klines: list[dict], tick_epoch: float) -> float | None:
    tick_ms = int(tick_epoch * 1000)
    for k in klines:
        start = k["open_time_ms"]
        end = start + 60_000
        if start <= tick_ms < end:
            return k["high"]
    return None


def replay_trade(
    buy: sqlite3.Row,
    sell: sqlite3.Row,
    ctx: dict,
    econ: ScalpEconomics,
    config,
    *,
    phase3g_by_id: dict[str, dict],
) -> ReplayResult:
    entry = float(buy["price"])
    qty = float(buy["quantity"])
    sym = str(buy["symbol"])
    entry_ts = _parse_ts(buy["created_at"])
    exit_ts = _parse_ts(sell["created_at"])
    entry_buy_i = float(ctx["entry_buy_impact_pct"])
    spread_e = float(ctx["spread_at_entry"])
    sell_i_e = float(ctx["sell_impact_at_entry"])
    notional = entry * qty

    start_ms = int(entry_ts.timestamp() * 1000) - 60_000
    end_ms = int(exit_ts.timestamp() * 1000) + 120_000
    klines = _fetch_klines(sym, start_ms, end_ms)

    target_bid = _target_bid_new(entry, entry_buy_i, sell_i_e, econ)
    best_bid = entry
    max_new = -999.0
    max_old = -999.0
    new_exit_time: str | None = None
    old_exit_time: str | None = None
    replay_pnl: float | None = None
    replay_exit_reason = "STALE_SCALP_TIMEOUT"
    ticks = 0

    entry_epoch = entry_ts.timestamp()
    deadline = entry_epoch + econ.stale_scalp_timeout_sec + LOOP_INTERVAL_SEC
    t = entry_epoch
    while t <= deadline:
        bid = _bid_for_tick(klines, t)
        if bid is not None:
            best_bid = max(best_bid, bid)
            snap = _snapshot_from_bid(sym, bid, spread_e, qty)
            new_pass, new_net, new_fill = _new_sell_gate(
                snap,
                econ,
                config,
                entry_price=entry,
                qty=qty,
                entry_buy_impact_pct=entry_buy_i,
            )
            old_pass, old_net, old_fill = _old_sell_gate(
                snap,
                econ,
                entry_price=entry,
                qty=qty,
                notional_usd=notional,
            )
            max_new = max(max_new, new_net)
            max_old = max(max_old, old_net)
            age = t - entry_epoch
            stale = age >= econ.stale_scalp_timeout_sec

            if new_pass and new_exit_time is None:
                new_exit_time = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                replay_exit_reason = "NET_PROFIT_TARGET"
                replay_pnl = _replay_pnl_usd(entry, new_fill, qty, econ)
                break
            if old_pass and old_exit_time is None:
                old_exit_time = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if stale and not new_pass:
                replay_pnl = _replay_pnl_usd(
                    entry,
                    old_fill if old_fill > 0 else bid,
                    qty,
                    econ,
                )
                break
        ticks += 1
        t += LOOP_INTERVAL_SEC

    orig_reason = str(sell["exit_reason"])
    p3g = phase3g_by_id.get(str(buy["trade_id"]), {})
    old_missed = orig_reason == "STALE_SCALP_TIMEOUT" and replay_exit_reason == "NET_PROFIT_TARGET" and p3g.get("classification") == "true_missed_exit_bug"
    stale_still = replay_exit_reason == "STALE_SCALP_TIMEOUT"

    return ReplayResult(
        trade_id=str(buy["trade_id"]),
        symbol=sym,
        entry_time=buy["created_at"],
        original_exit_reason=orig_reason,
        original_pnl_usd=float(sell["pnl_usd"] or 0),
        entry_price=entry,
        qty=qty,
        entry_buy_impact_pct=entry_buy_i,
        spread_at_entry_pct=spread_e,
        target_bid_required_new=target_bid,
        best_executable_bid_proxy=best_bid,
        old_gate_missed_target=old_missed,
        new_exit_reason=replay_exit_reason,
        new_exit_time_utc=new_exit_time,
        replay_pnl_usd=replay_pnl,
        stale_would_still_happen=stale_still,
        old_exit_time_utc=old_exit_time,
        bid_source=KLINE_PROXY_WARNING,
        ticks_simulated=ticks,
        max_new_net_pct=max_new,
        max_old_net_pct=max_old,
        phase3g_classification=p3g.get("classification"),
    )


def run_synthetic_tests(econ: ScalpEconomics, config) -> dict:
    """In-memory synthetic bid paths — no DB writes."""
    entry = 1000.0
    qty = 0.01
    entry_buy_i = 0.0001
    spread = 0.0002
    sell_i = 0.0
    target_bid = _target_bid_new(entry, entry_buy_i, sell_i, econ)

    # Path crosses target
    cross_bids = [entry, entry * 1.001, target_bid * 1.001, target_bid * 1.002]
    cross_exit = None
    for bid in cross_bids:
        snap = _snapshot_from_bid("ETHUSDT", bid, spread, qty)
        passed, net, fill = _new_sell_gate(
            snap,
            econ,
            config,
            entry_price=entry,
            qty=qty,
            entry_buy_impact_pct=entry_buy_i,
        )
        if passed:
            cross_exit = {"bid": bid, "net_pct": net, "fill": fill, "reason": "NET_PROFIT_TARGET"}
            break

    # Path stays below target until stale
    below_bids = [entry * 0.9995] * 80
    below_exit = None
    for i, bid in enumerate(below_bids):
        snap = _snapshot_from_bid("ETHUSDT", bid, spread, qty)
        passed, net, _ = _new_sell_gate(
            snap,
            econ,
            config,
            entry_price=entry,
            qty=qty,
            entry_buy_impact_pct=entry_buy_i,
        )
        age = (i + 1) * LOOP_INTERVAL_SEC
        if passed:
            below_exit = "NET_PROFIT_TARGET"
            break
        if age >= econ.stale_scalp_timeout_sec:
            below_exit = "STALE_SCALP_TIMEOUT"
            break

    return {
        "cross_target": {
            "passed": cross_exit is not None,
            "exit": cross_exit,
            "target_bid": target_bid,
        },
        "below_target": {
            "exit_reason": below_exit,
            "expected": "STALE_SCALP_TIMEOUT",
            "passed": below_exit == "STALE_SCALP_TIMEOUT",
        },
        "all_passed": cross_exit is not None and below_exit == "STALE_SCALP_TIMEOUT",
    }


def _day_snapshot() -> dict:
    with sqlite3.connect(DB) as c:
        led = c.execute("SELECT cash_balance, total_equity FROM portfolio_engine_ledger WHERE id=1").fetchone()
        xrp = c.execute("SELECT symbol, quantity, entry_price FROM portfolio_engine_positions WHERE symbol LIKE '%XRP%'").fetchall()
        pn = c.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
    ai = sorted(k for k in subprocess.check_output(["redis-cli", "KEYS", "ai_signal:day:*"], text=True).split() if k)
    try:
        health = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).status
    except Exception:
        health = 0
    cfg = get_scalp_config()
    return {
        "health": health,
        "ledger": {"cash_balance": led[0], "total_equity": led[1]} if led else None,
        "xrp": [{"symbol": r[0], "quantity": r[1], "entry_price": r[2]} for r in xrp],
        "paper_trades": pn,
        "ai_signal_day": ai,
        "SCALP_LIVE": cfg.scalp_live,
        "SCALP_PAPER_ENABLED": cfg.scalp_paper_enabled,
    }


def load_phase3g_map() -> dict[str, dict]:
    if not PHASE3G_AUDIT.exists():
        return {}
    data = json.loads(PHASE3G_AUDIT.read_text())
    return {t["trade_id"]: t for t in data.get("exit_timing_audits", [])}


def load_phase3f_summary() -> dict | None:
    if not PHASE3F_REPORT.exists():
        return None
    return json.loads(PHASE3F_REPORT.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay Phase 3f scalp exits (read-only)")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--soak-start", default=SOAK_START)
    args = parser.parse_args()

    day_before = _day_snapshot()
    econ = ScalpEconomics.from_env()
    config = get_scalp_config()
    assert not config.scalp_live, "SCALP_LIVE must be false"

    phase3g_map = load_phase3g_map()
    phase3f = load_phase3f_summary()

    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        buys = conn.execute(
            """
            SELECT * FROM scalp_paper_trades
            WHERE side='BUY' AND created_at >= ?
            ORDER BY id
            """,
            (args.soak_start,),
        ).fetchall()
        results: list[ReplayResult] = []
        for buy in buys:
            sell = conn.execute(
                "SELECT * FROM scalp_paper_trades WHERE trade_id=?",
                (f"{buy['trade_id']}_SELL",),
            ).fetchone()
            if not sell:
                continue
            ctx = _load_trade_context(buy, sell, conn)
            results.append(replay_trade(buy, sell, ctx, econ, config, phase3g_by_id=phase3g_map))

    synthetic = run_synthetic_tests(econ, config)
    day_after = _day_snapshot()

    eth_0514 = next(
        (r for r in results if r.symbol == "ETHUSDT" and "05:14" in r.entry_time),
        None,
    )
    eth_0628 = next(
        (r for r in results if r.symbol == "ETHUSDT" and "06:28" in r.entry_time),
        None,
    )

    missed_fixed = sum(1 for r in results if r.original_exit_reason == "STALE_SCALP_TIMEOUT" and r.new_exit_reason == "NET_PROFIT_TARGET")
    validated = synthetic["all_passed"] and eth_0514 is not None and eth_0628 is not None and eth_0514.new_exit_reason == "NET_PROFIT_TARGET" and eth_0628.new_exit_reason == "NET_PROFIT_TARGET"

    report = {
        "phase": "3l",
        "files_created": [
            str(REPO / "scripts/replay_scalp_exit_logic.py"),
            str(REPO / "tests/test_scalp_exit_replay.py"),
        ],
        "replay_data_sources": {
            "phase3f_report": str(PHASE3F_REPORT) if PHASE3F_REPORT.exists() else None,
            "phase3g_audit": str(PHASE3G_AUDIT) if PHASE3G_AUDIT.exists() else None,
            "db_readonly": str(DB),
            "klines": "binance.us api/v3/klines 1m",
            "bid_proxy_warning": KLINE_PROXY_WARNING,
        },
        "trades_replayed": [r.as_dict() for r in results],
        "eth_0514": eth_0514.as_dict() if eth_0514 else None,
        "eth_0628": eth_0628.as_dict() if eth_0628 else None,
        "old_vs_new_summary": {
            "total": len(results),
            "original_profit_target": sum(1 for r in results if r.original_exit_reason == "NET_PROFIT_TARGET"),
            "original_stale": sum(1 for r in results if r.original_exit_reason == "STALE_SCALP_TIMEOUT"),
            "new_profit_target": sum(1 for r in results if r.new_exit_reason == "NET_PROFIT_TARGET"),
            "new_stale": sum(1 for r in results if r.new_exit_reason == "STALE_SCALP_TIMEOUT"),
            "stale_to_profit_replays": missed_fixed,
            "phase3f_summary": {
                "trades_closed": phase3f.get("2_trades_closed") if phase3f else None,
                "pnl_usd": phase3f.get("4_pnl_usd") if phase3f else None,
            }
            if phase3f
            else None,
        },
        "synthetic_tests": synthetic,
        "phase3g_exit_fix_validated": validated,
        "day_untouched": {
            "health": day_after["health"],
            "ledger_ok": day_before["ledger"] == day_after["ledger"],
            "xrp_ok": day_before["xrp"] == day_after["xrp"],
            "paper_trades_ok": day_before["paper_trades"] == day_after["paper_trades"],
            "ai_ok": day_before["ai_signal_day"] == day_after["ai_signal_day"],
            "scalp_live": day_after["SCALP_LIVE"],
            "scalp_paper_enabled": day_after["SCALP_PAPER_ENABLED"],
        },
        "safe_to_continue": (
            day_after["health"] == 200
            and day_before["ledger"] == day_after["ledger"]
            and day_before["xrp"] == day_after["xrp"]
            and day_before["paper_trades"] == day_after["paper_trades"]
            and not day_after["SCALP_LIVE"]
            and not day_after["SCALP_PAPER_ENABLED"]
            and validated
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
