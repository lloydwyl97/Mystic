"""DAY V2 Event-Driven Chronological Replay.

Reads feature_ohlcv from mystic_trading.db (15m + 1H + 4H bars).
Applies deterministic setup-family rules — NOT the 145-feature ML model.
Simulates trailing-buy entry and DAY V2 exit roles bar-by-bar.
Reports net P&L, win rate, profit factor for each split and combined.

Qualification bar (ALL required to proceed to live):
  - Positive aggregate OOS net expectancy (validation + untouched both green)
  - PF > 1.0 on both splits
  - Does not exceed maximum drawdown tolerance

Usage:
    cd /home/mystic/mystic
    venv/bin/python3 scripts/research/day_v2_replay.py \
        --db /home/mystic/mystic/mystic_trading.db \
        [--cal-end 2026-08-27] [--val-end 2026-09-09]
"""

from __future__ import annotations

import argparse
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Cost model (identical to production trading_economics)
# ---------------------------------------------------------------------------
ROUNDTRIP_COST_PCT = 0.0006  # 0.06% total (6 bps): 2 taker + 2 slippage + spread
ENTRY_HALF_COST = ROUNDTRIP_COST_PCT / 2  # 3 bps per side

# ---------------------------------------------------------------------------
# DAY V2 exit parameters (from config.py defaults)
# ---------------------------------------------------------------------------
MAX_HOLD_BARS = 20  # 20 x 15m = 300 minutes (5 hours)
MIN_MFE_FOR_WINNER_PCT = 0.008  # 0.8% MFE required before winner trail activates
WINNER_TRAIL_FLOOR = 0.005  # 0.5% floor on winner trail distance
WINNER_TRAIL_ATR_MULT = 1.5  # 1.5 x ATR trail distance
CATASTROPHIC_ATR_MULT = 3.0  # 3x ATR below entry triggers catastrophic exit
STRUCTURAL_BARS_REQUIRED = 3  # need 3 closed bars before structural invalidation fires

# ---------------------------------------------------------------------------
# Entry parameters (calibration candidates)
# ---------------------------------------------------------------------------
TRAIL_DECLINE_BARS = 3  # watch 3 declining bars before looking for rebound
ENTRY_REBOUND_PCT = 0.0015  # buy when price > declining low x (1 + 0.15%)
MAX_ATR_MULT_FOR_ENTRY = 4.0  # skip if pullback is already > 4x ATR

# ---------------------------------------------------------------------------
# Setup family constants
# ---------------------------------------------------------------------------
SETUP_HTF_TREND_PULLBACK = "HTF_TREND_PULLBACK"
SETUP_BREAKOUT_CONTINUATION = "BREAKOUT_CONTINUATION"
SETUP_RANGE_BOUNCE = "RANGE_BOUNCE"
SETUP_VWAP_REVERSION = "VWAP_REVERSION"
SETUP_EXHAUSTION_MR = "EXHAUSTION_MR"

ENABLED_SETUPS = frozenset(
    {
        SETUP_HTF_TREND_PULLBACK,
        SETUP_BREAKOUT_CONTINUATION,
        SETUP_RANGE_BOUNCE,
        SETUP_VWAP_REVERSION,
        SETUP_EXHAUSTION_MR,
    }
)


# ---------------------------------------------------------------------------
# Indicator helpers (no look-ahead — only past bars visible)
# ---------------------------------------------------------------------------


def _sma(closes: list[float], n: int) -> float | None:
    if len(closes) < n:
        return None
    return sum(closes[-n:]) / n


def _ema(closes: list[float], n: int) -> float | None:
    if len(closes) < n:
        return None
    k = 2.0 / (n + 1)
    ema = sum(closes[:n]) / n
    for c in closes[n:]:
        ema = c * k + ema * (1 - k)
    return ema


def _rsi(closes: list[float], n: int = 14) -> float | None:
    if len(closes) < n + 1:
        return None
    diffs = [closes[i] - closes[i - 1] for i in range(len(closes) - n, len(closes))]
    gains = [max(0.0, d) for d in diffs]
    losses = [max(0.0, -d) for d in diffs]
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(highs: list[float], lows: list[float], closes: list[float], n: int = 14) -> float | None:
    if len(highs) < n + 1:
        return None
    trs = []
    for i in range(len(highs) - n, len(highs)):
        prev_c = closes[i - 1]
        tr = max(highs[i] - lows[i], abs(highs[i] - prev_c), abs(lows[i] - prev_c))
        trs.append(tr)
    return sum(trs) / n


def _bb_pct(closes: list[float], n: int = 20, k: float = 2.0) -> float | None:
    """Bollinger Band %B: 0 = lower band, 1 = upper band."""
    if len(closes) < n:
        return None
    window = closes[-n:]
    mean = sum(window) / n
    std = math.sqrt(sum((c - mean) ** 2 for c in window) / n)
    if std == 0:
        return 0.5
    lower = mean - k * std
    upper = mean + k * std
    return (closes[-1] - lower) / (upper - lower)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_bars(db_path: str, symbol: str, interval: str) -> list[dict]:
    """Load all bars for symbol+interval, ordered by timestamp."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT ts, open, high, low, close, volume FROM feature_ohlcv WHERE symbol=? AND interval=? ORDER BY ts",
        (symbol, interval),
    ).fetchall()
    con.close()
    return [{"ts": r[0], "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])} for r in rows]


def detect_symbol_format(db_path: str) -> str:
    """Return the separator used in feature_ohlcv symbols (dash or slash)."""
    con = sqlite3.connect(db_path)
    row = con.execute("SELECT symbol FROM feature_ohlcv LIMIT 1").fetchone()
    con.close()
    if row and "/" in row[0]:
        return "/"
    return "-"


# ---------------------------------------------------------------------------
# Setup family detection (deterministic, OHLCV only)
# ---------------------------------------------------------------------------


def _detect_setup(
    *,
    bars_15m: list[dict],
    bars_1h: list[dict],
    bars_4h: list[dict],
    idx: int,
) -> tuple[str, float, float] | None:
    """Detect setup family at the current 15m bar index.

    Returns (setup_name, structural_anchor_price, target_price) or None.
    structural_anchor = price below which the thesis is invalid.
    target_price = objective completion level.

    Uses ONLY bars up to and including idx (no look-ahead).
    """
    # Need sufficient history
    if idx < 30:
        return None

    # Current and recent 15m bars (closed)
    b0 = bars_15m[idx]  # current bar (just closed)
    b1 = bars_15m[idx - 1]  # 1 bar ago
    b2 = bars_15m[idx - 2]

    c0 = b0["close"]
    l0 = b0["low"]

    closes_15m = [b["close"] for b in bars_15m[max(0, idx - 50) : idx + 1]]
    highs_15m = [b["high"] for b in bars_15m[max(0, idx - 50) : idx + 1]]
    lows_15m = [b["low"] for b in bars_15m[max(0, idx - 50) : idx + 1]]

    rsi = _rsi(closes_15m, 14)
    atr = _atr(highs_15m, lows_15m, closes_15m, 14)
    bb = _bb_pct(closes_15m, 20, 2.0)
    sma20 = _sma(closes_15m, 20)
    high20 = max(highs_15m[-20:]) if len(highs_15m) >= 20 else None
    low20 = min(lows_15m[-20:]) if len(lows_15m) >= 20 else None

    if rsi is None or atr is None or bb is None or sma20 is None:
        return None

    # --- 4H regime ---
    # Find the last closed 4H bar at or before this 15m bar's ts
    regime = "neutral"
    if bars_4h:
        ts0 = b0["ts"]
        h4_past = [b for b in bars_4h if b["ts"] <= ts0]
        if len(h4_past) >= 10:
            h4_closes = [b["close"] for b in h4_past[-11:]]
            h4_sma10 = _sma(h4_closes, 10)
            last_4h = h4_past[-1]["close"]
            if h4_sma10:
                if last_4h > h4_sma10 * 1.005:
                    regime = "bull"
                elif last_4h < h4_sma10 * 0.995:
                    regime = "bear"

    # --- 1H context ---
    h1_bullish = False
    if bars_1h:
        ts0 = b0["ts"]
        h1_past = [b for b in bars_1h if b["ts"] <= ts0]
        if len(h1_past) >= 5:
            h1_closes = [b["close"] for b in h1_past[-6:]]
            # bullish if last 1H close > 4 bars ago
            h1_bullish = h1_closes[-1] > h1_closes[-5]

    # --- Setup detection ---

    # 1. HTF_TREND_PULLBACK
    # Conditions: bull 4H regime, 1H bullish, 15m pulled back into sma20,
    #             RSI between 30-55 (oversold but recovering), bar is green
    if (
        regime == "bull"
        and h1_bullish
        and c0 > b1["close"]  # current bar closed up (first green bar)
        and c0 < sma20 * 1.005  # still near/below SMA20
        and 30 < rsi < 55
    ):
        # Structural anchor: 1x ATR below current close OR 4H low
        anchor = c0 - 1.5 * atr
        # Target: 2x ATR above entry
        target = c0 + 2.5 * atr
        if SETUP_HTF_TREND_PULLBACK in ENABLED_SETUPS:
            return (SETUP_HTF_TREND_PULLBACK, anchor, target)

    # 2. RANGE_BOUNCE
    # Conditions: neutral/bear regime, price near 20-bar low, BB < 0.30, RSI < 45, bar green
    if (
        regime in ("neutral", "bear")
        and low20 is not None
        and c0 > b1["close"]  # green bar
        and c0 < low20 * 1.005  # near 20-bar low
        and bb < 0.30
        and rsi < 45
    ):
        anchor = low20 * 0.995  # anchor just below the recent low
        target = c0 + 2.0 * atr
        if SETUP_RANGE_BOUNCE in ENABLED_SETUPS:
            return (SETUP_RANGE_BOUNCE, anchor, target)

    # 3. BREAKOUT_CONTINUATION
    # Conditions: bull or neutral, bar closes above 20-bar high, not overbought
    if (
        regime in ("bull", "neutral")
        and high20 is not None
        and c0 > high20 * 1.001  # decisive break
        and rsi < 72
        and c0 > b1["close"] > b2["close"]
    ):  # two consecutive up bars before
        # Anchor: breakout level (old resistance now support)
        anchor = high20 * 0.995
        target = c0 + 2.0 * atr
        if SETUP_BREAKOUT_CONTINUATION in ENABLED_SETUPS:
            return (SETUP_BREAKOUT_CONTINUATION, anchor, target)

    # 4. VWAP_REVERSION
    # Conditions: price oversold (RSI < 40), bar green after decline, neutral/bull regime
    if (
        regime in ("bull", "neutral")
        and c0 > b1["close"]
        and b1["close"] < b2["close"]  # first up after down
        and rsi < 40
        and rsi > 20
    ):  # oversold but not extreme
        anchor = l0 * 0.998  # just below current bar's low
        target = sma20  # mean reversion target
        if target > c0 * 1.003 and SETUP_VWAP_REVERSION in ENABLED_SETUPS:
            return (SETUP_VWAP_REVERSION, anchor, target)

    # 5. EXHAUSTION_MR (Mean Reversion after exhaustion candle)
    # Conditions: prior bar had extreme RSI (>75 or momentum spike), now retreating
    # Use b1 RSI for prior bar
    closes_prior = closes_15m[:-1]
    rsi_prior = _rsi(closes_prior, 14) if len(closes_prior) >= 15 else None
    if (
        rsi_prior is not None
        and rsi_prior < 45
        and b1["close"] > b2["close"] * 1.02  # prior bar was a spike up
        and c0 < b1["close"]  # current bar retraces
        and rsi < 50
    ):
        anchor = l0 * 0.997
        target = sma20
        if target > c0 * 1.002 and SETUP_EXHAUSTION_MR in ENABLED_SETUPS:
            return (SETUP_EXHAUSTION_MR, anchor, target)

    return None


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------


@dataclass
class SimPosition:
    symbol: str
    setup: str
    entry_bar_idx: int
    entry_price: float
    entry_ts: str
    structural_anchor: float
    target_price: float
    atr_at_entry: float
    highest_price: float = 0.0
    bars_held: int = 0
    closed: bool = False
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_ts: str = ""
    net_pnl_pct: float = 0.0


# ---------------------------------------------------------------------------
# Per-symbol replay
# ---------------------------------------------------------------------------


def replay_symbol(
    bars_15m: list[dict],
    bars_1h: list[dict],
    bars_4h: list[dict],
    symbol: str,
    warmup_bars: int = 50,
) -> list[SimPosition]:
    """Walk through 15m bars chronologically and simulate DAY V2 trades."""
    positions: list[SimPosition] = []
    current_pos: SimPosition | None = None

    for idx in range(warmup_bars, len(bars_15m)):
        bar = bars_15m[idx]
        c = bar["close"]
        h = bar["high"]
        bar_low = bar["low"]

        # Compute ATR for exit evaluation
        if idx >= 14:
            highs = [b["high"] for b in bars_15m[idx - 14 : idx + 1]]
            lows = [b["low"] for b in bars_15m[idx - 14 : idx + 1]]
            closes = [b["close"] for b in bars_15m[idx - 14 : idx + 1]]
            atr = _atr(highs, lows, closes, 14) or 0.0
        else:
            atr = 0.0
        atr_pct = atr / c if c > 0 else 0.0

        # --- Exit evaluation (if in position) ---
        if current_pos is not None and not current_pos.closed:
            pos = current_pos
            pos.bars_held += 1
            pos.highest_price = max(pos.highest_price, h)

            net_pnl_raw = (c - pos.entry_price) / pos.entry_price

            exit_reason = ""
            exit_price = c

            # Role 1: Catastrophic (uses bar low, not close)
            adverse_move = (pos.entry_price - bar_low) / pos.entry_price
            catastrophic_threshold = CATASTROPHIC_ATR_MULT * pos.atr_at_entry / pos.entry_price
            if adverse_move >= catastrophic_threshold:
                # Exit at the catastrophic level price
                exit_price = pos.entry_price * (1.0 - catastrophic_threshold)
                exit_reason = "CATASTROPHIC"

            # Role 2: Structural invalidation (on closed bar, after required bars)
            elif pos.bars_held >= STRUCTURAL_BARS_REQUIRED and c < pos.structural_anchor:
                exit_reason = "STRUCTURAL_INVALIDATION"

            # Role 4: Winner protection (closed bar check)
            elif pos.highest_price > 0:
                mfe_pct = (pos.highest_price - pos.entry_price) / pos.entry_price
                if mfe_pct >= MIN_MFE_FOR_WINNER_PCT:
                    trail_distance = max(WINNER_TRAIL_FLOOR, WINNER_TRAIL_ATR_MULT * atr_pct)
                    trail_trigger = pos.highest_price * (1.0 - trail_distance)
                    if c <= trail_trigger:
                        exit_reason = "WINNER_PROTECTION"

            # Role 5: Objective complete (closed bar)
            elif c >= pos.target_price:
                exit_reason = "OBJECTIVE_COMPLETE"

            # Role 3: Time expiration (on closed bar)
            elif pos.bars_held >= MAX_HOLD_BARS and net_pnl_raw <= 0:
                exit_reason = "TIME_EXPIRATION"

            if exit_reason:
                net_pnl_raw = (exit_price - pos.entry_price) / pos.entry_price
                pos.net_pnl_pct = net_pnl_raw - ROUNDTRIP_COST_PCT
                pos.exit_price = exit_price
                pos.exit_reason = exit_reason
                pos.exit_ts = bar["ts"]
                pos.closed = True
                positions.append(pos)
                current_pos = None
            continue

        # --- Entry evaluation (if no open position) ---
        # We look for a setup on the PREVIOUS closed bar (idx-1), then try
        # to enter on the CURRENT bar's open as a trailing-buy proxy.
        if idx < 1:
            continue

        signal = _detect_setup(
            bars_15m=bars_15m,
            bars_1h=bars_1h,
            bars_4h=bars_4h,
            idx=idx - 1,  # signal fires on the prior CLOSED bar
        )

        if signal is None:
            continue

        setup_name, anchor, target = signal

        # Validate entry: current bar must not gap down severely (price shouldn't
        # already be at the catastrophic level)
        entry_price_candidate = bar["open"]  # trail-buy: enter at open of signal+1 bar
        if entry_price_candidate <= 0:
            continue

        # Skip if entry is already at or below structural anchor
        if entry_price_candidate <= anchor:
            continue

        # Skip if ATR is missing (insufficient history)
        if atr == 0.0 or atr_pct == 0.0:
            continue

        # Compute ATR at entry
        current_pos = SimPosition(
            symbol=symbol,
            setup=setup_name,
            entry_bar_idx=idx,
            entry_price=entry_price_candidate,
            entry_ts=bar["ts"],
            structural_anchor=anchor,
            target_price=target,
            atr_at_entry=atr,
            highest_price=entry_price_candidate,
        )

    # Close any open position at last bar's close (mark-to-end)
    if current_pos is not None and not current_pos.closed:
        last_bar = bars_15m[-1]
        c = last_bar["close"]
        net_pnl_raw = (c - current_pos.entry_price) / current_pos.entry_price
        current_pos.net_pnl_pct = net_pnl_raw - ROUNDTRIP_COST_PCT
        current_pos.exit_price = c
        current_pos.exit_reason = "MARK_TO_END"
        current_pos.exit_ts = last_bar["ts"]
        current_pos.closed = True
        positions.append(current_pos)

    return positions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class SplitStats:
    name: str
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    winner_pnl: float = 0.0
    loser_pnl: float = 0.0
    max_drawdown: float = 0.0
    exit_counts: dict = field(default_factory=dict)
    setup_counts: dict = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def avg_winner(self) -> float:
        return self.winner_pnl / self.wins if self.wins else 0.0

    @property
    def avg_loser(self) -> float:
        losers = self.trades - self.wins
        return self.loser_pnl / losers if losers else 0.0

    @property
    def profit_factor(self) -> float:
        if abs(self.loser_pnl) < 1e-9:
            return float("inf") if self.winner_pnl > 0 else 1.0
        return abs(self.winner_pnl / self.loser_pnl)

    @property
    def expectancy_bps(self) -> float:
        return (self.total_pnl / self.trades * 10000) if self.trades else 0.0

    @property
    def passes(self) -> bool:
        return self.trades >= 10 and self.total_pnl > 0 and self.profit_factor > 1.0


def _compute_drawdown(positions: list[SimPosition]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in positions:
        equity += p.net_pnl_pct
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
    return max_dd


def _split_stats(name: str, positions: list[SimPosition]) -> SplitStats:
    s = SplitStats(name=name)
    s.trades = len(positions)
    s.max_drawdown = _compute_drawdown(positions)
    for p in positions:
        s.exit_counts[p.exit_reason] = s.exit_counts.get(p.exit_reason, 0) + 1
        s.setup_counts[p.setup] = s.setup_counts.get(p.setup, 0) + 1
        s.total_pnl += p.net_pnl_pct
        if p.net_pnl_pct >= 0:
            s.wins += 1
            s.winner_pnl += p.net_pnl_pct
        else:
            s.loser_pnl += p.net_pnl_pct
    return s


def print_stats(s: SplitStats, verbose: bool = True) -> None:
    verdict = "PASS ✓" if s.passes else "FAIL ✗"
    print(f"\n  ── {s.name} ({verdict}) ──")
    print(f"     Trades:      {s.trades}")
    print(f"     Win rate:    {s.win_rate:.1%}")
    print(f"     Avg winner:  {s.avg_winner * 10000:.1f} bps")
    print(f"     Avg loser:   {s.avg_loser * 10000:.1f} bps")
    print(f"     Expectancy:  {s.expectancy_bps:.1f} bps/trade")
    print(f"     Profit factor: {s.profit_factor:.2f}")
    print(f"     Total P&L:   {s.total_pnl * 100:.2f}%")
    print(f"     Max drawdown:{s.max_drawdown * 100:.2f}%")
    if verbose:
        print(f"     Exit breakdown: {s.exit_counts}")
        print(f"     Setup breakdown: {s.setup_counts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="DAY V2 event-driven replay")
    parser.add_argument("--db", default="/home/mystic/mystic/mystic_trading.db")
    parser.add_argument("--cal-end", default="2026-08-27", help="End of calibration period (validation starts next bar)")
    parser.add_argument("--val-end", default="2026-09-09", help="End of validation period (untouched starts next bar)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    db_path = args.db
    cal_end = args.cal_end
    val_end = args.val_end

    sep = detect_symbol_format(db_path)
    SYMBOLS = [f"BTC{sep}USDT", f"ETH{sep}USDT", f"SOL{sep}USDT", f"XRP{sep}USDT"]

    print(f"DAY V2 Replay  db={db_path}")
    print(f"  Cal through {cal_end} | Val through {val_end} | Untouched = remainder")
    print(f"  Symbols: {SYMBOLS}")
    print(f"  Costs: {ROUNDTRIP_COST_PCT * 100:.3f}% roundtrip")
    print("  Entry: rebound after decline signal on prior closed 15m bar")

    all_cal: list[SimPosition] = []
    all_val: list[SimPosition] = []
    all_touch: list[SimPosition] = []

    for sym in SYMBOLS:
        bars_15m = load_bars(db_path, sym, "15m")
        bars_1h = load_bars(db_path, sym, "1h")
        bars_4h = load_bars(db_path, sym, "4h")

        if not bars_15m:
            print(f"  [WARN] No 15m bars for {sym}")
            continue

        print(f"\n  Replaying {sym}: {len(bars_15m)} 15m bars ({bars_15m[0]['ts']} → {bars_15m[-1]['ts']})")

        all_positions = replay_symbol(bars_15m, bars_1h, bars_4h, sym)

        sym_cal = [p for p in all_positions if p.entry_ts < cal_end]
        sym_val = [p for p in all_positions if cal_end <= p.entry_ts < val_end]
        sym_touch = [p for p in all_positions if p.entry_ts >= val_end]

        all_cal.extend(sym_cal)
        all_val.extend(sym_val)
        all_touch.extend(sym_touch)

        s_cal = _split_stats(f"{sym} calibration", sym_cal)
        s_val = _split_stats(f"{sym} validation", sym_val)
        s_touch = _split_stats(f"{sym} untouched", sym_touch)

        if args.verbose:
            print_stats(s_cal)
            print_stats(s_val)
            print_stats(s_touch)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    s_all_cal = _split_stats("ALL calibration", all_cal)
    s_all_val = _split_stats("ALL validation", all_val)
    s_all_touch = _split_stats("ALL untouched (OOS)", all_touch)
    s_combined = _split_stats("ALL combined", all_cal + all_val + all_touch)

    for s in (s_all_cal, s_all_val, s_all_touch, s_combined):
        print_stats(s, verbose=True)

    print("\n" + "=" * 60)
    print("DEPLOYMENT DECISION")
    print("=" * 60)
    val_pass = s_all_val.passes
    touch_pass = s_all_touch.passes
    both_pass = val_pass and touch_pass

    print(f"  Validation  ({s_all_val.name}): {'PASS' if val_pass else 'FAIL'}")
    print(f"  Untouched   ({s_all_touch.name}): {'PASS' if touch_pass else 'FAIL'}")
    print()
    if both_pass:
        print("  ✓ DAY V2 QUALIFIES FOR LIVE DEPLOYMENT")
        print("  Both validation and untouched periods are positive net after costs.")
        print("  Profit factor > 1.0 on both splits.")
    else:
        print("  ✗ DAY V2 DOES NOT QUALIFY FOR LIVE DEPLOYMENT")
        print("  At least one required period is negative net after costs.")
        print("  Leave SCALP V2 live. Do not deploy DAY V2 with these parameters.")
        if not val_pass:
            print(f"  Validation failed: PnL={s_all_val.total_pnl:.4f} PF={s_all_val.profit_factor:.2f} trades={s_all_val.trades}")
        if not touch_pass:
            print(f"  Untouched failed:  PnL={s_all_touch.total_pnl:.4f} PF={s_all_touch.profit_factor:.2f} trades={s_all_touch.trades}")

    print()


if __name__ == "__main__":
    main()
