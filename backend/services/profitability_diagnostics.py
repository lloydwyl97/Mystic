"""
Profitability Diagnostic Sweep - Read-only instrumentation.

Adds metrics, logs, counters, and reports. No strategy changes.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Any

from backend.config.trading_economics import SLIPPAGE_BUFFER, TAKER_FEE

logger = logging.getLogger(__name__)

DIAG_VERSION = "1.0.0"

# Rolling windows
WINDOWS = (50, 100, 250)
PROFIT_SNAPSHOT_EVERY = int(os.getenv("PROFIT_SNAPSHOT_EVERY", "25"))
FEE_LEAKAGE_LOG_EVERY = int(os.getenv("FEE_LEAKAGE_LOG_EVERY", "20"))
CONFIDENCE_BUCKETS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
HOLD_BUCKETS = [(0, 60), (60, 180), (180, 300), (300, 600), (600, float("inf"))]  # seconds


@dataclass
class ClosedTradeRecord:
    """Per-trade record for diagnostics."""

    symbol: str
    gross_pnl_pct: float
    fees_paid_pct: float
    slippage_pct: float
    net_pnl_pct: float
    hold_seconds: float
    exit_reason: str
    entry_confidence: float
    volatility_at_entry: float
    spread_pct_at_entry: float
    mfe_pct: float
    mae_pct: float
    captured_pct: float | None
    bars_held: int | None = None


class ProfitabilityDiagnostics:
    """In-memory diagnostics; no DB writes."""

    def __init__(self) -> None:
        self._closed_trades: deque[ClosedTradeRecord] = deque(maxlen=1000)
        self._equity_peak: float = 0.0
        self._max_drawdown_pct: float = 0.0
        self._last_equity: float = 0.0
        self._recovery_start_equity: float = 0.0
        self._in_drawdown: bool = False
        self._closed_count_since_snapshot: int = 0
        self._closed_count_since_fee_log: int = 0
        # Missed opportunity counters
        self._skipped_cooldown: int = 0
        self._skipped_max_positions: int = 0
        self._skipped_min_notional: int = 0
        self._skipped_confidence: int = 0
        self._skipped_spread: int = 0
        self._skipped_low_vol: int = 0
        self._instant_stop_count: int = 0

    def on_trade_closed(
        self,
        *,
        symbol: str,
        quantity: float,
        entry_price: float,
        fill_price: float,
        realized_pnl: float,
        pnl_pct: float,
        fee: float,
        slippage_cost: float,
        hold_time_seconds: float,
        exit_type: str,
        exit_trigger: str,
        entry_confidence: float = 0.0,
        atr_at_entry: float = 0.0,
        highest_price: float = 0.0,
        lowest_price: float = 0.0,
        bar_interval_seconds: int | None = 60,
        spread_pct: float = 0.0,
        total_equity: float = 0.0,
        price_at_exit: float | None = None,
    ) -> None:
        """Called after each trade close. Compute and log diagnostic metrics."""
        entry_cost = quantity * entry_price
        if entry_cost <= 0:
            return

        # 1) Per-trade net PnL breakdown
        # gross = before costs (use limit price if available, else fill)
        exit_price = price_at_exit if price_at_exit and price_at_exit > 0 else fill_price
        gross_pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        entry_fee_est = quantity * entry_price * TAKER_FEE
        total_fee = fee + entry_fee_est
        fees_paid_pct_full = (total_fee / entry_cost) * 100
        entry_slippage_est = quantity * entry_price * SLIPPAGE_BUFFER
        total_slippage = slippage_cost + entry_slippage_est
        slippage_pct_full = (total_slippage / entry_cost) * 100
        # net = gross - fees - slippage (same denominator: entry_cost)
        net_pnl_pct = gross_pnl_pct - fees_paid_pct_full - slippage_pct_full

        # MFE / MAE / captured
        mfe_pct = ((highest_price - entry_price) / entry_price) * 100 if highest_price > 0 else 0.0
        mae_pct = (max(0, entry_price - lowest_price) / entry_price) * 100 if lowest_price > 0 else 0.0
        captured_pct = (net_pnl_pct / mfe_pct) if mfe_pct > 0 else None

        bars_held = int(hold_time_seconds / bar_interval_seconds) if bar_interval_seconds and bar_interval_seconds > 0 else None

        # STEP 1: Diagnostic flag for losing trades that closed almost
        # instantly (hold <= 5s with a net-negative outcome). Mystic does
        # not run stop-loss orders; this counter only highlights pathological
        # MANUAL closes or legacy rows. Active sells are net-profit-only.
        exit_reason_val = exit_type or exit_trigger
        if hold_time_seconds <= 5 and net_pnl_pct < 0:
            self._instant_stop_count += 1
            logger.warning(
                "INSTANT_LOSS_CLOSE: %s hold=%ss conf=%.2f spread=%.3f reason=%s",
                symbol,
                hold_time_seconds,
                entry_confidence,
                spread_pct,
                exit_reason_val,
            )

        record = ClosedTradeRecord(
            symbol=symbol,
            gross_pnl_pct=gross_pnl_pct,
            fees_paid_pct=fees_paid_pct_full,
            slippage_pct=slippage_pct_full,
            net_pnl_pct=net_pnl_pct,
            hold_seconds=hold_time_seconds,
            exit_reason=exit_type or exit_trigger,
            entry_confidence=entry_confidence,
            volatility_at_entry=atr_at_entry,
            spread_pct_at_entry=spread_pct,
            mfe_pct=mfe_pct,
            mae_pct=mae_pct,
            captured_pct=captured_pct,
            bars_held=bars_held,
        )
        self._closed_trades.append(record)

        # Drawdown & recovery (peak only updates when equity exceeds prior peak)
        self._last_equity = total_equity
        if total_equity > self._equity_peak:
            self._equity_peak = total_equity
            self._in_drawdown = False
        else:
            self._in_drawdown = True
            if self._equity_peak > 0:
                dd = ((self._equity_peak - total_equity) / self._equity_peak) * 100
                self._max_drawdown_pct = max(self._max_drawdown_pct, dd)

        # Log per-trade breakdown
        logger.info(
            "PROFIT_CLOSE: symbol=%s gross_pnl=%.3f%% fees=%.3f%% slippage=%.3f%% net=%.3f%% hold=%.0fs exit=%s mfe=%.2f%% mae=%.2f%% captured=%s",
            symbol,
            gross_pnl_pct,
            fees_paid_pct_full,
            slippage_pct_full,
            net_pnl_pct,
            hold_time_seconds,
            exit_type or exit_trigger,
            mfe_pct,
            mae_pct,
            f"{captured_pct:.2f}" if captured_pct is not None else "n/a",
        )

        # 2) Rolling win rate / expectancy
        self._log_rolling_stats_if_needed()

        # 4) Exit efficiency (already in PROFIT_CLOSE)

        # 6) Fee & slippage leakage
        self._closed_count_since_fee_log += 1
        if self._closed_count_since_fee_log >= FEE_LEAKAGE_LOG_EVERY:
            self._log_fee_leakage()
            self._closed_count_since_fee_log = 0

        # 10) Profitability snapshot
        self._closed_count_since_snapshot += 1
        if self._closed_count_since_snapshot >= PROFIT_SNAPSHOT_EVERY:
            self._log_profit_snapshot()
            self._closed_count_since_snapshot = 0

    def _log_rolling_stats_if_needed(self) -> None:
        trades = list(self._closed_trades)
        n_total = len(trades)
        if n_total < 10 or n_total % 10 != 0:
            return
        # Use largest window with enough data
        win = min(WINDOWS[-1], n_total)
        for w in reversed(WINDOWS):
            if n_total >= w:
                win = w
                break
        window = trades[-win:]
        wins = [t for t in window if t.net_pnl_pct > 0]
        losses = [t for t in window if t.net_pnl_pct < 0]
        n = len(window)
        if n == 0:
            return
        wr = len(wins) / n
        avg_win = sum(t.net_pnl_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss_mag = sum(t.net_pnl_pct for t in losses) / len(losses) if losses else 0.0
        avg_loss_mag = abs(avg_loss_mag)
        exp = (wr * avg_win) - ((1 - wr) * avg_loss_mag)
        sum_wins = sum(t.net_pnl_pct for t in wins)
        sum_losses = abs(sum(t.net_pnl_pct for t in losses))
        pf = sum_wins / sum_losses if sum_losses > 0 else (float("inf") if sum_wins > 0 else 1.0)
        logger.info(
            "PROFIT_ROLLING: n=%d win_rate=%.1f%% expectancy=%.3f%% profit_factor=%.2f avg_win=%.2f%% avg_loss=%.2f%%",
            n,
            wr * 100,
            exp,
            pf,
            avg_win,
            avg_loss_mag,
        )

    def _log_fee_leakage(self) -> None:
        trades = list(self._closed_trades)[-100:]
        if not trades:
            return
        total_fees = sum(t.fees_paid_pct for t in trades)
        total_slip = sum(t.slippage_pct for t in trades)
        avg_cost = (total_fees + total_slip) / len(trades)
        gross_profit = sum(t.gross_pnl_pct for t in trades if t.gross_pnl_pct > 0)
        cost_as_pct_gross = (total_fees + total_slip) / gross_profit * 100 if gross_profit > 0 else 0
        logger.info(
            "PROFIT_FEE_LEAKAGE: last_100 trades fees=%.3f%% slippage=%.3f%% avg_cost=%.3f%% cost_as_pct_gross=%.1f%%",
            total_fees / len(trades),
            total_slip / len(trades),
            avg_cost,
            cost_as_pct_gross,
        )

    def _log_profit_snapshot(self) -> None:
        trades = list(self._closed_trades)
        if not trades:
            return
        n = len(trades)
        wins = [t for t in trades if t.net_pnl_pct > 0]
        losses = [t for t in trades if t.net_pnl_pct < 0]
        wr = len(wins) / n * 100 if n > 0 else 0
        avg_win = sum(t.net_pnl_pct for t in wins) / len(wins) if wins else 0.0
        avg_loss_mag = abs(sum(t.net_pnl_pct for t in losses) / len(losses)) if losses else 0.0
        exp = (len(wins) / n * avg_win) - (len(losses) / n * avg_loss_mag) if n > 0 else 0.0
        sum_wins = sum(t.net_pnl_pct for t in wins)
        sum_losses = abs(sum(t.net_pnl_pct for t in losses))
        pf = sum_wins / sum_losses if sum_losses > 0 else (float("inf") if sum_wins > 0 else 1.0)
        avg_hold = sum(t.hold_seconds for t in trades) / n
        fees_avg = sum(t.fees_paid_pct for t in trades) / n
        slip_avg = sum(t.slippage_pct for t in trades) / n
        by_sym: dict[str, list[ClosedTradeRecord]] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)
        # Per-symbol stats (last 50-100 trades)
        sym_stats = []
        for s, ts in by_sym.items():
            sym_short = s.split("/")[0]
            sym_trades = ts[-100:]
            sym_n = len(sym_trades)
            sym_wins = [t for t in sym_trades if t.net_pnl_pct > 0]
            sym_losses = [t for t in sym_trades if t.net_pnl_pct < 0]
            sym_net = sum(t.net_pnl_pct for t in sym_trades)
            sym_wr = len(sym_wins) / sym_n * 100 if sym_n > 0 else 0
            sym_avg_hold = sum(t.hold_seconds for t in sym_trades) / sym_n if sym_n > 0 else 0
            sym_avg_win = sum(t.net_pnl_pct for t in sym_wins) / len(sym_wins) if sym_wins else 0
            sym_avg_loss = abs(sum(t.net_pnl_pct for t in sym_losses) / len(sym_losses)) if sym_losses else 0
            sym_exp = (len(sym_wins) / sym_n * sym_avg_win) - (len(sym_losses) / sym_n * sym_avg_loss) if sym_n > 0 else 0
            sum_sw = sum(t.net_pnl_pct for t in sym_wins)
            sum_sl = abs(sum(t.net_pnl_pct for t in sym_losses))
            sym_pf = sum_sw / sum_sl if sum_sl > 0 else (float("inf") if sum_sw > 0 else 1.0)
            # STEP 6: Symbol loss spike detector
            if sym_losses and sym_avg_loss > 2.0:
                logger.warning("SYMBOL_SPIKE: %s avg_loss=%.2f%%", sym_short, -sym_avg_loss)
            sym_stats.append((sym_short, sym_net, sym_wr, sym_exp, sym_pf, sym_avg_hold, sym_n))
        sym_stats.sort(key=lambda x: x[1], reverse=True)
        top = [s[0] for s in sym_stats[:3]]
        worst = [s[0] for s in sym_stats[-3:]]
        dd = ((self._equity_peak - self._last_equity) / self._equity_peak) * 100 if self._equity_peak > 0 else 0

        # Time-in-trade histogram
        hold_bucket_stats = []
        for lo, hi in HOLD_BUCKETS:
            bucket_trades = [t for t in trades if lo <= t.hold_seconds < hi]
            if bucket_trades:
                b_wins = [t for t in bucket_trades if t.net_pnl_pct > 0]
                b_wr = len(b_wins) / len(bucket_trades) * 100
                b_avg = sum(t.net_pnl_pct for t in bucket_trades) / len(bucket_trades)
                hold_bucket_stats.append(f"{int(lo)}-{int(hi)}s:wr={b_wr:.0f}% avg={b_avg:.2f}% n={len(bucket_trades)}")

        # Confidence bucket report (inclusive lower, exclusive upper; 0.90-1.01 includes 1.0)
        conf_lines = []
        for lo, hi in CONFIDENCE_BUCKETS:
            bucket_trades = [t for t in trades if lo <= t.entry_confidence < hi]
            if bucket_trades:
                b_wins = [t for t in bucket_trades if t.net_pnl_pct > 0]
                b_losses = [t for t in bucket_trades if t.net_pnl_pct < 0]
                b_wr = len(b_wins) / len(bucket_trades) * 100
                b_avg_win = sum(t.net_pnl_pct for t in b_wins) / len(b_wins) if b_wins else 0.0
                b_avg_loss = abs(sum(t.net_pnl_pct for t in b_losses) / len(b_losses)) if b_losses else 0.0
                b_exp = (len(b_wins) / len(bucket_trades) * b_avg_win) - (len(b_losses) / len(bucket_trades) * b_avg_loss)
                conf_lines.append(f"{lo:.2f}-{hi:.2f}:wr={b_wr:.0f}% exp={b_exp:.2f}% n={len(bucket_trades)}")

        miss = self.get_missed_counts()

        # STEP 1: instant_stop_rate
        instant_stop_rate = (self._instant_stop_count / n * 100) if n > 0 else 0.0

        # STEP 5: Risk ratio (tp/sl) — use default profile for snapshot
        try:
            from backend.services.portfolio_engine import get_coin_profile

            profile = get_coin_profile("BTC/USDT")
            tp_pct = profile.get("tp", 0.01) * 100
            sl_pct = profile.get("sl", 0.006) * 100
            risk_ratio = tp_pct / sl_pct if sl_pct > 0 else 0
            logger.info("RISK_RATIO: tp_pct=%.2f%% sl_pct=%.2f%% ratio=%.2f", tp_pct, sl_pct, risk_ratio)
        except Exception:
            risk_ratio = 0.0

        logger.info(
            "PROFIT_SNAPSHOT: v=%s trades=%d win_rate=%.0f%% expectancy=%.2f%% profit_factor=%.2f "
            "avg_hold=%.1fm fees_avg=%.2f%% slippage_avg=%.2f%% top=%s worst=%s drawdown=%.1f%% (max=%.1f%%) "
            "instant_stop_rate=%.1f%% spread_skips=%d low_vol_skips=%d risk_ratio=%.2f",
            DIAG_VERSION,
            n,
            wr,
            exp,
            pf,
            avg_hold / 60,
            fees_avg,
            slip_avg,
            top,
            worst,
            dd,
            self._max_drawdown_pct,
            instant_stop_rate,
            miss.get("spread", 0),
            miss.get("low_vol", 0),
            risk_ratio,
        )
        if hold_bucket_stats:
            logger.info("PROFIT_HOLD_HISTOGRAM: %s", " | ".join(hold_bucket_stats))
        if conf_lines:
            logger.info("PROFIT_CONFIDENCE_BUCKETS: %s", " | ".join(conf_lines))
        if any(miss.values()):
            logger.info(
                "PROFIT_MISSED: cooldown=%d max_pos=%d min_notional=%d confidence=%d spread=%d low_vol=%d",
                miss["cooldown"],
                miss["max_positions"],
                miss["min_notional"],
                miss["confidence"],
                miss.get("spread", 0),
                miss.get("low_vol", 0),
            )

    def record_skip(self, reason: str) -> None:
        """Increment missed opportunity counter."""
        r = reason.lower()
        if "cooldown" in r or "buy_cooldown" in r:
            self._skipped_cooldown += 1
        elif ("max" in r and "position" in r) or "max_positions" in r or "disciplined" in r:
            self._skipped_max_positions += 1
        elif "notional" in r or "exchange_constraint" in r or "min " in r:
            self._skipped_min_notional += 1
        elif "confidence" in r:
            self._skipped_confidence += 1
        elif "spread" in r:
            self._skipped_spread += 1
        elif "low_vol" in r or "low_volatility" in r:
            self._skipped_low_vol += 1

    def get_missed_counts(self) -> dict[str, int]:
        return {
            "cooldown": self._skipped_cooldown,
            "max_positions": self._skipped_max_positions,
            "min_notional": self._skipped_min_notional,
            "confidence": self._skipped_confidence,
            "spread": self._skipped_spread,
            "low_vol": self._skipped_low_vol,
        }


_diagnostics: ProfitabilityDiagnostics | None = None


def get_profitability_diagnostics() -> ProfitabilityDiagnostics:
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = ProfitabilityDiagnostics()
    return _diagnostics
