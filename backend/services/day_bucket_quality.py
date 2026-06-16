"""
DAY bucket quality — fat-tail control, kill list, risk sizing (no red thesis sells).

Tracks symbol/regime/thesis buckets. Negative buckets are blocked at entry.
Penalizes long-hold losers, high MAE, trapped capital before entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.services.day_regime_router import DAY_REGIME_NEUTRAL, DAY_REGIME_RANGE
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_VWAP_REVERSION,
)

# Replay-backed global kills: regime+thesis (any symbol)
GLOBAL_KILLED_REGIME_THESIS: frozenset[tuple[str, str]] = frozenset({
    (DAY_REGIME_NEUTRAL, SETUP_BREAKOUT_CONTINUATION),
    (DAY_REGIME_NEUTRAL, SETUP_HTF_TREND_PULLBACK),
    (DAY_REGIME_RANGE, SETUP_BREAKOUT_CONTINUATION),
    (DAY_REGIME_RANGE, SETUP_HTF_TREND_PULLBACK),
})

# Train walk-forward failures: hard-disable range VWAP on top losers
REPLAY_KILLED_BUCKETS: frozenset[tuple[str, str, str]] = frozenset({
    ("BTC/USDT", DAY_REGIME_RANGE, SETUP_VWAP_REVERSION),
    ("ETH/USDT", DAY_REGIME_RANGE, SETUP_VWAP_REVERSION),
    ("XRP/USDT", DAY_REGIME_RANGE, SETUP_VWAP_REVERSION),
})

# Symbol size penalties (replay 90d losers)
SYMBOL_SIZE_PENALTY: dict[str, float] = {
    "SOL/USDT": 0.72,
    "SOLUSDT": 0.72,
    "XRP/USDT": 0.78,
    "XRPUSDT": 0.78,
}

REGIME_SIZE_PENALTY: dict[str, float] = {
    DAY_REGIME_NEUTRAL: 0.72,
    DAY_REGIME_RANGE: 0.85,
}

# Fat-tail entry block thresholds
MAX_AVG_HOLD_HOURS_KILL = 72.0
MAX_AVG_HOLD_HOURS_RANGE_KILL = 40.0
MIN_TRADES_FOR_KILL = 3
MAX_MAE_PCT_KILL = -0.035
MIN_BUCKET_NET_PNL_KILL = -40.0
MIN_FAILED_PROFIT_FLOOR_KILL = 2


@dataclass
class BucketMetrics:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    net_pnl_usd: float = 0.0
    total_hold_sec: float = 0.0
    max_hold_sec: float = 0.0
    max_loss_usd: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown_usd: float = 0.0
    mae_sum: float = 0.0
    mfe_sum: float = 0.0
    trapped_capital_hours: float = 0.0
    failed_profit_floor: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades else 0.0

    @property
    def avg_hold_hours(self) -> float:
        return (self.total_hold_sec / self.trades / 3600.0) if self.trades else 0.0

    @property
    def expectancy_usd(self) -> float:
        return self.net_pnl_usd / self.trades if self.trades else 0.0

    @property
    def avg_mae_pct(self) -> float:
        return self.mae_sum / self.trades if self.trades else 0.0

    @property
    def avg_mfe_pct(self) -> float:
        return self.mfe_sum / self.trades if self.trades else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "net_pnl_usd": round(self.net_pnl_usd, 2),
            "expectancy_usd": round(self.expectancy_usd, 2),
            "avg_hold_hours": round(self.avg_hold_hours, 1),
            "max_hold_hours": round(self.max_hold_sec / 3600.0, 1),
            "max_loss_usd": round(self.max_loss_usd, 2),
            "max_drawdown_usd": round(self.max_drawdown_usd, 2),
            "avg_mae_pct": round(self.avg_mae_pct, 5),
            "avg_mfe_pct": round(self.avg_mfe_pct, 5),
            "trapped_capital_hours": round(self.trapped_capital_hours, 1),
            "failed_profit_floor": self.failed_profit_floor,
        }


def bucket_key(symbol: str, regime: str, thesis: str) -> tuple[str, str, str]:
    sym = (symbol or "").replace("/", "").upper()
    if sym.endswith("USDT") and len(sym) > 4:
        sym = f"{sym[:-4]}/USDT"
    if "/" not in sym and len(sym) > 3:
        sym = f"{sym[:-4]}/USDT" if sym.endswith("USDT") else sym
    # normalize BTCUSDT -> BTC/USDT
    if "/" not in sym:
        for base in ("BTC", "ETH", "SOL", "XRP"):
            if sym == f"{base}USDT":
                sym = f"{base}/USDT"
                break
    return (sym, str(regime or "neutral").lower(), str(thesis or ""))


def _norm_sym(symbol: str) -> str:
    s = (symbol or "").upper().replace("/", "")
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return symbol


def evaluate_bucket_entry(
    *,
    symbol: str,
    regime: str,
    setup: str,
    bucket_stats: dict[tuple[str, str, str], BucketMetrics] | None = None,
    extra_killed: frozenset[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """
    Pre-entry gate: block killed buckets, penalize risky buckets via size_factor.
    Returns allowed, block_reason, bucket_size_factor, bucket_rank_delta.
    """
    sym = _norm_sym(symbol)
    reg = str(regime or "neutral").lower()
    setup_s = str(setup or "")
    key = bucket_key(sym, reg, setup_s)
    stats = (bucket_stats or {}).get(key)

    if (reg, setup_s) in GLOBAL_KILLED_REGIME_THESIS:
        return {
            "allowed": False,
            "block_reason": "BUCKET_KILL_REGIME_THESIS",
            "bucket_size_factor": 0.0,
            "bucket_rank_delta": -0.20,
        }

    if key in REPLAY_KILLED_BUCKETS:
        return {
            "allowed": False,
            "block_reason": "BUCKET_KILL_REPLAY_RANGE_VWAP",
            "bucket_size_factor": 0.0,
            "bucket_rank_delta": -0.20,
        }

    if extra_killed and key in extra_killed:
        return {
            "allowed": False,
            "block_reason": "BUCKET_KILL_WALKFORWARD",
            "bucket_size_factor": 0.0,
            "bucket_rank_delta": -0.20,
        }

    if stats and stats.trades >= MIN_TRADES_FOR_KILL:
        if stats.net_pnl_usd <= MIN_BUCKET_NET_PNL_KILL:
            return {
                "allowed": False,
                "block_reason": "BUCKET_KILL_NEGATIVE_EXPECTANCY",
                "bucket_size_factor": 0.0,
                "bucket_rank_delta": -0.15,
            }
        hold_kill = MAX_AVG_HOLD_HOURS_RANGE_KILL if reg == DAY_REGIME_RANGE else MAX_AVG_HOLD_HOURS_KILL
        if stats.avg_hold_hours >= hold_kill and stats.net_pnl_usd < 0:
            return {
                "allowed": False,
                "block_reason": "BUCKET_KILL_FAT_TAIL_HOLD",
                "bucket_size_factor": 0.0,
                "bucket_rank_delta": -0.12,
            }
        if stats.failed_profit_floor >= MIN_FAILED_PROFIT_FLOOR_KILL and stats.net_pnl_usd < 0:
            return {
                "allowed": False,
                "block_reason": "BUCKET_KILL_FAILED_PROFIT_FLOOR",
                "bucket_size_factor": 0.0,
                "bucket_rank_delta": -0.12,
            }
        if stats.avg_mfe_pct > 0 and stats.avg_mae_pct < 0 and abs(stats.avg_mae_pct) > stats.avg_mfe_pct * 1.5:
            if stats.net_pnl_usd < 0 and stats.trades >= MIN_TRADES_FOR_KILL:
                return {
                    "allowed": False,
                    "block_reason": "BUCKET_KILL_HIGH_MAE_LOW_MFE",
                    "bucket_size_factor": 0.0,
                    "bucket_rank_delta": -0.10,
                }

    size = 1.0
    rank_delta = 0.0

    size *= REGIME_SIZE_PENALTY.get(reg, 1.0)
    size *= SYMBOL_SIZE_PENALTY.get(sym, 1.0)
    size *= SYMBOL_SIZE_PENALTY.get(sym.replace("/", ""), 1.0)

    if stats and stats.trades >= 2:
        if stats.expectancy_usd < 0:
            size *= 0.55
            rank_delta -= 0.08
        if stats.avg_hold_hours > 48:
            size *= max(0.45, 1.0 - (stats.avg_hold_hours - 48) / 200)
            rank_delta -= 0.04
        if stats.failed_profit_floor >= 2 and stats.trades >= 3:
            size *= 0.60
            rank_delta -= 0.06

    if setup_s == SETUP_VWAP_REVERSION and reg == DAY_REGIME_NEUTRAL:
        size = min(1.0, size * 1.05)
    if setup_s == SETUP_VWAP_REVERSION and reg == DAY_REGIME_RANGE:
        size = min(size, 0.55)
        rank_delta -= 0.08

    size = max(0.22, min(1.0, size))
    return {
        "allowed": True,
        "block_reason": "",
        "bucket_size_factor": round(size, 4),
        "bucket_rank_delta": round(rank_delta, 4),
    }


def record_bucket_outcome(
    bucket_stats: dict[tuple[str, str, str], BucketMetrics],
    *,
    symbol: str,
    regime: str,
    setup: str,
    pnl_usd: float,
    hold_sec: float,
    mae_pct: float = 0.0,
    mfe_pct: float = 0.0,
    exit_reason: str = "",
    notional_usd: float = 2500.0,
) -> None:
    key = bucket_key(symbol, regime, setup)
    st = bucket_stats.get(key) or BucketMetrics()
    st.trades += 1
    st.net_pnl_usd += pnl_usd
    if pnl_usd > 0:
        st.wins += 1
    else:
        st.losses += 1
    st.total_hold_sec += hold_sec
    st.max_hold_sec = max(st.max_hold_sec, hold_sec)
    st.max_loss_usd = min(st.max_loss_usd, pnl_usd)
    st.mae_sum += mae_pct
    st.mfe_sum += mfe_pct
    st.trapped_capital_hours += (hold_sec / 3600.0) * (notional_usd / 25000.0)
    if exit_reason and "NET_PROFIT" not in exit_reason.upper():
        st.failed_profit_floor += 1
    st.peak_pnl = max(st.peak_pnl, st.net_pnl_usd)
    dd = st.peak_pnl - st.net_pnl_usd
    st.max_drawdown_usd = max(st.max_drawdown_usd, dd)
    bucket_stats[key] = st


def buckets_negative(bucket_stats: dict[tuple[str, str, str], BucketMetrics], min_trades: int = 3) -> frozenset[tuple[str, str, str]]:
    killed: set[tuple[str, str, str]] = set(REPLAY_KILLED_BUCKETS)
    for key, st in bucket_stats.items():
        if st.trades < min_trades:
            continue
        sym, reg, _ = key
        hold_kill = MAX_AVG_HOLD_HOURS_RANGE_KILL if reg == DAY_REGIME_RANGE else MAX_AVG_HOLD_HOURS_KILL
        if st.net_pnl_usd < 0:
            killed.add(key)
        if st.avg_hold_hours >= hold_kill and st.net_pnl_usd < 0:
            killed.add(key)
        if st.failed_profit_floor >= MIN_FAILED_PROFIT_FLOOR_KILL and st.net_pnl_usd < 0:
            killed.add(key)
        if st.avg_mae_pct < 0 and st.avg_mfe_pct > 0 and abs(st.avg_mae_pct) > st.avg_mfe_pct * 1.5 and st.net_pnl_usd < 0:
            killed.add(key)
    return frozenset(killed)


def active_allowed_buckets(
    bucket_stats: dict[tuple[str, str, str], BucketMetrics],
    *,
    extra_killed: frozenset[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Buckets that may still trade live (positive expectancy or insufficient sample)."""
    killed = buckets_negative(bucket_stats) | (extra_killed or frozenset())
    allowed = []
    for key, st in sorted(bucket_stats.items()):
        if key in killed:
            continue
        if st.trades >= MIN_TRADES_FOR_KILL and st.expectancy_usd <= 0:
            continue
        sym, reg, thesis = key
        allowed.append({"symbol": sym, "regime": reg, "thesis": thesis, **st.to_dict()})
    return allowed


def bucket_report(bucket_stats: dict[tuple[str, str, str], BucketMetrics]) -> list[dict[str, Any]]:
    rows = []
    for (sym, reg, thesis), st in sorted(bucket_stats.items()):
        rows.append({"symbol": sym, "regime": reg, "thesis": thesis, **st.to_dict()})
    return rows


__all__ = [
    "BucketMetrics",
    "GLOBAL_KILLED_REGIME_THESIS",
    "REPLAY_KILLED_BUCKETS",
    "active_allowed_buckets",
    "bucket_key",
    "bucket_report",
    "buckets_negative",
    "evaluate_bucket_entry",
    "record_bucket_outcome",
]
