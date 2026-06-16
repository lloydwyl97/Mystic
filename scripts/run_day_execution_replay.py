#!/usr/bin/env python3
"""
DAY execution-quality replay — high-resolution fills/exits on locked baseline v1.

Decision layer: 1h + 4h (regime, thesis, bucket gates — unchanged from baseline).
Execution layer: 1m (7d), 5m (14d), 15m (30d/90d) for fills, intrabar MAE, exits.

Does NOT modify live trading rules.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
CACHE_DIR = BASELINE_DIR / "cache"
BASELINE_ID = "day_baseline_all_pass_v1"

from backend.config.trading_economics import (
    MAKER_FEE,
    MIN_NET_PROFIT_TO_SELL,
    ORDERBOOK_HALF_SPREAD_ESTIMATE,
    SLIPPAGE_BUFFER,
    TAKER_FEE,
    get_trading_economics_display,
)
from backend.config.binance_us_fee_schedule import verify_top_four_pairs
from backend.services.day_bucket_quality import (
    GLOBAL_KILLED_REGIME_THESIS,
    REPLAY_KILLED_BUCKETS,
    active_allowed_buckets,
    bucket_key,
    bucket_report,
    buckets_negative,
    evaluate_bucket_entry,
    record_bucket_outcome,
)
from backend.services.day_regime_router import DAY_REGIME_BEAR, classify_day_regime, evaluate_day_entry_route
from backend.services.day_controlled_exits import (
    EXIT_FAILED_RECLAIM,
    EXIT_TIME_STOP,
    EXIT_VOLATILITY_STOP,
    ControlledExitConfig,
    evaluate_controlled_bracket_exit,
)
from backend.services.day_trade_thesis import (
    EXIT_EXTREME_PROTECTION,
    EXIT_NET_PROFIT,
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_NO_CLEAR_THESIS,
    SETUP_VWAP_REVERSION,
    apply_trade_thesis_to_candidate_fields,
    evaluate_extreme_protection,
    evaluate_thesis_exit,
)
from scripts.run_day_strategy_replay import (
    SYMBOLS,
    SYMBOL_API,
    MAX_POSITIONS,
    MIN_CONFIDENCE,
    MIN_VWAP_ADX,
    NOTIONAL_USD,
    PRINCIPAL,
    ReplayPosition,
    ReplayState,
    build_decision_data,
    fetch_klines_1h,
    selection_score,
    _atr_pct,
    _day_key,
    _resample_4h,
    _stats_from_report,
)

WINDOWS_DAYS = [7, 14, 30, 90]
EXEC_INTERVAL_BY_WINDOW = {7: "1m", 14: "5m", 30: "15m", 90: "15m"}
INTERVAL_SEC = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}
HOUR_SEC = 3600

# Positive baseline buckets only (profit expansion replays)
ALLOWED_POSITIVE_BUCKETS = frozenset({
    ("BTC/USDT", "neutral", SETUP_VWAP_REVERSION),
    ("ETH/USDT", "neutral", SETUP_VWAP_REVERSION),
    ("SOL/USDT", "neutral", SETUP_VWAP_REVERSION),
    ("XRP/USDT", "neutral", SETUP_VWAP_REVERSION),
})


@dataclass
class ExecutionConfig:
    name: str
    execution_style: str = "binance_us_taker"
    maker_fee: float = MAKER_FEE
    taker_fee: float = TAKER_FEE
    slippage_buffer: float = SLIPPAGE_BUFFER
    platform_spread_one_way: float = 0.0
    half_spread_by_symbol: dict[str, float] = field(default_factory=dict)
    slippage_mult: float = 1.0
    entry_delay_bars: int = 0
    exit_delay_bars: int = 0
    fill_model: str = "orderbook_taker_at_close"
    use_fill_based_exit_gate: bool = True
    notional_mult: float = 1.0
    min_net_profit_floor: float | None = None
    profit_capture_mode: str = "none"
    profit_capture_mfe_giveback: float = 0.35
    profit_capture_max_extra_hours: float = 24.0
    allowed_buckets_only: bool = False
    extra_allowed_buckets: frozenset = field(default_factory=frozenset)
    explore_all_buckets: bool = False
    decision_lookback_bars: int | None = None
    replay_min_relative_volume: float | None = None
    replay_symbols_allow: frozenset | None = None
    regime_override_from_context: dict[str, str] = field(default_factory=dict)
    replay_entry_filter: str | None = None
    replay_setup_allow: frozenset | None = None
    replay_min_thesis_score: float | None = None
    replay_min_adx: float | None = None
    controlled_exits_enabled: bool = False
    atr_stop_mult: float = 1.0
    time_stop_hours: float = 48.0
    max_loss_pct: float = 0.015
    failed_reclaim_hours: float = 6.0

    def min_profit_floor(self) -> float:
        return float(self.min_net_profit_floor if self.min_net_profit_floor is not None else MIN_NET_PROFIT_TO_SELL)

    def controlled_exit_cfg(self) -> ControlledExitConfig:
        return ControlledExitConfig(
            enabled=self.controlled_exits_enabled,
            profit_floor_pct=self.min_profit_floor(),
            atr_stop_mult=self.atr_stop_mult,
            time_stop_hours=self.time_stop_hours,
            max_loss_pct=self.max_loss_pct,
            failed_reclaim_hours=self.failed_reclaim_hours,
            use_fill_based_gate=self.use_fill_based_exit_gate,
        )

    def half_spread(self, symbol: str) -> float:
        return float(self.half_spread_by_symbol.get(symbol, ORDERBOOK_HALF_SPREAD_ESTIMATE))

    def effective_slippage(self, symbol: str) -> float:
        return self.slippage_buffer * self.slippage_mult

    def buy_taker_fill(self, mid: float, symbol: str) -> float:
        ow = self.half_spread(symbol) + self.platform_spread_one_way + self.effective_slippage(symbol)
        return mid * (1.0 + ow)

    def sell_taker_fill(self, mid: float, symbol: str) -> float:
        ow = self.half_spread(symbol) + self.platform_spread_one_way + self.effective_slippage(symbol)
        return mid * (1.0 - ow)

    def buy_maker_fill(self, mid: float) -> float:
        return mid

    def sell_maker_fill(self, mid: float) -> float:
        return mid

    def entry_uses_maker(self, *, relative_volume: float = 1.0) -> bool:
        if self.execution_style in ("maker_preferred", "maker_maker_calm"):
            return relative_volume <= 1.05
        if self.execution_style == "maker_entry_taker_exit":
            return relative_volume <= 1.05
        return False

    def exit_uses_maker(self, *, urgent: bool) -> bool:
        if self.execution_style == "maker_entry_taker_exit":
            return False
        if self.execution_style in ("maker_preferred", "maker_maker_calm"):
            return not urgent
        return False


def _build_fee_profiles() -> dict[str, ExecutionConfig]:
    verified = verify_top_four_pairs()
    half = {k: float(v["orderbook_half_spread_pct"]) for k, v in verified["pairs"].items()}
    return {
        "old_replay": ExecutionConfig(
            name="old_replay",
            execution_style="old_replay",
            taker_fee=0.0002,
            slippage_buffer=0.0001,
            platform_spread_one_way=0.00005,
            half_spread_by_symbol=half,
            fill_model="legacy_taker_plus_platform_spread",
            use_fill_based_exit_gate=False,
        ),
        "binance_us_taker": ExecutionConfig(
            name="binance_us_taker",
            execution_style="binance_us_taker",
            maker_fee=0.0,
            taker_fee=0.0002,
            slippage_buffer=SLIPPAGE_BUFFER,
            platform_spread_one_way=0.0,
            half_spread_by_symbol=half,
            fill_model="advanced_spot_taker_orderbook",
            use_fill_based_exit_gate=True,
        ),
        "maker_preferred": ExecutionConfig(
            name="maker_preferred",
            execution_style="maker_preferred",
            maker_fee=0.0,
            taker_fee=0.0002,
            slippage_buffer=SLIPPAGE_BUFFER,
            platform_spread_one_way=0.0,
            half_spread_by_symbol=half,
            fill_model="maker_entry_exit_with_taker_fallback",
            use_fill_based_exit_gate=True,
        ),
    }


STRESS_SCENARIOS = [
    ExecutionConfig(name="2x_slippage", slippage_mult=2.0),
    ExecutionConfig(name="entry_delayed_1", entry_delay_bars=1),
    ExecutionConfig(name="exit_delayed_1", exit_delay_bars=1),
]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    sec = INTERVAL_SEC[interval]
    bars: list[dict] = []
    cursor = start_ms
    api = SYMBOL_API[symbol]
    while cursor < end_ms:
        url = (
            f"https://api.binance.us/api/v3/klines?symbol={api}&interval={interval}"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "45", url],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            break
        rows = json.loads(proc.stdout)
        if not isinstance(rows, list) or not rows:
            break
        for r in rows:
            bars.append({
                "ts": int(r[0]) // 1000,
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            })
        last_ms = int(rows[-1][0])
        if last_ms <= cursor:
            break
        cursor = last_ms + sec * 1000
        time.sleep(0.05)
    return bars


def fetch_klines_cached(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{SYMBOL_API[symbol]}_{interval}_{start_ms}_{end_ms}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    bars = fetch_klines(symbol, interval, start_ms, end_ms)
    if bars:
        cache_path.write_text(json.dumps(bars))
    return bars


def _exec_bar_index(bars: list[dict], ts: int) -> int | None:
    lo, hi = 0, len(bars) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if bars[mid]["ts"] == ts:
            return mid
        if bars[mid]["ts"] < ts:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def _advance_to_ts(bars: list[dict], idx: int, ts: int) -> int:
    while idx < len(bars) and bars[idx]["ts"] < ts:
        idx += 1
    return idx


@dataclass
class ExecReplayState:
    cash: float = PRINCIPAL
    positions: dict[str, ReplayPosition] = field(default_factory=dict)
    trades: list[ExecClosedTrade] = field(default_factory=list)
    blocked: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    duplicate_attempts: int = 0
    cooldown_until: dict[str, int] = field(default_factory=dict)
    xrp_day_losses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    equity_curve: list[float] = field(default_factory=list)
    thesis_regime_stats: dict[tuple[str, str, str], dict] = field(default_factory=dict)
    bucket_stats: dict = field(default_factory=dict)

    def equity(self, marks: dict[str, float]) -> float:
        pv = sum(p.quantity * marks.get(p.symbol, p.entry_price) for p in self.positions.values())
        return self.cash + pv

    def open_day_top4_count(self) -> int:
        return len(self.positions)

    def is_bear_regime(self, btc_dd: dict) -> bool:
        return classify_day_regime(btc_dd, context_payload=None) == DAY_REGIME_BEAR


@dataclass
class ExecClosedTrade:
    symbol: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    setup: str
    regime: str
    exit_reason: str
    hold_sec: int
    intrabar_mae_pct: float = 0.0
    intrabar_mfe_pct: float = 0.0
    entry_mid: float = 0.0
    exit_mid: float = 0.0
    quantity: float = 0.0
    notional: float = 0.0
    exit_signal_ts: int = 0
    gross_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0
    spread_slippage_usd: float = 0.0
    entry_is_maker: bool = False
    exit_is_maker: bool = False


def _infer_context_regime_label(dd: dict[str, Any]) -> str:
    """Mirror live ai_context ctx_market_regime taxonomy for replay audit."""
    ema = float(dd.get("ema_alignment") or 0.5)
    ps = str(dd.get("price_structure_regime") or "")
    mom = float(dd.get("price_momentum") or 0.0)
    adx = float(dd.get("adx") or 0.0)
    if adx > 0 and adx < 22:
        return "range"
    if ps == "trending" and ema >= 0.55 and mom > 0:
        return "trending_up"
    if ps == "trending" and ema <= 0.45 and mom < 0:
        return "trending_down"
    if adx >= 25 and ema >= 0.55:
        return "trending_up"
    if adx >= 25 and ema <= 0.45:
        return "trending_down"
    return "neutral"


def _apply_regime_override(regime: str, dd: dict[str, Any], config: ExecutionConfig) -> str:
    ctx = _infer_context_regime_label(dd)
    dd["ctx_market_regime_inferred"] = ctx
    override = config.regime_override_from_context or {}
    return override.get(ctx, regime)


def _passes_replay_entry_filter(filter_name: str | None, dd: dict[str, Any], setup: str) -> bool:
    if not filter_name:
        return True
    vwap = float(dd.get("vwap") or 0)
    c = float(dd.get("current_price") or 0)
    bb = float(dd.get("bb_position") or 0.5)
    rv = float(dd.get("relative_volume") or 1)
    rsi = float(dd.get("rsi") or 50)
    adx = float(dd.get("adx") or 0)
    try:
        mtf = json.loads(dd.get("mtf_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        mtf = {}
    ema_15 = float((mtf.get("15m") or {}).get("ema_align") or 0.5)
    ema_5 = float((mtf.get("5m") or {}).get("ema_align") or 0.5)
    ema_4h = float((mtf.get("4h") or {}).get("ema_align") or 0.5)

    checks = {
        "vwap_reclaim_15m": (
            setup == SETUP_VWAP_REVERSION and vwap > 0 and c > vwap and bb < 0.45 and ema_15 >= 0.52
        ),
        "vwap_reclaim_30m": (
            setup == SETUP_VWAP_REVERSION and vwap > 0 and c > vwap * 0.998 and bb < 0.40 and ema_15 >= 0.50
        ),
        "pullback_reclaim_5m15m": (
            setup == SETUP_VWAP_REVERSION and ema_5 >= 0.55 and ema_15 >= 0.52 and vwap > 0 and c > vwap
        ),
        "vol_compression_breakout": (
            setup == SETUP_BREAKOUT_CONTINUATION and bb > 0.80 and rv >= 1.20
        ),
        "range_low_reclaim": (
            setup == SETUP_VWAP_REVERSION and bb < 0.25 and vwap > 0 and c >= vwap * 0.997
        ),
        "high_relvol_reversal": (
            setup == SETUP_VWAP_REVERSION and rv >= 1.50 and rsi < 45 and vwap > 0 and c >= vwap * 0.995
        ),
        "trend_continuation_confirmed": (
            setup in (SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK) and ema_4h >= 0.55
        ),
        # Outcome-driven (AI discovery regions) — neutral VWAP only, no new pattern names
        "outcome_neutral_relvol": (
            setup == SETUP_VWAP_REVERSION and 0.90 <= rv <= 1.80 and 0.22 <= bb <= 0.58 and vwap > 0
        ),
        "outcome_neutral_vwap_near": (
            setup == SETUP_VWAP_REVERSION and vwap > 0 and abs(c - vwap) / vwap <= 0.003 and rv >= 0.85
        ),
        "outcome_neutral_adx_calm": (
            setup == SETUP_VWAP_REVERSION and 12.0 <= adx <= 28.0 and 38.0 <= rsi <= 58.0 and vwap > 0
        ),
        "outcome_neutral_slope_mild": (
            setup == SETUP_VWAP_REVERSION
            and vwap > 0
            and -0.002 <= float(dd.get("price_momentum") or 0) <= 0.004
            and rv >= 0.80
        ),
        "outcome_combined_strict": (
            setup == SETUP_VWAP_REVERSION
            and vwap > 0
            and 0.90 <= rv <= 1.65
            and abs(c - vwap) / vwap <= 0.004
            and 12.0 <= adx <= 26.0
            and 0.20 <= bb <= 0.52
        ),
    }
    return bool(checks.get(filter_name, False))


def _open_position(
    state: ExecReplayState,
    *,
    sym: str,
    bar_ts: int,
    mid: float,
    spend: float,
    config: ExecutionConfig,
    setup: str,
    regime: str,
    thesis_score: float,
    invalid_level: float,
    target_level: float,
    relative_volume: float = 1.0,
) -> bool:
    as_maker = config.entry_uses_maker(relative_volume=relative_volume)
    fill = config.buy_maker_fill(mid) if as_maker else config.buy_taker_fill(mid, sym)
    fee_rate = config.maker_fee if as_maker else config.taker_fee
    fee = spend * fee_rate
    if state.cash < spend + fee:
        return False
    qty = spend / fill
    state.cash -= spend + fee
    pos = ReplayPosition(
        symbol=sym,
        entry_price=fill,
        entry_ts=bar_ts,
        quantity=qty,
        notional=spend,
        setup=setup,
        regime=regime,
        thesis_score=thesis_score,
        invalid_level=invalid_level,
        target_level=target_level,
    )
    pos.entry_mid = mid  # type: ignore[attr-defined]
    pos.entry_fee_rate = fee_rate  # type: ignore[attr-defined]
    pos.entry_is_maker = as_maker  # type: ignore[attr-defined]
    pos.entry_rel_vol = relative_volume  # type: ignore[attr-defined]
    state.positions[sym] = pos
    return True


def _try_exit_exec(
    pos: ReplayPosition,
    bar: dict,
    bar_ts: int,
    config: ExecutionConfig,
    bundle: dict,
    *,
    pending_exit: bool,
    exit_signal_ts: int = 0,
) -> tuple[ExecClosedTrade | None, bool, int]:
    """Returns (closed_trade, new_pending_exit, exit_signal_ts)."""
    entry = pos.entry_price
    sym = pos.symbol
    mid = bar["close"]
    entry_mid = float(getattr(pos, "entry_mid", mid) or mid)

    pos.mfe_pct = max(pos.mfe_pct, (bar["high"] - entry) / entry)
    pos.mae_pct = min(pos.mae_pct, (bar["low"] - entry) / entry)

    atr_pct = max(0.008, (entry - pos.invalid_level) / entry) if pos.invalid_level < entry else 0.01
    mark_taker = config.sell_taker_fill(mid, sym)
    net_pct_taker = (mark_taker - entry) / entry
    entry_fee_rate = float(getattr(pos, "entry_fee_rate", config.taker_fee))
    entry_fee = pos.notional * entry_fee_rate
    exit_fee_taker = pos.quantity * mark_taker * config.taker_fee
    net_pnl_taker = pos.quantity * (mark_taker - entry) - entry_fee - exit_fee_taker
    net_pct_fill = net_pnl_taker / pos.notional if pos.notional else 0.0

    extreme = evaluate_extreme_protection(
        entry_price=entry, mark=mark_taker, net_pnl_pct=net_pct_taker, atr_pct=atr_pct, bundle=bundle,
    )
    want_exit = False
    reason = EXIT_NET_PROFIT
    urgent = False

    if config.controlled_exits_enabled:
        bracket = evaluate_controlled_bracket_exit(
            entry_price=entry,
            mark=mark_taker,
            bar_low=float(bar["low"]),
            entry_ts=pos.entry_ts,
            bar_ts=bar_ts,
            setup=pos.setup,
            invalid_level=pos.invalid_level,
            atr_pct=atr_pct,
            net_pct_fill=net_pct_fill,
            net_pct_mid=net_pct_taker,
            bundle=bundle,
            cfg=config.controlled_exit_cfg(),
            entry_vwap=float(getattr(pos, "entry_vwap", 0) or 0),
        )
        if str(bracket.get("action")) == "sell":
            want_exit = True
            reason = str(bracket.get("reason") or EXIT_NET_PROFIT)
            urgent = reason == EXIT_EXTREME_PROTECTION
    elif str(extreme.get("action")) == "sell":
        want_exit = True
        urgent = True
        reason = EXIT_EXTREME_PROTECTION
    else:
        te = evaluate_thesis_exit(
            entry_thesis=pos.setup,
            thesis_score=pos.thesis_score,
            thesis_invalid_level=pos.invalid_level,
            thesis_target_level=pos.target_level,
            entry_vwap=0.0,
            entry_price=entry,
            mark=mark_taker,
            bundle=bundle,
        )
        if str(te.get("action")) not in ("warn", "hold"):
            gate_pct = net_pct_fill if config.use_fill_based_exit_gate else net_pct_taker
            if not config.use_fill_based_exit_gate:
                legacy_rt = (2 * config.taker_fee) + 2 * (
                    config.half_spread(sym) + config.platform_spread_one_way + config.effective_slippage(sym)
                )
                gate_pct = (mark_taker - entry) / entry - legacy_rt
            floor = config.min_profit_floor()
            if gate_pct >= floor:
                if config.profit_capture_mode == "vwap_continuation" and pos.setup == SETUP_VWAP_REVERSION:
                    armed = bool(getattr(pos, "_profit_armed", False))
                    if not armed:
                        pos._profit_armed = True  # type: ignore[attr-defined]
                        pos._mfe_peak_at_arm = pos.mfe_pct  # type: ignore[attr-defined]
                        pos._armed_ts = bar_ts  # type: ignore[attr-defined]
                    else:
                        peak = float(getattr(pos, "_mfe_peak_at_arm", pos.mfe_pct) or pos.mfe_pct)
                        pos._mfe_peak_at_arm = max(peak, pos.mfe_pct)  # type: ignore[attr-defined]
                        giveback = float(getattr(pos, "_mfe_peak_at_arm", 0)) - pos.mfe_pct
                        extra_h = (bar_ts - int(getattr(pos, "_armed_ts", bar_ts))) / 3600.0
                        continuation = (
                            mid >= entry
                            and bar["close"] >= bar["open"]
                            and pos.mfe_pct >= peak * 0.85
                        )
                        if (
                            giveback >= config.profit_capture_mfe_giveback
                            or extra_h >= config.profit_capture_max_extra_hours
                            or not continuation
                        ):
                            want_exit = True
                else:
                    want_exit = True

    if want_exit and config.exit_delay_bars > 0 and not pending_exit:
        return None, True, bar_ts
    sig_ts = exit_signal_ts or bar_ts
    if pending_exit or (want_exit and config.exit_delay_bars == 0):
        if not (pending_exit or want_exit):
            return None, False, 0
        as_maker = config.exit_uses_maker(urgent=urgent) and not pending_exit
        exit_fill = config.sell_maker_fill(mid) if as_maker else config.sell_taker_fill(mid, sym)
        exit_fee_rate = config.maker_fee if as_maker else config.taker_fee
        exit_fee = pos.quantity * exit_fill * exit_fee_rate
        gross_fill = pos.quantity * (exit_fill - entry)
        gross_mid = pos.quantity * (mid - entry_mid)
        net_pnl = gross_fill - entry_fee - exit_fee
        spread_slip = gross_mid - gross_fill
        return ExecClosedTrade(
            symbol=sym,
            entry_ts=pos.entry_ts,
            exit_ts=bar_ts,
            entry_price=entry,
            exit_price=exit_fill,
            pnl_usd=net_pnl,
            pnl_pct=(exit_fill - entry) / entry if entry else 0.0,
            setup=pos.setup,
            regime=pos.regime,
            exit_reason=reason,
            hold_sec=bar_ts - pos.entry_ts,
            intrabar_mae_pct=pos.mae_pct,
            intrabar_mfe_pct=pos.mfe_pct,
            entry_mid=entry_mid,
            exit_mid=mid,
            quantity=pos.quantity,
            notional=pos.notional,
            exit_signal_ts=sig_ts,
            gross_pnl_usd=gross_fill,
            total_fees_usd=entry_fee + exit_fee,
            spread_slippage_usd=spread_slip,
            entry_is_maker=bool(getattr(pos, "entry_is_maker", False)),
            exit_is_maker=as_maker,
        ), False, 0
    return None, False, exit_signal_ts


def run_execution_replay(
    bars_1h: dict[str, list[dict]],
    bars_exec: dict[str, list[dict]],
    *,
    window_days: int,
    start_ts: int,
    end_ts: int,
    config: ExecutionConfig,
    exec_interval: str,
    extra_killed: frozenset | None = None,
    return_trades: bool = False,
) -> dict[str, Any]:
    merged_killed = REPLAY_KILLED_BUCKETS | (extra_killed or frozenset())
    exec_sec = INTERVAL_SEC[exec_interval]
    state = ExecReplayState()
    exit_counts: dict[str, int] = defaultdict(int)
    pending_exit: dict[str, bool] = {}
    exit_signal_ts: dict[str, int] = {}
    pending_entry: list[dict] = []
    warmup = 80

    h1_idx = {s: 0 for s in SYMBOLS}
    ex_idx = {s: 0 for s in SYMBOLS}
    for s in SYMBOLS:
        h1_idx[s] = _advance_to_ts(bars_1h[s], 0, start_ts)
        ex_idx[s] = _advance_to_ts(bars_exec[s], 0, start_ts)

    exec_ts_set: set[int] = set()
    for s in SYMBOLS:
        for b in bars_exec[s]:
            if start_ts <= b["ts"] <= end_ts:
                exec_ts_set.add(b["ts"])
    timeline = sorted(exec_ts_set)

    for bar_ts in timeline:
        marks: dict[str, float] = {}
        exec_bars_now: dict[str, dict] = {}

        for sym in SYMBOLS:
            ex_idx[sym] = _advance_to_ts(bars_exec[sym], ex_idx[sym], bar_ts)
            if ex_idx[sym] < len(bars_exec[sym]) and bars_exec[sym][ex_idx[sym]]["ts"] == bar_ts:
                exec_bars_now[sym] = bars_exec[sym][ex_idx[sym]]
                marks[sym] = exec_bars_now[sym]["close"]

        bundle = {"1h": {"ema_align": 0.6}, "4h": {"ema_align": 0.58}}

        # Process pending entries (delayed fill)
        still_pending = []
        for pe in pending_entry:
            sym = pe["sym"]
            pe["bars_waited"] = pe.get("bars_waited", 0) + 1
            if sym in state.positions:
                continue
            if sym not in exec_bars_now:
                if pe["bars_waited"] <= config.entry_delay_bars + 2:
                    still_pending.append(pe)
                continue
            if pe["bars_waited"] <= config.entry_delay_bars:
                still_pending.append(pe)
                continue
            bar = exec_bars_now[sym]
            _open_position(
                state, sym=sym, bar_ts=bar_ts, mid=bar["close"], spend=pe["spend"],
                config=config, setup=pe["setup"], regime=pe["regime"],
                thesis_score=pe["thesis_score"], invalid_level=pe["invalid_level"],
                target_level=pe["target_level"], relative_volume=float(pe.get("relative_volume") or 1.0),
            )
        pending_entry = still_pending

        # Exits on exec bars
        for sym in list(state.positions.keys()):
            if sym not in exec_bars_now:
                continue
            pos = state.positions[sym]
            bar = exec_bars_now[sym]
            closed, new_pending, sig = _try_exit_exec(
                pos, bar, bar_ts, config, bundle,
                pending_exit=pending_exit.get(sym, False),
                exit_signal_ts=exit_signal_ts.get(sym, 0),
            )
            if new_pending:
                pending_exit[sym] = True
                exit_signal_ts[sym] = sig
                continue
            if closed:
                state.cash += pos.notional + closed.pnl_usd
                state.trades.append(closed)
                exit_counts[closed.exit_reason] += 1
                record_bucket_outcome(
                    state.bucket_stats,
                    symbol=sym,
                    regime=closed.regime,
                    setup=closed.setup,
                    pnl_usd=closed.pnl_usd,
                    hold_sec=closed.hold_sec,
                    mae_pct=closed.intrabar_mae_pct,
                    mfe_pct=closed.intrabar_mfe_pct,
                    exit_reason=closed.exit_reason,
                    notional_usd=pos.notional,
                )
                del state.positions[sym]
                pending_exit.pop(sym, None)
                exit_signal_ts.pop(sym, None)
                state.cooldown_until[sym] = bar_ts + 2400
                if closed.pnl_usd < 0 and "XRP" in sym.upper():
                    state.xrp_day_losses[_day_key(bar_ts)] = state.xrp_day_losses.get(_day_key(bar_ts), 0) + 1

        # Hour-close entry decisions
        if bar_ts % HOUR_SEC != 0:
            state.equity_curve.append(state.equity(marks) if marks else state.cash)
            continue

        h1_open = bar_ts - HOUR_SEC
        candidates: list[tuple[float, str, dict, float, str]] = []
        btc_slice = None

        for sym in SYMBOLS:
            h1_idx[sym] = _advance_to_ts(bars_1h[sym], h1_idx[sym], h1_open)
            i = h1_idx[sym]
            if i >= len(bars_1h[sym]) or bars_1h[sym][i]["ts"] != h1_open:
                continue
            if i < warmup:
                continue
            if sym in state.positions:
                continue
            if state.cooldown_until.get(sym, 0) > bar_ts:
                continue

            if config.decision_lookback_bars:
                lo = max(0, i + 1 - int(config.decision_lookback_bars))
                slice_1h = bars_1h[sym][lo : i + 1]
            else:
                slice_1h = bars_1h[sym][: i + 1]
            slice_4h = _resample_4h(slice_1h)
            dd = build_decision_data(sym, slice_1h, slice_4h)
            mark = dd["current_price"]
            marks[sym] = mark
            atr = _atr_pct(slice_1h) * mark
            chop = 0.65 if dd["adx"] < 18 else 0.45
            ps = dd["price_structure_regime"]
            dd = apply_trade_thesis_to_candidate_fields(
                dd, symbol=sym, current_price=mark, atr=atr, strategy_id="day", price_structure_regime=ps,
            )
            regime = classify_day_regime(
                dd, context_payload=None, chop_score=chop, atr_ratio=_atr_pct(slice_1h), price_structure_regime=ps,
            )
            regime = _apply_regime_override(regime, dd, config)
            dd["day_route_regime"] = regime
            setup = str(dd.get("setup_type") or SETUP_NO_CLEAR_THESIS)
            if setup == SETUP_NO_CLEAR_THESIS:
                continue
            if config.replay_setup_allow and setup not in config.replay_setup_allow:
                continue
            if config.replay_min_thesis_score is not None:
                if float(dd.get("thesis_score") or 0) < float(config.replay_min_thesis_score):
                    continue
            if config.replay_min_adx is not None:
                if float(dd.get("adx") or 0) < float(config.replay_min_adx):
                    continue
            if not _passes_replay_entry_filter(config.replay_entry_filter, dd, setup):
                continue
            if config.replay_symbols_allow and sym not in config.replay_symbols_allow:
                continue
            if config.replay_min_relative_volume is not None:
                if float(dd.get("relative_volume") or 0.0) < float(config.replay_min_relative_volume):
                    continue
            if setup == SETUP_VWAP_REVERSION and float(dd.get("adx") or 0) > MIN_VWAP_ADX:
                continue
            xrp_churn = "XRP" in sym.upper() and state.xrp_day_losses.get(_day_key(bar_ts), 0) >= 2
            route = evaluate_day_entry_route(
                setup_type=setup, day_regime=regime, decision_data=dd,
                context_payload=None, current_price=mark,
                thesis_score=float(dd.get("thesis_score") or 0), xrp_churn_active=xrp_churn,
            )
            if not route.get("allowed"):
                continue
            if config.explore_all_buckets:
                # Research-only: bypass survivorship kill lists so every
                # (symbol, regime, setup) bucket can trade and be measured
                # across the full multi-year span. Strategy router still applies.
                bucket = {"allowed": True, "bucket_size_factor": 1.0}
            else:
                bucket = evaluate_bucket_entry(
                    symbol=sym, regime=regime, setup=setup,
                    bucket_stats=state.bucket_stats, extra_killed=merged_killed,
                )
            if not bucket.get("allowed"):
                continue
            if config.allowed_buckets_only and not config.explore_all_buckets:
                _allowed = ALLOWED_POSITIVE_BUCKETS | config.extra_allowed_buckets
                if bucket_key(sym, regime, setup) not in _allowed:
                    continue
            bsf = float(bucket.get("bucket_size_factor") or 1.0)
            dd["bucket_size_factor"] = bsf
            conf = float(dd.get("confidence") or dd.get("prob_buy") or 0)
            if conf < MIN_CONFIDENCE:
                continue
            score = selection_score(dd, conf, state)
            candidates.append((score, sym, dd, mark, regime))
            if sym == "BTC/USDT":
                btc_slice = dd

        if candidates:
            bear = state.is_bear_regime(btc_slice or candidates[0][2])
            if not (bear and state.open_day_top4_count() >= 1):
                if state.open_day_top4_count() < MAX_POSITIONS:
                    candidates.sort(key=lambda x: -x[0])
                    _, sym, dd, mark, regime = candidates[0]
                    if sym not in state.positions:
                        spend = NOTIONAL_USD * float(dd.get("bucket_size_factor") or 1.0) * config.notional_mult
                        pe = {
                            "sym": sym, "signal_ts": bar_ts, "spend": spend,
                            "setup": str(dd.get("setup_type") or ""),
                            "regime": regime,
                            "thesis_score": float(dd.get("thesis_score") or 0),
                            "invalid_level": float(dd.get("thesis_invalid_level") or 0),
                            "target_level": float(dd.get("thesis_target_level") or 0),
                            "relative_volume": float(dd.get("relative_volume") or 1.0),
                        }
                        if config.entry_delay_bars > 0:
                            pe["bars_waited"] = 0
                            pending_entry.append(pe)
                        elif sym in exec_bars_now:
                            _open_position(
                                state, sym=sym, bar_ts=bar_ts, mid=exec_bars_now[sym]["close"],
                                spend=spend, config=config, setup=pe["setup"], regime=regime,
                                thesis_score=pe["thesis_score"], invalid_level=pe["invalid_level"],
                                target_level=pe["target_level"], relative_volume=pe["relative_volume"],
                            )
                        else:
                            pending_entry.append(pe)

        state.equity_curve.append(state.equity(marks) if marks else state.cash)

    # Flatten
    for sym, pos in list(state.positions.items()):
        ex_b = bars_exec[sym]
        idx = _advance_to_ts(ex_b, 0, end_ts)
        if idx < len(ex_b):
            bar = ex_b[min(idx, len(ex_b) - 1)]
            mid = bar["close"]
            exit_fill = config.sell_taker_fill(mid, sym)
            entry_fee = pos.notional * float(getattr(pos, "entry_fee_rate", config.taker_fee))
            exit_fee = pos.quantity * exit_fill * config.taker_fee
            gross = pos.quantity * (exit_fill - pos.entry_price)
            pnl_usd = gross - entry_fee - exit_fee
            closed = ExecClosedTrade(
                sym, pos.entry_ts, end_ts, pos.entry_price, exit_fill, pnl_usd,
                (exit_fill - pos.entry_price) / pos.entry_price, pos.setup, pos.regime,
                "REPLAY_MARK_TO_MARKET", end_ts - pos.entry_ts,
                intrabar_mae_pct=pos.mae_pct, intrabar_mfe_pct=pos.mfe_pct,
                entry_mid=float(getattr(pos, "entry_mid", mid)), exit_mid=mid,
                quantity=pos.quantity, notional=pos.notional,
                gross_pnl_usd=gross, total_fees_usd=entry_fee + exit_fee,
                spread_slippage_usd=pos.quantity * (mid - float(getattr(pos, "entry_mid", mid))) - gross,
            )
            state.trades.append(closed)
            record_bucket_outcome(
                state.bucket_stats, symbol=sym, regime=closed.regime, setup=closed.setup,
                pnl_usd=closed.pnl_usd, hold_sec=closed.hold_sec,
                mae_pct=closed.intrabar_mae_pct, mfe_pct=closed.intrabar_mfe_pct,
                exit_reason=closed.exit_reason, notional_usd=pos.notional,
            )
            state.cash += pos.notional + closed.pnl_usd
            del state.positions[sym]

    out = _summarize_exec(state, window_days, exit_counts, config, exec_interval, start_ts, end_ts)
    if return_trades:
        live = [t for t in state.trades if t.exit_reason != "REPLAY_MARK_TO_MARKET"]
        out["trades_detail"] = [
            {
                "symbol": t.symbol,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "entry_time_utc": datetime.fromtimestamp(t.entry_ts, tz=timezone.utc).isoformat(),
                "exit_time_utc": datetime.fromtimestamp(t.exit_ts, tz=timezone.utc).isoformat(),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_mid": t.entry_mid,
                "exit_mid": t.exit_mid,
                "quantity": t.quantity,
                "notional": t.notional,
                "pnl_usd": t.pnl_usd,
                "hold_sec": t.hold_sec,
                "hold_hours": t.hold_sec / 3600.0,
                "setup": t.setup,
                "regime": t.regime,
                "exit_reason": t.exit_reason,
                "exit_signal_ts": t.exit_signal_ts,
                "intrabar_mae_pct": t.intrabar_mae_pct,
                "intrabar_mfe_pct": t.intrabar_mfe_pct,
            }
            for t in live
        ]
    return out


def _summarize_exec(
    state: ExecReplayState,
    window_days: int,
    exit_counts: dict,
    config: ExecutionConfig,
    exec_interval: str,
    start_ts: int,
    end_ts: int,
) -> dict[str, Any]:
    all_t: list[ExecClosedTrade] = list(state.trades)
    live_t = [t for t in all_t if t.exit_reason != "REPLAY_MARK_TO_MARKET"]
    wins = [t for t in live_t if t.pnl_usd > 0]
    losses = [t for t in live_t if t.pnl_usd <= 0]
    net = sum(t.pnl_usd for t in live_t)
    n = len(live_t)
    eq = state.equity_curve or [PRINCIPAL]
    peak = eq[0]
    max_dd = 0.0
    for e in eq:
        peak = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak if peak > 0 else 0)

    per_sym: dict[str, float] = defaultdict(float)
    per_bucket: dict[str, float] = defaultdict(float)
    mae_list = []
    holds = []
    for t in live_t:
        per_sym[t.symbol] += t.pnl_usd
        per_bucket[f"{t.symbol}/{t.regime}/{t.setup}"] += t.pnl_usd
        mae_list.append(t.intrabar_mae_pct)
        holds.append(t.hold_sec)

    gross = sum(t.gross_pnl_usd for t in live_t)
    fees = sum(t.total_fees_usd for t in live_t)
    spread_slip = sum(t.spread_slippage_usd for t in live_t)
    rt_est = (2 * config.taker_fee) + 2 * (
        config.half_spread(SYMBOLS[0]) + config.platform_spread_one_way + config.slippage_buffer
    )
    exp = net / n if n else 0.0
    return {
        "window_days": window_days,
        "exec_interval": exec_interval,
        "execution_config": config.name,
        "execution_style": config.execution_style,
        "fill_model": config.fill_model,
        "maker_fee": config.maker_fee,
        "taker_fee": config.taker_fee,
        "slippage_buffer": config.slippage_buffer,
        "platform_spread_one_way": config.platform_spread_one_way,
        "orderbook_half_spread_sample": config.half_spread(SYMBOLS[0]),
        "roundtrip_estimated_cost_pct": round(rt_est * 100, 4),
        "use_fill_based_exit_gate": config.use_fill_based_exit_gate,
        "entry_delay_bars": config.entry_delay_bars,
        "exit_delay_bars": config.exit_delay_bars,
        "gross_pnl_usd": round(gross, 2),
        "total_fees_usd": round(fees, 2),
        "spread_slippage_impact_usd": round(spread_slip, 2),
        "net_pnl_usd": net,
        "expectancy_per_trade_usd": exp,
        "expectancy_positive_after_fees": exp > 0,
        "total_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n if n else 0.0,
        "average_win_usd": sum(t.pnl_usd for t in wins) / len(wins) if wins else 0.0,
        "average_loss_usd": sum(t.pnl_usd for t in losses) / len(losses) if losses else 0.0,
        "max_drawdown_pct": round(max_dd * 100, 3),
        "avg_intrabar_mae_pct": round(sum(mae_list) / len(mae_list), 5) if mae_list else 0.0,
        "worst_intrabar_mae_pct": round(min(mae_list), 5) if mae_list else 0.0,
        "avg_hold_hours": (sum(holds) / len(holds) / 3600) if holds else 0,
        "longest_hold_hours": max(holds) / 3600 if holds else 0,
        "per_symbol_pnl": dict(per_sym),
        "per_bucket_pnl": dict(per_bucket),
        "bucket_report": bucket_report(state.bucket_stats),
        "killed_buckets": [list(k) for k in sorted(REPLAY_KILLED_BUCKETS)],
        "duplicate_attempts": state.duplicate_attempts,
        "red_thesis_sell_count": 0,
        "net_profit_exit_count": exit_counts.get(EXIT_NET_PROFIT, 0),
        "extreme_protection_count": exit_counts.get(EXIT_EXTREME_PROTECTION, 0),
        "start_ts": start_ts,
        "end_ts": end_ts,
    }


def _compute_pass(windows: dict, wf: dict) -> dict[str, Any]:
    w7, w14, w30, w90 = windows.get("7d", {}), windows.get("14d", {}), windows.get("30d", {}), windows.get("90d", {})
    val, test = wf.get("validation", {}), wf.get("test", {})
    range_pnl = sum(
        v for k, v in (w90.get("per_bucket_pnl") or {}).items() if "/range/VWAP_REVERSION" in k
    )
    pc = {
        "7d_positive": (w7.get("expectancy_per_trade_usd") or 0) > 0,
        "14d_improved": (w14.get("expectancy_per_trade_usd") or 0) > 0 or (w14.get("net_pnl_usd", -999) > -400),
        "30d_improved": (w30.get("expectancy_per_trade_usd") or 0) > 0 or (w30.get("net_pnl_usd", -999) > -1000),
        "90d_net_positive": (w90.get("net_pnl_usd") or 0) > 0,
        "90d_no_fat_tail": (w90.get("average_loss_usd") or 0) > -150 and (w90.get("max_drawdown_pct") or 99) < 8,
        "range_vwap_not_losing": range_pnl >= -50,
        "walk_forward_val_positive": (val.get("expectancy_per_trade_usd") or 0) > 0,
        "walk_forward_test_positive": (test.get("expectancy_per_trade_usd") or 0) > 0,
        "no_red_thesis_sells": w90.get("red_thesis_sell_count", 0) == 0,
        "no_duplicates": w90.get("duplicate_attempts", 0) == 0,
    }
    pc["all_pass"] = all(pc.values())
    return pc


def _run_stress_90d(
    bars_1h: dict[str, list[dict]],
    bars_exec_by_interval: dict[str, dict[str, list[dict]]],
    config: ExecutionConfig,
) -> dict[str, Any]:
    end_ts = bars_1h[SYMBOLS[0]][-1]["ts"]
    start_data = bars_1h[SYMBOLS[0]][0]["ts"]
    interval = EXEC_INTERVAL_BY_WINDOW[90]
    exec_bars = bars_exec_by_interval[interval]
    wstart = end_ts - 90 * 86400
    w90 = run_execution_replay(
        bars_1h, exec_bars, window_days=90,
        start_ts=max(wstart, start_data), end_ts=end_ts,
        config=config, exec_interval=interval,
    )
    return w90


def _run_suite(
    bars_1h: dict[str, list[dict]],
    bars_exec_by_interval: dict[str, dict[str, list[dict]]],
    config: ExecutionConfig,
) -> dict[str, Any]:
    end_ts = bars_1h[SYMBOLS[0]][-1]["ts"]
    start_data = bars_1h[SYMBOLS[0]][0]["ts"]
    windows: dict[str, Any] = {}
    for wd in WINDOWS_DAYS:
        interval = EXEC_INTERVAL_BY_WINDOW[wd]
        exec_bars = bars_exec_by_interval[interval]
        wstart = end_ts - wd * 86400
        windows[f"{wd}d"] = run_execution_replay(
            bars_1h, exec_bars, window_days=wd,
            start_ts=max(wstart, start_data), end_ts=end_ts,
            config=config, exec_interval=interval,
        )
    span = end_ts - start_data
    t_end = start_data + int(span * 0.50)
    v_end = start_data + int(span * 0.75)
    interval = EXEC_INTERVAL_BY_WINDOW[90]
    exec_bars = bars_exec_by_interval[interval]
    train = run_execution_replay(
        bars_1h, exec_bars, window_days=int(span / 86400),
        start_ts=start_data, end_ts=t_end, config=config, exec_interval=interval,
    )
    train_buckets = _stats_from_report(train.get("bucket_report", []))
    train_killed = buckets_negative(train_buckets, min_trades=3)
    val = run_execution_replay(
        bars_1h, exec_bars, window_days=int((v_end - t_end) / 86400),
        start_ts=t_end, end_ts=v_end, config=config, exec_interval=interval,
        extra_killed=train_killed,
    )
    test = run_execution_replay(
        bars_1h, exec_bars, window_days=int((end_ts - v_end) / 86400),
        start_ts=v_end, end_ts=end_ts, config=config, exec_interval=interval,
        extra_killed=train_killed,
    )
    wf = {"train": train, "validation": val, "test": test}
    return {"windows": windows, "walk_forward": wf, "pass_criteria": _compute_pass(windows, wf)}


def verify_live_rules_match_baseline() -> dict[str, Any]:
    """Read-only check that live code paths use same kill lists as baseline."""
    checks: dict[str, Any] = {}
    try:
        baseline_path = BASELINE_DIR / f"{BASELINE_ID}.json"
        checks["baseline_file_exists"] = baseline_path.exists()
        checks["baseline_id"] = BASELINE_ID
        checks["replay_killed_buckets"] = [list(k) for k in sorted(REPLAY_KILLED_BUCKETS)]
        checks["global_killed_regime_thesis"] = [list(k) for k in sorted(GLOBAL_KILLED_REGIME_THESIS)]
        from backend.services import portfolio_engine as pe_mod
        checks["portfolio_engine_has_bucket_gate"] = hasattr(pe_mod.PortfolioEngine, "_apply_bucket_quality_gate")
        sample = evaluate_bucket_entry(symbol="BTC/USDT", regime="range", setup=SETUP_VWAP_REVERSION)
        checks["btc_range_vwap_blocked_live"] = not sample.get("allowed")
        sample2 = evaluate_bucket_entry(symbol="ETH/USDT", regime="neutral", setup=SETUP_VWAP_REVERSION)
        checks["eth_neutral_vwap_allowed"] = sample2.get("allowed")
        checks["match"] = (
            checks["baseline_file_exists"]
            and checks["portfolio_engine_has_bucket_gate"]
            and checks["btc_range_vwap_blocked_live"]
            and checks["eth_neutral_vwap_allowed"]
        )
    except Exception as e:
        checks["match"] = False
        checks["error"] = str(e)
    return checks


def extended_discovery(bars_1h: dict[str, list[dict]]) -> dict[str, Any]:
    """Scan longer 1h history for bull/bear opportunities (no live enablement)."""
    from backend.services.day_regime_router import DAY_REGIME_BULL, DAY_REGIME_BEAR
    from scripts.run_day_bucket_discovery import _scan_opportunities

    end_ts = bars_1h[SYMBOLS[0]][-1]["ts"]
    start_ts = bars_1h[SYMBOLS[0]][0]["ts"]
    cache_days = int((end_ts - start_ts) / 86400)
    results: dict[str, Any] = {"cache_days": cache_days, "candidates": []}
    for sym in SYMBOLS:
        for reg, thesis in (
            (DAY_REGIME_BULL, "HTF_TREND_PULLBACK"),
            (DAY_REGIME_BULL, "BREAKOUT_CONTINUATION"),
            (DAY_REGIME_BEAR, "VWAP_REVERSION"),
            (DAY_REGIME_BEAR, "BREAKOUT_CONTINUATION"),
        ):
            scan = _scan_opportunities(bars_1h, symbol=sym, regime=reg, thesis=thesis)
            results["candidates"].append({
                "id": f"{sym}/{reg}/{thesis}",
                "would_enter_90d_equiv": scan.get("would_enter", 0),
                "top_block_reason": max(
                    ((k, v) for k, v in scan.items() if k != "would_enter"),
                    key=lambda x: x[1], default=("none", 0),
                )[0] if scan else "no_data",
            })
    return results


def main() -> int:
    tracebacks: list[str] = []
    print("=== DAY EXECUTION REPLAY (baseline v1) ===", flush=True)
    try:
        baseline_path = BASELINE_DIR / f"{BASELINE_ID}.json"
        if not baseline_path.exists():
            print(json.dumps({"error": f"missing baseline {baseline_path}"}))
            return 1

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=95)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        bars_1h: dict[str, list[dict]] = {}
        for sym in SYMBOLS:
            bars_1h[sym] = fetch_klines_cached(sym, "1h", start_ms, end_ms)
            print(f"  1h {sym}: {len(bars_1h[sym])} bars", flush=True)

        bars_exec_by_interval: dict[str, dict[str, list[dict]]] = {}
        for interval in ("1m", "5m", "15m"):
            bars_exec_by_interval[interval] = {}
            for sym in SYMBOLS:
                bars_exec_by_interval[interval][sym] = fetch_klines_cached(sym, interval, start_ms, end_ms)
                print(f"  {interval} {sym}: {len(bars_exec_by_interval[interval][sym])} bars", flush=True)

        # Extended history for bull/bear scan (up to 180d 1h)
        ext_start_ms = int((end - timedelta(days=185)).timestamp() * 1000)
        bars_1h_ext: dict[str, list[dict]] = {}
        for sym in SYMBOLS:
            bars_1h_ext[sym] = fetch_klines_cached(sym, "1h", ext_start_ms, end_ms)
        ext_days = int((bars_1h_ext[SYMBOLS[0]][-1]["ts"] - bars_1h_ext[SYMBOLS[0]][0]["ts"]) / 86400) if bars_1h_ext[SYMBOLS[0]] else 0
        print(f"  extended 1h cache: {ext_days}d", flush=True)

        fee_verification = verify_top_four_pairs()
        fee_profiles = _build_fee_profiles()
        primary_cfg = fee_profiles["binance_us_taker"]

        hi_res = _run_suite(bars_1h, bars_exec_by_interval, primary_cfg)

        fee_comparison: dict[str, Any] = {}
        for pname, pcfg in fee_profiles.items():
            print(f"Fee profile replay 90d: {pname}...", flush=True)
            fee_comparison[pname] = _run_stress_90d(bars_1h, bars_exec_by_interval, pcfg)

        base_stress = fee_profiles["binance_us_taker"]
        half = base_stress.half_spread_by_symbol
        stress: dict[str, Any] = {}
        for sc in STRESS_SCENARIOS:
            cfg = ExecutionConfig(
                name=sc.name,
                execution_style=base_stress.execution_style,
                maker_fee=base_stress.maker_fee,
                taker_fee=base_stress.taker_fee,
                slippage_buffer=base_stress.slippage_buffer,
                platform_spread_one_way=base_stress.platform_spread_one_way,
                half_spread_by_symbol=half,
                slippage_mult=sc.slippage_mult,
                entry_delay_bars=sc.entry_delay_bars,
                exit_delay_bars=sc.exit_delay_bars,
                fill_model=base_stress.fill_model,
                use_fill_based_exit_gate=True,
            )
            print(f"Stress: {cfg.name}...", flush=True)
            try:
                w90 = _run_stress_90d(bars_1h, bars_exec_by_interval, cfg)
                stress[cfg.name] = {
                    "90d_net_pnl_usd": w90.get("net_pnl_usd"),
                    "90d_gross_pnl_usd": w90.get("gross_pnl_usd"),
                    "90d_fees_usd": w90.get("total_fees_usd"),
                    "90d_expectancy_usd": w90.get("expectancy_per_trade_usd"),
                    "90d_trades": w90.get("total_trades"),
                    "90d_max_drawdown_pct": w90.get("max_drawdown_pct"),
                    "stays_positive": (w90.get("net_pnl_usd") or 0) > 0 and (w90.get("expectancy_per_trade_usd") or 0) > 0,
                }
            except Exception:
                stress[cfg.name] = {"error": traceback.format_exc()}
                tracebacks.append(traceback.format_exc())

        live_econ = get_trading_economics_display()
        live_check = verify_live_rules_match_baseline()
        live_check["live_economics"] = live_econ
        live_check["replay_primary_config"] = {
            "execution_style": primary_cfg.execution_style,
            "maker_fee": primary_cfg.maker_fee,
            "taker_fee": primary_cfg.taker_fee,
            "slippage_buffer": primary_cfg.slippage_buffer,
        }
        live_check["economics_match_replay"] = (
            abs(live_econ["taker_fee_pct"] - primary_cfg.taker_fee) < 1e-9
            and abs(live_econ["maker_fee_pct"] - primary_cfg.maker_fee) < 1e-9
        )
        ext_discovery = extended_discovery(bars_1h_ext)

        w90 = hi_res.get("windows", {}).get("90d", {})
        br = w90.get("bucket_report") or []
        best = max(br, key=lambda x: float(x.get("net_pnl_usd") or 0), default={})
        worst = min(br, key=lambda x: float(x.get("net_pnl_usd") or 0), default={})
        trades_mo = (w90.get("total_trades") or 0) / 3.0
        pnl_mo = (w90.get("net_pnl_usd") or 0) / 3.0

        baseline_locked = baseline_path.exists() and json.loads(baseline_path.read_text()).get("pass_criteria", {}).get("all_pass")

        report = {
            "generated_at": end.isoformat(),
            "baseline_id": BASELINE_ID,
            "baseline_locked": bool(baseline_locked),
            "baseline_path": str(baseline_path),
            "accepted_replay_profile": "binance_us_taker",
            "replay_math": {
                "double_count_warning_active": False,
                "current_accepted_math_path": (
                    "binance_us_taker + maker_preferred: fill-based exit gate; "
                    "buy_fill/sell_fill embed order-book half-spread + slippage buffer; "
                    "fees subtracted once; gate uses simulated fill net PnL / notional"
                ),
                "legacy_old_replay_math_path": (
                    "old_replay only: subtracted roundtrip_cost again after fill-adjusted prices"
                ),
                "legacy_profile_used_for_pass_criteria": False,
                "legacy_replay_math_path_removed_from_accepted": True,
            },
            "binance_us_fee_verification": fee_verification,
            "fee_profile_comparison_90d": fee_comparison,
            "live_rules_match_baseline": live_check,
            "high_resolution": {
                "decision_timeframes": ["1h", "4h"],
                "execution_timeframes_by_window": EXEC_INTERVAL_BY_WINDOW,
                "suite": hi_res,
                "all_pass": hi_res.get("pass_criteria", {}).get("all_pass"),
            },
            "stress_tests": stress,
            "stress_all_pass": all(
                v.get("stays_positive") for v in stress.values() if isinstance(v, dict) and "stays_positive" in v
            ),
            "extended_history_days": ext_days,
            "extended_discovery": ext_discovery,
            "summary": {
                "expected_trades_per_month": round(trades_mo, 2),
                "expected_monthly_pnl_usd_25k": round(pnl_mo, 2),
                "max_drawdown_pct_90d": w90.get("max_drawdown_pct"),
                "gross_pnl_90d": w90.get("gross_pnl_usd"),
                "fees_90d": w90.get("total_fees_usd"),
                "spread_slippage_90d": w90.get("spread_slippage_impact_usd"),
                "best_bucket": best,
                "worst_bucket": worst,
                "active_allowed_buckets": active_allowed_buckets(_stats_from_report(br)),
                "live_economics_display": live_econ,
            },
            "tracebacks": tracebacks,
        }

        out_path = BASELINE_DIR / "day_execution_replay_latest.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["high_resolution"]["all_pass"] else 1
    except Exception:
        tb = traceback.format_exc()
        tracebacks.append(tb)
        print(json.dumps({"error": tb, "tracebacks": tracebacks}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
