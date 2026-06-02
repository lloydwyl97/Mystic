"""
Risk + Edge Governance Layer (24/7).

Sits between ranking and execution. Replaces daily trade-count throttle with:
- Layer 1: Hard risk circuit breakers (drawdown, consecutive losses)
- Layer 2: Market regime filter (chop, spread)
- Layer 3: Adaptive selectivity + pacing + allocation caps

DAILY_LIMIT remains as emergency anomaly fuse only (set high).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field

from backend.config.buy_admission import buy_margin_threshold_active, buy_margin_threshold_core

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration (env, with conservative defaults)
# -----------------------------------------------------------------------------
MAX_DRAWDOWN_24H_PCT = float(os.getenv("MAX_DRAWDOWN_24H_PCT", "0.03"))  # 3% (tier A/B boundary)
MAX_CONSEC_LOSSES = int(os.getenv("MAX_CONSEC_LOSSES", "3"))
LOSS_HOLD_COOLDOWN_MIN = int(os.getenv("LOSS_HOLD_COOLDOWN_MIN", "60"))  # minutes to hold buys after hitting loss cap, then allow in tier C
MAX_CASH_FRACTION_PER_TRADE = float(os.getenv("MAX_CASH_FRACTION_PER_TRADE", "0.30"))  # 30%
MAX_COIN_EXPOSURE_PCT = float(os.getenv("MAX_COIN_EXPOSURE_PCT", "0.25"))  # 25%
MAX_TOTAL_DEPLOYED_PCT = float(os.getenv("MAX_TOTAL_DEPLOYED_PCT", "0.85"))  # 85%
MAX_TRADES_PER_BAR = int(os.getenv("MAX_TRADES_PER_BAR", "2"))
MAX_NOTIONAL_PER_TRADE = float(os.getenv("MAX_NOTIONAL_PER_TRADE", "0"))  # 0 = no hard cap
# Local staged testing only (deploy/core_only_local.env). Default false — production never sets this.
PORTFOLIO_LOCAL_SKIP_DECISION_MIN_NOTIONAL_BLOCK = os.getenv("PORTFOLIO_LOCAL_SKIP_DECISION_MIN_NOTIONAL_BLOCK", "false").lower() == "true"
MIN_CONFIDENCE_REGIME = float(os.getenv("MIN_CONFIDENCE_REGIME", "0.68"))  # Tier A normal min
CHOP_BLOCK_THRESHOLD = float(os.getenv("CHOP_BLOCK_THRESHOLD", "0.65"))  # Block buy if chop > this
# Only block on chop when trend is weak (or edge low). Strong trend overrides chop.
TREND_OVERRIDE_CHOP_MIN = float(os.getenv("TREND_OVERRIDE_CHOP_MIN", "0.5"))  # trend_score >= this -> don't block on chop alone
SPREAD_BLOCK_PCT_OF_TP = float(os.getenv("SPREAD_BLOCK_PCT_OF_TP", "0.15"))  # Block if spread > 15% of TP move
# Default false: local and CI match production (hard gate). Set GOVERNANCE_SHADOW_ONLY=true for advisory-only.
GOVERNANCE_SHADOW_ONLY = os.getenv("GOVERNANCE_SHADOW_ONLY", "false").lower() == "true"
# Explicit: when True, decide() runs for logs only; execute_buy_fifo + bar path both honor this flag.
GOVERNANCE_ENFORCES = not GOVERNANCE_SHADOW_ONLY
# Local staged testing only (set in deploy/core_only_local.env). Does not ship on production droplets.
GOVERNANCE_LOCAL_SKIP_HOLD_CONSEC_LOSSES = os.getenv("GOVERNANCE_LOCAL_SKIP_HOLD_CONSEC_LOSSES", "false").lower() == "true"
# Local staged testing only (deploy/core_only_local.env). Default false — production never sets this.
GOVERNANCE_LOCAL_SKIP_TIER_D_RECOVERY_SYMBOL = os.getenv("GOVERNANCE_LOCAL_SKIP_TIER_D_RECOVERY_SYMBOL", "false").lower() == "true"
# Local staged testing only (deploy/core_only_local.env). Default false — production never sets this.
GOVERNANCE_LOCAL_SKIP_EXPOSURE_CAP = os.getenv("GOVERNANCE_LOCAL_SKIP_EXPOSURE_CAP", "false").lower() == "true"

# Drawdown tiers: A normal, B risk-off, C deep (tiny/selective), D circuit break (pause buys)
# P4: Governance unblocks when drawdown < DRAWDOWN_TIER_D_PCT and consec_losses < MAX_CONSEC_LOSSES
DRAWDOWN_TIER_B_PCT = float(os.getenv("DRAWDOWN_TIER_B_PCT", "0.03"))  # dd >= this -> B
DRAWDOWN_TIER_C_PCT = float(os.getenv("DRAWDOWN_TIER_C_PCT", "0.10"))  # dd >= this -> C
DRAWDOWN_TIER_D_PCT = float(os.getenv("DRAWDOWN_TIER_D_PCT", "0.35"))  # dd >= this -> D (pause)

# Tier C adaptive min confidence (tighten after losses, loosen when losses == 0)
TIER_C_BASE_MIN_CONF = float(os.getenv("TIER_C_BASE_MIN_CONF", "0.72"))
TIER_C_MIN_CONF_FLOOR = float(os.getenv("TIER_C_MIN_CONF_FLOOR", "0.70"))
TIER_C_MIN_CONF_CEILING = float(os.getenv("TIER_C_MIN_CONF_CEILING", "0.75"))
TIER_C_LOSS_STEP = float(os.getenv("TIER_C_LOSS_STEP", "0.01"))
TIER_C_LOSS_CAP = 3  # max consecutive losses that affect adjustment

# Tier D Recovery Mode (dd >= DRAWDOWN_TIER_D_PCT): very limited safe trading instead of full circuit break
TIER_D_RECOVERY_MAX_TRADES = 1
TIER_D_RECOVERY_SIZE_MULT = 0.10
TIER_D_RECOVERY_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "BTC/USDT", "ETH/USDT"})
TIER_D_RECOVERY_CHOP_LIMIT = 0.72  # BTCUSDT/ETHUSDT only: allow chop up to this if trend >= 0.52
TIER_D_RECOVERY_TREND_MIN = 0.52


def _passes_governance_selectivity(c: CandidateInfo, min_conf: float) -> bool:
    """Tier min confidence OR buy_margin clears sleeve admission (same contract as bar ranking)."""
    if c.confidence >= min_conf:
        return True
    if c.buy_margin is None:
        return False
    try:
        bmf = float(c.buy_margin)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(bmf):
        return False
    sleeve_u = (c.sleeve or "").strip().upper()
    thr = buy_margin_threshold_core() if sleeve_u == "CORE" else buy_margin_threshold_active()
    return bmf >= thr


@dataclass
class AccountSnapshot:
    """Authoritative account state for governance."""

    equity: float
    free_usdt: float
    positions_value: float
    open_positions_count: int
    exposure_per_coin: dict[str, float]  # symbol -> notional
    rolling_24h_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    max_positions: int = 8
    loss_hold_until: float | None = None  # UTC timestamp; when set and now < this, hold buys; when expired, allow in tier C
    current_time_utc: float = 0.0  # for cooldown comparison


@dataclass
class CandidateInfo:
    """One candidate as seen by governance (from BuyCandidate)."""

    symbol: str
    composite_score: float
    confidence: float
    trend_score: float
    chop_score: float
    current_price: float
    atr: float
    stop_distance_pct: float = 0.0  # e.g. (entry - stop) / entry
    effective_min_notional: float = 11.0  # Binance US min=10, use 11 for safety
    sleeve: str = "ACTIVE"  # CORE | ACTIVE (aligns buy_margin floors with portfolio engine)
    buy_margin: float | None = None  # P_buy - max(P_hold, P_sell); None = confidence-only gate


@dataclass
class Rejection:
    """Why a candidate was rejected."""

    symbol: str
    reason_code: str
    detail: str = ""


@dataclass
class GovernanceResult:
    """Output of governance decide()."""

    allowed_candidates: list[CandidateInfo] = field(default_factory=list)
    max_trades_this_bar: int = 1
    allocation_plan: dict[str, float] = field(default_factory=dict)  # symbol -> target_notional
    account_hold_reason: str | None = None  # If set, no buys this bar (Layer 1 trip)
    rejections: list[Rejection] = field(default_factory=list)
    dynamic_min_confidence: float = 0.0
    drawdown_tier: str = "A"  # A=normal, B=risk-off, C=deep, D=circuit_break


class RiskGovernor:
    """
    Governance layer: risk + edge gating, allocation caps, explainable reasons.
    """

    def __init__(
        self,
        shadow_only: bool | None = None,
    ) -> None:
        self.shadow_only = GOVERNANCE_SHADOW_ONLY if shadow_only is None else shadow_only
        self._max_drawdown = MAX_DRAWDOWN_24H_PCT
        self._max_consec = MAX_CONSEC_LOSSES
        self._max_cash_frac = MAX_CASH_FRACTION_PER_TRADE
        self._max_coin_exposure = MAX_COIN_EXPOSURE_PCT
        self._max_deployed = MAX_TOTAL_DEPLOYED_PCT
        self._max_trades_per_bar = MAX_TRADES_PER_BAR
        self._chop_block = CHOP_BLOCK_THRESHOLD
        self._trend_override_chop = TREND_OVERRIDE_CHOP_MIN
        self._min_conf_regime = MIN_CONFIDENCE_REGIME

    def decide(
        self,
        account: AccountSnapshot,
        candidates: list[CandidateInfo],
    ) -> GovernanceResult:
        """
        Run three layers; return allowed_candidates, allocation_plan, rejections.
        Drawdown tiers: A (<3%) normal, B (3-10%) risk-off, C (10-25%) deep, D (>=35%) recovery mode (1 trade, 10% size, BTC/ETH only, min_conf 0.78).
        """
        dd = account.rolling_24h_drawdown_pct
        if dd >= DRAWDOWN_TIER_D_PCT:
            min_conf = 0.75
            tier, max_trades, size_mult = "D", TIER_D_RECOVERY_MAX_TRADES, TIER_D_RECOVERY_SIZE_MULT
        elif dd >= DRAWDOWN_TIER_C_PCT:
            tier, max_trades, size_mult = "C", 1, float(os.getenv("TIER_C_SIZE_MULT", "0.25"))
            # Adaptive: tighten after losses, loosen when consec_losses == 0
            adj = TIER_C_LOSS_STEP * min(account.consecutive_losses, TIER_C_LOSS_CAP)
            if account.consecutive_losses == 0:
                adj = -TIER_C_LOSS_STEP
            min_conf = max(
                TIER_C_MIN_CONF_FLOOR,
                min(TIER_C_BASE_MIN_CONF + adj, TIER_C_MIN_CONF_CEILING),
            )
        elif dd >= DRAWDOWN_TIER_B_PCT:
            tier, max_trades, min_conf, size_mult = "B", 1, 0.65, 0.5
        else:
            tier, max_trades, min_conf, size_mult = "A", self._max_trades_per_bar, MIN_CONFIDENCE_REGIME, 1.0

        result = GovernanceResult(
            max_trades_this_bar=TIER_D_RECOVERY_MAX_TRADES if tier == "D" else min(max_trades, max(1, len(candidates))),
            dynamic_min_confidence=min_conf,
            drawdown_tier=tier,
        )

        # Layer 1: Tier D is now Recovery Mode (no hold_reason); consec loss and max_positions still apply
        if account.consecutive_losses >= self._max_consec:
            now_utc = account.current_time_utc
            if account.loss_hold_until is not None and now_utc < account.loss_hold_until:
                if GOVERNANCE_LOCAL_SKIP_HOLD_CONSEC_LOSSES:
                    logger.info(
                        "GOVERNANCE_TELEMETRY HOLD_CONSEC_LOSSES would_block consec_losses=%s>=%s hold_until=%s (GOVERNANCE_LOCAL_SKIP_HOLD_CONSEC_LOSSES=true — not enforcing)",
                        account.consecutive_losses,
                        self._max_consec,
                        account.loss_hold_until,
                    )
                else:
                    result.account_hold_reason = "HOLD_CONSEC_LOSSES"
                    result.rejections.append(Rejection("", "HOLD_CONSEC_LOSSES", f"consec_losses={account.consecutive_losses} >= {self._max_consec}"))
                    _log_governance("Layer1", result, account, candidates)
                    return result
            # Cooldown expired: allow buys in strict (tier C) mode; do not overwrite when already in Tier D recovery
            elif tier != "D":
                size_mult = float(os.getenv("TIER_C_SIZE_MULT", "0.25"))
                adj = TIER_C_LOSS_STEP * min(account.consecutive_losses, TIER_C_LOSS_CAP)
                min_conf = max(TIER_C_MIN_CONF_FLOOR, min(TIER_C_BASE_MIN_CONF + adj, TIER_C_MIN_CONF_CEILING))
                result.max_trades_this_bar = 1
                result.dynamic_min_confidence = min_conf
                result.drawdown_tier = "C"
        # Max positions: enforced in PortfolioEngine._can_open_position before candidates reach here.
        # Avoid duplicate HOLD_MAX_POSITIONS / BUY_BLOCKED_MAX_POSITIONS from governance.

        # Layer 2 + 3: Per-candidate regime + allocation (use tier min_conf for telemetry only)
        for c in candidates:
            if tier == "D" and c.symbol not in TIER_D_RECOVERY_SYMBOLS:
                if GOVERNANCE_LOCAL_SKIP_TIER_D_RECOVERY_SYMBOL:
                    logger.info(
                        "QUALITY_TELEMETRY TIER_D_RECOVERY_SYMBOL would_reject symbol=%s (GOVERNANCE_LOCAL_SKIP_TIER_D_RECOVERY_SYMBOL=true — not enforcing)",
                        c.symbol,
                    )
                else:
                    result.rejections.append(Rejection(c.symbol, "TIER_D_RECOVERY_SYMBOL", "only BTCUSDT, ETHUSDT allowed in Tier D recovery"))
                    continue
            # Layer 2: chop / min-confidence — telemetry only (ML signal is authoritative; do not re-decide)
            if tier == "D" and c.symbol in TIER_D_RECOVERY_SYMBOLS and c.chop_score <= TIER_D_RECOVERY_CHOP_LIMIT and c.trend_score >= TIER_D_RECOVERY_TREND_MIN:
                pass
            else:
                chop_high = c.chop_score >= self._chop_block
                trend_weak = c.trend_score < self._trend_override_chop
                score_low = c.composite_score < 0.15
                if chop_high and (trend_weak or score_low):
                    logger.info(
                        "GOVERNANCE_TELEMETRY chop_would_block symbol=%s chop=%.3f trend=%.3f score=%.3f tier=%s",
                        c.symbol,
                        c.chop_score,
                        c.trend_score,
                        c.composite_score,
                        tier,
                    )
            if not _passes_governance_selectivity(c, result.dynamic_min_confidence):
                logger.info(
                    "GOVERNANCE_TELEMETRY min_conf_would_block symbol=%s conf=%.3f min_conf=%.3f buy_margin=%s",
                    c.symbol,
                    c.confidence,
                    result.dynamic_min_confidence,
                    c.buy_margin,
                )

            # Layer 3: Allocation caps (notional bounds)
            cap_notional = account.free_usdt * self._max_cash_frac
            if MAX_NOTIONAL_PER_TRADE > 0:
                cap_notional = min(cap_notional, MAX_NOTIONAL_PER_TRADE)
            coin_exposure = account.exposure_per_coin.get(c.symbol, 0.0)
            coin_cap = account.equity * self._max_coin_exposure - coin_exposure
            if coin_cap <= 0:
                if GOVERNANCE_LOCAL_SKIP_EXPOSURE_CAP:
                    logger.info(
                        "QUALITY_TELEMETRY BUY_BLOCKED_EXPOSURE_CAP would_reject symbol=%s coin_exposure=%.2f coin_cap=%.4f (GOVERNANCE_LOCAL_SKIP_EXPOSURE_CAP=true — not enforcing)",
                        c.symbol,
                        coin_exposure,
                        coin_cap,
                    )
                    # Keep allocation math finite: fall back to per-trade cash cap only (local paper path).
                    coin_cap = max(float(cap_notional), 1e-9)
                else:
                    result.rejections.append(Rejection(c.symbol, "BUY_BLOCKED_EXPOSURE_CAP", f"coin_exposure={coin_exposure:.2f} at cap"))
                    continue
            deployed = account.positions_value
            remaining_deployed = account.equity * self._max_deployed - deployed
            if remaining_deployed <= 0:
                result.rejections.append(Rejection(c.symbol, "BUY_BLOCKED_TOTAL_DEPLOYED", f"deployed={deployed:.2f} at cap"))
                continue

            target_notional = min(cap_notional, coin_cap, remaining_deployed)
            # BUG #C3 FIX: Use per-symbol min notional from engine constraints instead of hardcoded 10.0
            symbol_min_notional = getattr(c, "effective_min_notional", 11.0)
            below_min_notional = target_notional < symbol_min_notional
            if below_min_notional:
                if PORTFOLIO_LOCAL_SKIP_DECISION_MIN_NOTIONAL_BLOCK:
                    logger.info(
                        "QUALITY_TELEMETRY BUY_BLOCKED_MIN_NOTIONAL would_reject symbol=%s target_notional=%.2f min=%.2f (PORTFOLIO_LOCAL_SKIP_DECISION_MIN_NOTIONAL_BLOCK=true — not enforcing)",
                        c.symbol,
                        target_notional,
                        symbol_min_notional,
                    )
                else:
                    result.rejections.append(Rejection(c.symbol, "BUY_BLOCKED_MIN_NOTIONAL", f"target_notional={target_notional:.2f} < min={symbol_min_notional:.2f}"))
                    continue

            result.allowed_candidates.append(c)
            # Floor allocation to at least min notional (Binance US min=10) so Tier C never caps below executable size
            raw_alloc = target_notional * size_mult
            alloc_floor = symbol_min_notional
            if PORTFOLIO_LOCAL_SKIP_DECISION_MIN_NOTIONAL_BLOCK and below_min_notional:
                alloc_floor = max(target_notional, 1e-9)
            result.allocation_plan[c.symbol] = max(raw_alloc, alloc_floor)
            if len(result.allowed_candidates) >= result.max_trades_this_bar:
                break

        _log_governance("Layer2+3", result, account, candidates)
        return result


def _log_governance(
    layer: str,
    result: GovernanceResult,
    account: AccountSnapshot,
    candidates: list[CandidateInfo],
) -> None:
    """Emit summary log for observability (per bar)."""
    logger.info(
        "GOVERNANCE %s tier=%s candidates=%s allowed=%s max_trades_this_bar=%s hold_reason=%s rejections=%s equity=%.2f free_usdt=%.2f drawdown_24h_pct=%.2f consec_losses=%s",
        layer,
        result.drawdown_tier,
        len(candidates),
        [c.symbol for c in result.allowed_candidates],
        result.max_trades_this_bar,
        result.account_hold_reason or "",
        [(r.symbol, r.reason_code) for r in result.rejections],
        account.equity,
        account.free_usdt,
        account.rolling_24h_drawdown_pct * 100,
        account.consecutive_losses,
    )
    if result.allocation_plan:
        logger.info("GOVERNANCE allocation_plan %s", result.allocation_plan)


logger.info(
    "RiskGovernor startup: GOVERNANCE_SHADOW_ONLY=%s (buy_blocking_from_governor=%s). Default is enforcement; set GOVERNANCE_SHADOW_ONLY=true for advisory-only (log-only) governance.",
    GOVERNANCE_SHADOW_ONLY,
    GOVERNANCE_ENFORCES,
)
