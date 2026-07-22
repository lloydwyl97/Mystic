"""Route paper scalp entries to highest-scoring valid strategy setup."""

from __future__ import annotations

import logging
import os
from typing import Any

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot, ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics, MomentumTracker
from backend.services.binance_scalp.scalp_candidate_ranking import (
    RankedCandidate,
    pick_best_ranked,
    prepare_entry_signal,
    rank_setup_signal,
)
from backend.services.binance_scalp.scalp_regime_classifier import (
    REGIME_DATA_MISSING,
    REGIME_RANGE,
    classify_scalp_regime,
)
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.kline_cache import KlineCache, MIN_REGIME_1H_BARS

logger = logging.getLogger(__name__)


def _mtf_confirmation_gate_enabled() -> bool:
    """When true (default), long SCALP entries require non-down 5m (and 15m) trend."""
    return str(os.getenv("SCALP_MTF_CONFIRMATION_GATE_ENABLED", "true")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _mtf_require_15m() -> bool:
    return str(os.getenv("SCALP_MTF_REQUIRE_15M", "true")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class ScalpStrategyRouter:
    def __init__(
        self,
        *,
        config: ScalpConfig,
        econ: ScalpEconomics,
        reader: ScalpMarketReader,
        momentum: MomentumTracker,
        klines: KlineCache | None = None,
    ) -> None:
        self.config = config
        self.econ = econ
        self.reader = reader
        self.momentum = momentum
        self.klines = klines or KlineCache()

    def evaluate_symbol(
        self,
        symbol: str,
        *,
        epoch: float,
        notional_usd: float,
        snap: MarketSnapshot | None = None,
        mom: MomentumDiagnostics | None = None,
        bars: list[dict] | None = None,
    ) -> tuple[ScalpSetupSignal | None, list[ScalpSetupSignal], dict[str, Any]]:
        sym = symbol.strip().upper()
        meta: dict[str, Any] = {
            "symbol": sym,
            "regime": REGIME_RANGE,
            "ranked": [],
            "best_rank_score": 0.0,
            "entry_eligible": False,
            "hard_block": None,
            "best_setup": None,
        }
        snap = snap or self.reader.read(sym)
        if snap is None:
            meta["hard_block"] = "STALE_DATA"
            return None, [], meta
        if mom is None:
            self.momentum.record(sym, epoch, snap.best_bid, snap.mid)
            mom = self.momentum.diagnostics(sym, epoch, snap.best_bid, snap.mid)
        bars = bars if bars is not None else self.klines.get(sym)

        ctx = StrategyMarketContext(
            symbol=sym,
            snap=snap,
            mom=mom,
            bars_1m=bars,
            econ=self.econ,
            config=self.config,
            notional_usd=notional_usd,
        )

        regime = self._current_regime(sym, epoch, bars)
        meta["regime"] = regime
        if regime == REGIME_DATA_MISSING:
            meta["hard_block"] = "REGIME_DATA_MISSING"
            return None, [], meta
        # Multi-timeframe confirmation between 1m setups and 1h regime.
        # When SCALP_MTF_CONFIRMATION_GATE_ENABLED (default true), a down 5m/15m
        # trend blocks entry_eligible. Missing history (None) does not block.
        mtf_5m_trend_pct, mtf_5m_aligned = self._mtf_trend_confirmation(sym, "5m")
        mtf_15m_trend_pct, mtf_15m_aligned = self._mtf_trend_confirmation(sym, "15m")
        meta["mtf_5m_trend_pct"] = mtf_5m_trend_pct
        meta["mtf_5m_aligned"] = mtf_5m_aligned
        meta["mtf_15m_trend_pct"] = mtf_15m_trend_pct
        meta["mtf_15m_aligned"] = mtf_15m_aligned

        signals: list[ScalpSetupSignal] = []
        ranked_list: list[RankedCandidate] = []

        # Ranking engine: evaluate every enabled strategy; soft misses score, hard safety blocks.
        for strategy in enabled_strategies(self.config):
            try:
                sig = strategy.evaluate(ctx)
            except Exception as exc:
                logger.warning("strategy %s failed %s: %s", getattr(strategy, "name", "?"), sym, exc)
                sig = ScalpSetupSignal(
                    symbol=sym,
                    side="BUY",
                    score=0.0,
                    setup_name=getattr(strategy, "name", "unknown"),
                    confidence=0.0,
                    entry_reason="",
                    invalidation_reason=None,
                    required_target_pct=0.0,
                    expected_move_pct=0.0,
                    spread_pct=getattr(ctx.snap, "spread_pct", 0.0),
                    impact_pct=0.0,
                    depth_sufficient=False,
                    limit_buy_price=getattr(ctx.snap, "best_ask", 0.0),
                    passed=False,
                    reject_reason=f"STRATEGY_ERROR:{exc}",
                    setup_context={"error": str(exc)[:200]},
                )
            signals.append(sig)
            ranked_list.append(rank_setup_signal(sig, regime=regime, ctx=ctx))

        meta["ranked"] = [
            {
                "setup_name": r.signal.setup_name,
                "rank_score": r.rank_score,
                "entry_eligible": r.entry_eligible,
                "hard_block": r.hard_block,
                "soft_reason": r.soft_reason,
                "passed": r.signal.passed,
                "regime_native": r.regime_native,
                "reachability_surplus": r.reachability_surplus,
                "selection_confidence": r.selection_confidence,
                # Diagnostics (existing values computed in rank_setup_signal)
                "base_score": getattr(r, "base_score", None),
                "momentum_boost": getattr(r, "momentum_boost", None),
                "reachability_multiplier": getattr(r, "reachability_multiplier", None),
                "expected_move_pct": getattr(r.signal, "expected_move_pct", None),
                "required_target_pct": getattr(r.signal, "required_target_pct", None),
                "target_gap_pct": getattr(r, "target_gap_pct", None),
                "regime": regime,
                "memory_delta": None,  # filled by enrichment in evaluate_all for global pick
                "recent_win_rate": None,
                "m15": None,
                "m30": None,
                "m60": None,
            }
            for r in ranked_list
        ]

        best_ranked = pick_best_ranked(ranked_list)
        if best_ranked is None:
            meta["hard_block"] = "NO_CANDIDATES"
            return None, signals, meta

        meta["reachability_surplus"] = best_ranked.reachability_surplus
        meta["selection_confidence"] = best_ranked.selection_confidence
        meta["best_rank_score"] = best_ranked.rank_score
        meta["best_setup"] = best_ranked.signal.setup_name
        entry_eligible = bool(best_ranked.entry_eligible)
        soft_reason = best_ranked.soft_reason
        hard_block = best_ranked.hard_block
        selection_confidence = best_ranked.selection_confidence

        if entry_eligible and _mtf_confirmation_gate_enabled():
            if mtf_5m_aligned is False:
                entry_eligible = False
                soft_reason = "MTF_5M_NOT_ALIGNED"
                selection_confidence = "mtf_confirmation_blocked"
            elif _mtf_require_15m() and mtf_15m_aligned is False:
                entry_eligible = False
                soft_reason = "MTF_15M_NOT_ALIGNED"
                selection_confidence = "mtf_confirmation_blocked"

        meta["entry_eligible"] = entry_eligible
        meta["hard_block"] = hard_block
        meta["soft_reason"] = soft_reason
        meta["selection_confidence"] = selection_confidence

        if entry_eligible:
            entry_sig = prepare_entry_signal(best_ranked, ctx)
            return entry_sig, signals, meta

        # No trade — return best scored signal for status/diagnostics only.
        display_sig = best_ranked.signal
        return display_sig, signals, meta

    def _mtf_trend_confirmation(self, symbol: str, interval: str) -> tuple[float | None, bool | None]:
        """Best-effort higher-TF trend confirmation for long-only SCALP.

        Compares mean close of the last 3 bars vs the 3 before them.
        "Aligned" means trend_pct > 0. Returns (None, None) when history is
        insufficient — callers must not treat that as flat/down.
        """
        try:
            sym = symbol.strip().upper()
            if interval == "5m":
                bars = self.klines.get_5m(sym) or []
            elif interval == "15m":
                bars = self.klines.get_15m(sym) or []
            else:
                return None, None
            if len(bars) < 6:
                return None, None
            recent = bars[-3:]
            prior = bars[-6:-3]
            recent_mean = sum(float(b["close"]) for b in recent) / 3.0
            prior_mean = sum(float(b["close"]) for b in prior) / 3.0
            if prior_mean <= 0:
                return None, None
            trend_pct = (recent_mean - prior_mean) / prior_mean
            return trend_pct, trend_pct > 0.0
        except Exception:
            return None, None

    def _current_regime(self, symbol: str, epoch: float, bars: list[dict] | None) -> str:
        """1h regime from kline cache. Returns REGIME_DATA_MISSING on fetch failure (hard-blocks entry)."""
        sym = symbol.strip().upper()
        try:
            bars_1h = self.klines.get_1h(sym) or []
            if len(bars_1h) >= MIN_REGIME_1H_BARS:
                st = classify_scalp_regime(bars_1h, len(bars_1h) - 1)
                if st and st.regime:
                    return st.regime
        except Exception as e:
            logger.warning("[SCALP_REGIME] Failed to determine regime for %s: %s", sym, e)
            return REGIME_DATA_MISSING
        return REGIME_RANGE

    def evaluate_all(
        self,
        *,
        epoch: float,
        notional_usd: float,
    ) -> list[dict[str, Any]]:
        """Evaluate all products; return ranked entry candidates (eligible first)."""
        rows: list[dict[str, Any]] = []
        for sym in self.config.products:
            # Fetch snapshot once and pass it through — evaluate_symbol() would
            # otherwise independently re-fetch the same symbol's depth via
            # self.reader.read(sym) internally (snap = snap or self.reader.read(sym)),
            # doubling Binance /api/v3/depth calls per symbol per cycle for no
            # benefit (same 5s cycle, no ranking/scoring logic depends on which
            # fetch is used). Pure duplicate-call elimination; no logic change.
            snap = self.reader.read(sym)
            if snap is None:
                continue
            best, all_sigs, meta = self.evaluate_symbol(sym, epoch=epoch, notional_usd=notional_usd, snap=snap)
            if best is None and not meta.get("ranked"):
                continue
            row = {
                "symbol": sym,
                "snap": snap,
                "signal": best,
                "all_signals": [s.as_dict() for s in all_sigs],
                "rank_meta": meta,
                "rank_score": float(meta.get("best_rank_score") or 0.0),
                "entry_eligible": bool(meta.get("entry_eligible")),
                "hard_block": meta.get("hard_block"),
                "best_setup": meta.get("best_setup"),
                "soft_reason": meta.get("soft_reason"),
                "reachability_surplus": meta.get("reachability_surplus"),
                "selection_confidence": meta.get("selection_confidence"),
            }
            rows.append(row)

        rows.sort(
            key=lambda r: (
                -int(bool(r.get("entry_eligible"))),
                -float(r.get("rank_score") or 0.0),
                -float((r.get("rank_meta") or {}).get("reachability_surplus") or 0.0),
            )
        )
        return rows

    def strategy_inventory(self) -> dict[str, Any]:
        disabled = self.config.disabled_strategies
        return {
            "all": list(STRATEGY_NAMES),
            "enabled": [s.name for s in enabled_strategies(self.config)],
            "disabled": sorted(disabled),
        }
