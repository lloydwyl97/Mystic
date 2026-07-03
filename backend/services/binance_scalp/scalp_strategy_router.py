"""Route paper scalp entries to highest-scoring valid strategy setup."""

from __future__ import annotations

import logging
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
    REGIME_RANGE,
    classify_scalp_regime,
)
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.kline_cache import KlineCache, MIN_REGIME_1H_BARS

logger = logging.getLogger(__name__)


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
            }
            for r in ranked_list
        ]

        best_ranked = pick_best_ranked(ranked_list)
        if best_ranked is None:
            meta["hard_block"] = "NO_CANDIDATES"
            return None, signals, meta

        meta["best_rank_score"] = best_ranked.rank_score
        meta["best_setup"] = best_ranked.signal.setup_name
        meta["entry_eligible"] = best_ranked.entry_eligible
        meta["hard_block"] = best_ranked.hard_block
        meta["soft_reason"] = best_ranked.soft_reason

        if best_ranked.entry_eligible:
            entry_sig = prepare_entry_signal(best_ranked, ctx)
            return entry_sig, signals, meta

        # No trade — return best scored signal for status/diagnostics only.
        display_sig = best_ranked.signal
        return display_sig, signals, meta

    def _current_regime(self, symbol: str, epoch: float, bars: list[dict] | None) -> str:
        """Best-effort 1h regime from kline cache (falls back to range)."""
        sym = symbol.strip().upper()
        try:
            bars_1h = self.klines.get_1h(sym) or []
            if len(bars_1h) >= MIN_REGIME_1H_BARS:
                st = classify_scalp_regime(bars_1h, len(bars_1h) - 1)
                if st and st.regime:
                    return st.regime
        except Exception:
            pass
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
            best, all_sigs, meta = self.evaluate_symbol(sym, epoch=epoch, notional_usd=notional_usd)
            snap = self.reader.read(sym)
            if snap is None:
                continue
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
            }
            rows.append(row)

        rows.sort(
            key=lambda r: (
                -int(bool(r.get("entry_eligible"))),
                -float(r.get("rank_score") or 0.0),
                float(getattr(r.get("signal"), "spread_pct", 0) or 0),
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
