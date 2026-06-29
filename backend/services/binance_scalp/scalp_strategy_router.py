"""Route paper scalp entries to highest-scoring valid strategy setup."""

from __future__ import annotations

import logging
from typing import Any

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot, ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics, MomentumTracker
from backend.services.binance_scalp.scalp_regime_classifier import (
    REGIME_RANGE,
    STRATEGY_NATIVE_REGIMES,
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
    ) -> tuple[ScalpSetupSignal | None, list[ScalpSetupSignal]]:
        sym = symbol.strip().upper()
        snap = snap or self.reader.read(sym)
        if snap is None:
            return None, []
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

        signals: list[ScalpSetupSignal] = []

        # Regime-aware filter: only evaluate strategies whose native regime matches current 1h regime
        regime = self._current_regime(sym, epoch, bars)
        native_for_regime = {name for name, regs in STRATEGY_NATIVE_REGIMES.items() if regime in regs}

        for strategy in enabled_strategies(self.config):
            if strategy.name not in native_for_regime:
                # Report a proper rejected signal for diagnostics/status (must support .as_dict())
                signals.append(
                    ScalpSetupSignal(
                        symbol=sym,
                        side="BUY",
                        score=0.0,
                        setup_name=strategy.name,
                        confidence=0.0,
                        entry_reason="",
                        invalidation_reason=None,
                        required_target_pct=0.0,
                        expected_move_pct=0.0,
                        spread_pct=0.0,
                        impact_pct=0.0,
                        depth_sufficient=False,
                        limit_buy_price=0.0,
                        passed=False,
                        reject_reason=f"REGIME_BLOCKED:{regime}",
                        setup_context={"regime_blocked": regime},
                    )
                )
                continue
            try:
                sig = strategy.evaluate(ctx)
                signals.append(sig)
            except Exception as exc:
                logger.warning("strategy %s failed %s: %s", getattr(strategy, "name", "?"), sym, exc)
                # Always append a proper rejected signal so callers can rely on .passed / .reject_reason / .as_dict()
                signals.append(
                    ScalpSetupSignal(
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
                )

        passed = [s for s in signals if getattr(s, "passed", False)]
        if not passed:
            return None, signals
        best = max(passed, key=lambda s: (getattr(s, "score", 0), getattr(s, "confidence", 0), -getattr(s, "spread_pct", 0)))
        return best, signals

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
        """Evaluate all products; return ranked entry candidates."""
        rows: list[dict[str, Any]] = []
        for sym in self.config.products:
            best, all_sigs = self.evaluate_symbol(sym, epoch=epoch, notional_usd=notional_usd)
            if best is None:
                continue
            snap = self.reader.read(sym)
            if snap is None:
                continue
            rows.append(
                {
                    "symbol": sym,
                    "snap": snap,
                    "signal": best,
                    "all_signals": [s.as_dict() for s in all_sigs],
                }
            )
        rows.sort(key=lambda r: (-r["signal"].score, r["signal"].spread_pct))
        return rows

    def strategy_inventory(self) -> dict[str, Any]:
        disabled = self.config.disabled_strategies
        return {
            "all": list(STRATEGY_NAMES),
            "enabled": [s.name for s in enabled_strategies(self.config)],
            "disabled": sorted(disabled),
        }
