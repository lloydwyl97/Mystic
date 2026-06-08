"""Route paper scalp entries to highest-scoring valid strategy setup."""

from __future__ import annotations

import logging
from typing import Any

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot, ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics, MomentumTracker
from backend.services.binance_scalp.strategies import STRATEGY_NAMES, enabled_strategies
from backend.services.binance_scalp.strategies.base import ScalpSetupSignal, StrategyMarketContext
from backend.services.binance_scalp.strategies.kline_cache import KlineCache

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
        for strategy in enabled_strategies(self.config):
            try:
                sig = strategy.evaluate(ctx)
                signals.append(sig)
            except Exception as exc:
                logger.warning("strategy %s failed %s: %s", getattr(strategy, "name", "?"), sym, exc)

        passed = [s for s in signals if s.passed]
        if not passed:
            return None, signals
        best = max(passed, key=lambda s: (s.score, s.confidence, -s.spread_pct))
        return best, signals

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
