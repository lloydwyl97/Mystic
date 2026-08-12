"""
ScalpLiveEngine — live execution counterpart to BinanceScalpPaperEngine.
Shares signal generation and candidate ranking from paper_engine but
routes order placement through ScalpOrderBridge.

Only active when: SCALP_LIVE=true, SCALP_LIVE_ARMED=true.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.scalp_order_bridge import ScalpFill, ScalpOrderBridge

logger = logging.getLogger(__name__)

SCALP_LIVE_ARMED = os.getenv("SCALP_LIVE_ARMED", "false").lower() == "true"
SCALP_LIVE_MAX_NOTIONAL = float(os.getenv("SCALP_LIVE_MAX_NOTIONAL", "50.0"))  # Hard cap
SCALP_LIVE_MAX_OPEN = int(os.getenv("SCALP_LIVE_MAX_OPEN", "2"))  # Conservative default


class ScalpLiveEngine:
    """
    Live SCALP execution engine.

    This engine:
    - Uses the same signal/ranking logic as BinanceScalpPaperEngine
    - Routes fills to ScalpOrderBridge for real Binance.US execution
    - Maintains its own live position ledger
    - Has independent circuit breakers

    NOT YET CONNECTED TO paper_engine signals — Phase 2 wiring.
    """

    def __init__(self, config: ScalpConfig) -> None:
        self._config = config
        self._bridge: Optional[ScalpOrderBridge] = None
        self._open_positions: dict[str, ScalpFill] = {}  # symbol -> entry ScalpFill
        self._daily_pnl: float = 0.0
        self._armed = False

        if not config.scalp_live:
            raise RuntimeError("ScalpLiveEngine instantiated with scalp_live=False. Use paper engine.")

    # ------------------------------------------------------------------
    # Arming
    # ------------------------------------------------------------------

    def arm(self, api_key: str, api_secret: str) -> None:
        """Arm the live engine. Creates and arms the order bridge."""
        self._bridge = ScalpOrderBridge(api_key, api_secret)
        self._bridge.arm()
        self._armed = True
        logger.warning("[SCALP_LIVE] Engine ARMED")

    def disarm(self) -> None:
        if self._bridge is not None:
            self._bridge.disarm()
        self._armed = False
        logger.warning("[SCALP_LIVE] Engine disarmed")

    def is_armed(self) -> bool:
        return self._armed and self._bridge is not None

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------

    async def execute_live_entry(self, symbol: str, notional: float, ref_price: float) -> bool:
        """
        Execute a live scalp entry.
        Returns True if fill succeeded and position opened.
        """
        if not self.is_armed():
            logger.error("[SCALP_LIVE] execute_live_entry called but engine not armed")
            return False

        if self._is_circuit_open():
            logger.warning("[SCALP_LIVE] Circuit open — blocking entry for %s", symbol)
            return False

        # Hard notional cap
        notional = min(notional, SCALP_LIVE_MAX_NOTIONAL)

        if len(self._open_positions) >= SCALP_LIVE_MAX_OPEN:
            logger.debug("[SCALP_LIVE] Max open positions (%d) reached", SCALP_LIVE_MAX_OPEN)
            return False

        if symbol in self._open_positions:
            logger.debug("[SCALP_LIVE] Already have position in %s", symbol)
            return False

        assert self._bridge is not None  # guarded by is_armed()
        fill = await self._bridge.place_buy(symbol, notional, ref_price)
        if fill is None:
            logger.error("[SCALP_LIVE] BUY fill failed for %s", symbol)
            return False

        self._open_positions[symbol] = fill
        logger.info(
            "[SCALP_LIVE] BUY filled %s qty=%.8f @ %.4f fee=%.4f",
            symbol,
            fill.qty,
            fill.fill_price,
            fill.fee_usdt,
        )
        return True

    async def execute_live_exit(
        self,
        symbol: str,
        ref_price: float,
        *,
        is_urgent_exit: bool = False,
        spread_pct: float = 0.0,
        adverse_selection_risk: float = 0.0,
    ) -> Optional[float]:
        """
        Execute a live scalp exit.
        Returns realized PnL if successful, None if failed.

        Pass `is_urgent_exit=True` for catastrophic-stop/circuit-breaker/
        max-hold exits — forces a guaranteed-fill MARKET order (item p21).
        """
        if not self.is_armed():
            return None

        entry_fill = self._open_positions.get(symbol)
        if entry_fill is None:
            logger.warning("[SCALP_LIVE] exit called for %s but no open position", symbol)
            return None

        assert self._bridge is not None
        fill = await self._bridge.place_sell(
            symbol,
            entry_fill.qty,
            ref_price,
            is_urgent_exit=is_urgent_exit,
            spread_pct=spread_pct,
            adverse_selection_risk=adverse_selection_risk,
        )
        if fill is None:
            logger.error("[SCALP_LIVE] SELL fill failed for %s — position remains open", symbol)
            return None

        entry_cost = entry_fill.qty * entry_fill.fill_price + entry_fill.fee_usdt
        exit_proceeds = fill.qty * fill.fill_price - fill.fee_usdt
        pnl = exit_proceeds - entry_cost

        self._daily_pnl += pnl
        del self._open_positions[symbol]
        logger.info(
            "[SCALP_LIVE] SELL filled %s qty=%.8f @ %.4f pnl=%.4f",
            symbol,
            fill.qty,
            fill.fill_price,
            pnl,
        )
        return pnl

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _is_circuit_open(self) -> bool:
        """Returns True if circuit breaker should halt new entries.

        Trips when daily realized PnL falls below the configured loss limit,
        scaled by live max notional and max open positions.
        """
        limit = self._config.daily_loss_limit_pct * SCALP_LIVE_MAX_NOTIONAL * SCALP_LIVE_MAX_OPEN
        if self._daily_pnl < -limit:
            logger.warning(
                "[SCALP_LIVE] Daily loss circuit: pnl=%.2f limit=-%.2f",
                self._daily_pnl,
                limit,
            )
            return True
        return False

    def reset_daily_pnl(self) -> None:
        """Call at start of new trading day to reset the daily loss counter."""
        self._daily_pnl = 0.0
        logger.info("[SCALP_LIVE] Daily PnL counter reset")

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            "armed": self._armed,
            "open_positions": len(self._open_positions),
            "daily_pnl": round(self._daily_pnl, 4),
            "circuit_open": self._is_circuit_open(),
            "symbols": list(self._open_positions.keys()),
            "live_max_notional": SCALP_LIVE_MAX_NOTIONAL,
            "live_max_open": SCALP_LIVE_MAX_OPEN,
        }
