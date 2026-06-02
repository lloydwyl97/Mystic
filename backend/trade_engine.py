#!/usr/bin/env python3
"""
Unified Trade Decision Engine
Combines all three tiers of signals and makes trading decisions every 3-10 seconds
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class CoinState:
    symbol: str
    last_price: float
    last_volume: float
    rsi: float
    macd: dict[str, float]
    mystic_alignment_score: float
    is_active_buy_signal: bool
    is_active_sell_signal: bool
    signal_strength: SignalStrength
    confidence: float
    last_update: str
    api_source: str


@dataclass
class TradeDecision:
    symbol: str
    action: TradeAction
    confidence: float
    price: float
    reason: str
    tier1_signals: dict[str, Any]
    tier2_signals: dict[str, Any]
    tier3_signals: dict[str, Any]
    timestamp: str


class TradeEngine:
    def __init__(self, redis_client: Any) -> None:
        self.redis_client = redis_client
        self.is_running = False

        # Engine Configuration
        self.config = {
            "decision_interval": 5,  # 3-10 seconds
            "cache_ttl": 60,  # 1 minute
            "min_confidence": 0.75,  # Match AI_CONFIDENCE_THRESHOLD config
            "max_confidence": 0.95,
            "price_deviation_threshold": 0.02,  # 2%
            "volume_spike_threshold": 0.2,  # 20%
            "momentum_flip_threshold": 0.05,  # 5%
        }

        # Coin state tracking
        self.coin_states: dict[str, CoinState] = {}

        # Trading thresholds
        self.thresholds = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_bullish": 0.001,
            "macd_bearish": -0.001,
            "cosmic_alignment_min": 60,
            "volatility_max": 80,
        }

        logger.info("Trade Engine initialized")

    # -------------------------
    # Internal helpers
    # -------------------------

    def _decode_bytes(self, data: Any) -> Any:
        """Decode Redis bytes to str if necessary."""
        if isinstance(data, bytes):
            try:
                return data.decode("utf-8")
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                # last resort: try latin-1 to avoid crashes
                return data.decode("latin-1", errors="ignore")
        return data

    def _get_json(self, key: str) -> dict[str, Any]:
        """Safe JSON fetch from Redis."""
        try:
            raw = self.redis_client.get(key)
            if not raw:
                return {}
            text = self._decode_bytes(raw)
            if isinstance(text, (dict, list)):
                # Some clients may already return parsed
                return text  # type: ignore[return-value]
            return json.loads(text)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error reading JSON for key '{key}': {e}")
            return {}

    def _to_jsonable(self, obj: Any) -> Any:
        """Recursively convert Enums and dataclasses for JSON serialization."""
        if isinstance(obj, Enum):
            return obj.value
        if is_dataclass(obj):
            return {k: self._to_jsonable(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: self._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(v) for v in obj]
        return obj

    def _set_json_with_ttl(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Safe JSON write to Redis with TTL."""
        try:
            payload = json.dumps(self._to_jsonable(value))
            # Redis-py expects seconds then value for setex(name, time, value)
            self.redis_client.setex(key, ttl_seconds, payload)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error writing JSON for key '{key}': {e}")

    # -------------------------
    # Signal fetchers
    # -------------------------

    async def get_tier1_signals(self) -> dict[str, Any]:
        """Get Tier 1 signals from cache"""
        return self._get_json("tier1_signals")

    async def get_tier2_signals(self) -> dict[str, Any]:
        """Get Tier 2 signals from cache"""
        return self._get_json("tier2_indicators")

    async def get_tier3_signals(self) -> dict[str, Any]:
        """Get Tier 3 signals from cache"""
        return self._get_json("cosmic_signals")

    # -------------------------
    # Core calculations
    # -------------------------

    def calculate_signal_strength(
        self,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        tier3: dict[str, Any],
    ) -> SignalStrength:
        """Calculate overall signal strength from all tiers"""
        try:
            strength_score = 0
            factors = 0

            # Tier 1 factors (price momentum)
            if "change_1m" in tier1:
                try:
                    momentum = abs(float(tier1["change_1m"]))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    momentum = 0.0
                if momentum > 5:
                    strength_score += 3
                elif momentum > 2:
                    strength_score += 2
                elif momentum > 1:
                    strength_score += 1
                factors += 1

            # Tier 2 factors (technical indicators)
            if "rsi" in tier2:
                try:
                    rsi = float(tier2["rsi"])
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    rsi = 50.0
                if rsi < 20 or rsi > 80:
                    strength_score += 3
                elif rsi < 30 or rsi > 70:
                    strength_score += 2
                factors += 1

            macd = tier2.get("macd") or {}
            if isinstance(macd, dict) and "histogram" in macd:
                try:
                    macd_hist = abs(float(macd.get("histogram", 0)))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    macd_hist = 0.0
                if macd_hist > 0.01:
                    strength_score += 2
                elif macd_hist > 0.005:
                    strength_score += 1
                factors += 1

            # Tier 3 factors (cosmic alignment)
            if "cosmic_timing_score" in tier3:
                try:
                    cosmic_score = float(tier3["cosmic_timing_score"])
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    cosmic_score = 50.0
                if cosmic_score > 80:
                    strength_score += 2
                elif cosmic_score > 60:
                    strength_score += 1
                factors += 1

            # Calculate average strength
            if factors > 0:
                avg_strength = strength_score / factors

                if avg_strength >= 2.5:
                    return SignalStrength.VERY_STRONG
                if avg_strength >= 2.0:
                    return SignalStrength.STRONG
                if avg_strength >= 1.5:
                    return SignalStrength.MODERATE
                return SignalStrength.WEAK
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error calculating signal strength: {e}")
            return SignalStrength.WEAK
        else:
            return SignalStrength.WEAK

    def calculate_confidence(
        self,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        tier3: dict[str, Any],
    ) -> float:
        """Calculate trading confidence from all tiers"""
        try:
            confidence_factors: list[float] = []

            # Price stability (Tier 1)
            if "price" in tier1:
                confidence_factors.append(0.8)  # Base confidence for price data

            # Technical indicator agreement (Tier 2)
            macd = tier2.get("macd") or {}
            if "rsi" in tier2 and isinstance(macd, dict):
                try:
                    rsi = float(tier2["rsi"])
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    rsi = 50.0
                try:
                    macd_hist = float(macd.get("histogram", 0.0))
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    macd_hist = 0.0

                # Check if RSI and MACD agree
                rsi_bullish = rsi < 70  # (kept as in original logic)
                macd_bullish = macd_hist > 0

                if rsi_bullish == macd_bullish:
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.6)

            # Cosmic alignment (Tier 3)
            if "cosmic_timing_score" in tier3:
                try:
                    cosmic_score = float(tier3["cosmic_timing_score"])
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    cosmic_score = 50.0
                if cosmic_score > 70:
                    confidence_factors.append(0.85)
                elif cosmic_score > 50:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.5)

            # Calculate average confidence
            if confidence_factors:
                avg_confidence = sum(confidence_factors) / len(confidence_factors)
                return min(
                    float(self.config["max_confidence"]),
                    max(float(self.config["min_confidence"]), avg_confidence),
                )

            return float(self.config["min_confidence"])

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error calculating confidence: {e}")
            return float(self.config["min_confidence"])

    def determine_trade_action(
        self,
        symbol: str,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        tier3: dict[str, Any],
    ) -> tuple[TradeAction, str]:
        """Determine trade action based on aggregated signals."""
        try:
            buy_signals = 0
            sell_signals = 0
            reasons: list[str] = []

            # Tier 1: momentum / price change
            if "change_1m" in tier1:
                try:
                    change_1m = float(tier1.get("change_1m", 0.0) or 0.0)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    change_1m = 0.0
                # Positive momentum => buy, negative => sell
                if change_1m > 1.0:
                    buy_signals += 1
                    reasons.append(f"Momentum up {change_1m}%")
                elif change_1m < -1.0:
                    sell_signals += 1
                    reasons.append(f"Momentum down {change_1m}%")

            # Tier 1: volume spike
            try:
                volume = float(tier1.get("volume_1m", 0.0) or 0.0)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                volume = 0.0
            last_volume = 0.0
            if symbol in self.coin_states:
                try:
                    last_volume = float(self.coin_states[symbol].last_volume or 0.0)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    last_volume = 0.0
            if last_volume > 0 and volume > 0:
                try:
                    vol_change = (volume - last_volume) / last_volume
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    vol_change = 0.0
                if vol_change > float(self.config.get("volume_spike_threshold", 0.2)):
                    buy_signals += 1
                    reasons.append(f"Volume spike {vol_change:.2f}")
                elif vol_change < -float(self.config.get("volume_spike_threshold", 0.2)):
                    sell_signals += 1
                    reasons.append(f"Volume drop {vol_change:.2f}")

            # Tier 2: RSI
            if "rsi" in tier2:
                try:
                    rsi = float(tier2.get("rsi", 50.0) or 50.0)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    rsi = 50.0
                if rsi <= self.thresholds.get("rsi_oversold", 30):
                    buy_signals += 1
                    reasons.append(f"RSI oversold {rsi}")
                if rsi >= self.thresholds.get("rsi_overbought", 70):
                    sell_signals += 1
                    reasons.append(f"RSI overbought {rsi}")

            # Tier 2: MACD histogram
            macd = tier2.get("macd") or {}
            if isinstance(macd, dict):
                try:
                    macd_hist = float(macd.get("histogram", 0.0) or 0.0)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    macd_hist = 0.0
                if macd_hist > float(self.thresholds.get("macd_bullish", 0.001)):
                    buy_signals += 1
                    reasons.append(f"MACD bullish hist {macd_hist}")
                if macd_hist < float(self.thresholds.get("macd_bearish", -0.001)):
                    sell_signals += 1
                    reasons.append(f"MACD bearish hist {macd_hist}")

            # Tier 3: cosmic alignment - can bias towards buy if strong
            if "cosmic_timing_score" in tier3:
                try:
                    cosmic_score = float(tier3.get("cosmic_timing_score", 50.0) or 50.0)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    cosmic_score = 50.0
                if cosmic_score >= self.thresholds.get("cosmic_alignment_min", 60):
                    # reinforce majority signal or add a mild buy bias
                    if buy_signals >= sell_signals:
                        buy_signals += 1
                        reasons.append(f"Cosmic alignment strong {cosmic_score}")
                    else:
                        # if sells dominate, slightly reduce conviction to discourage buy
                        reasons.append(f"Cosmic alignment {cosmic_score} (offset)")

            # Determine final action based on signal balance
            if buy_signals > sell_signals and buy_signals >= 2:
                return TradeAction.BUY, " | ".join(reasons) if reasons else "Buy signals"
            if sell_signals > buy_signals and sell_signals >= 2:
                return TradeAction.SELL, " | ".join(reasons) if reasons else "Sell signals"
            return TradeAction.HOLD, "Insufficient signal strength" if not reasons else " | ".join(reasons)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error determining trade action for {symbol}: {e}")
            return TradeAction.HOLD, f"Error in analysis: {e!s}"

    # -------------------------
    # State updates & decisions
    # -------------------------

    async def update_coin_state(
        self,
        symbol: str,
        tier1: dict[str, Any],
        tier2: dict[str, Any],
        tier3: dict[str, Any],
    ):
        """Update coin state with latest signals"""
        try:
            # Extract data from tiers
            price = float(tier1.get("price", 0.0) or 0.0)
            volume = float(tier1.get("volume_1m", 0.0) or 0.0)
            rsi = float(tier2.get("rsi", 50.0) or 50.0)
            macd = tier2.get("macd") or {
                "macd_line": 0.0,
                "signal_line": 0.0,
                "histogram": 0.0,
            }
            if not isinstance(macd, dict):
                macd = {"macd_line": 0.0, "signal_line": 0.0, "histogram": 0.0}
            cosmic_score = float(tier3.get("cosmic_timing_score", 50.0) or 50.0)
            api_source = str(tier1.get("api_source", "unknown"))

            # Calculate signal strength and confidence
            signal_strength = self.calculate_signal_strength(tier1, tier2, tier3)
            confidence = self.calculate_confidence(tier1, tier2, tier3)

            # Determine trade action
            action, _ = self.determine_trade_action(symbol, tier1, tier2, tier3)

            # Update coin state
            self.coin_states[symbol] = CoinState(
                symbol=symbol,
                last_price=price,
                last_volume=volume,
                rsi=rsi,
                macd=macd,  # type: ignore[arg-type]
                mystic_alignment_score=cosmic_score,
                is_active_buy_signal=(action == TradeAction.BUY),
                is_active_sell_signal=(action == TradeAction.SELL),
                signal_strength=signal_strength,
                confidence=confidence,
                last_update=datetime.now(timezone.utc).isoformat(),
                api_source=api_source,
            )

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error updating coin state for {symbol}: {e}")

    async def generate_trade_decisions(self) -> list[TradeDecision]:
        """Generate trade decisions for all coins"""
        try:
            # Get all signal tiers
            tier1_signals = await self.get_tier1_signals()
            tier2_signals = await self.get_tier2_signals()
            tier3_signals = await self.get_tier3_signals()

            decisions: list[TradeDecision] = []

            prices_map = tier1_signals.get("prices", {})
            if not isinstance(prices_map, dict):
                prices_map = {}

            indicators_map = tier2_signals.get("indicators", {}) if isinstance(tier2_signals, dict) else {}
            if not isinstance(indicators_map, dict):
                indicators_map = {}

            # Some caches may put cosmic values directly; support both
            cosmic_block = {}
            if isinstance(tier3_signals, dict):
                cosmic_block = tier3_signals.get("cosmic_signals", tier3_signals)
                if not isinstance(cosmic_block, dict):
                    cosmic_block = {}

            # Process each coin
            for symbol in prices_map:
                try:
                    tier1 = prices_map.get(symbol, {}) or {}
                    tier2 = indicators_map.get(symbol, {}) or {}
                    tier3 = cosmic_block or {}

                    # Update coin state
                    await self.update_coin_state(symbol, tier1, tier2, tier3)

                    # Generate trade decision
                    action, reason = self.determine_trade_action(symbol, tier1, tier2, tier3)
                    confidence = self.calculate_confidence(tier1, tier2, tier3)
                    price = float(tier1.get("price", 0.0) or 0.0)

                    decision = TradeDecision(
                        symbol=symbol,
                        action=action,
                        confidence=confidence,
                        price=price,
                        reason=reason,
                        tier1_signals=tier1,
                        tier2_signals=tier2,
                        tier3_signals=tier3,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                    decisions.append(decision)

                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Error processing {symbol}: {e}")
                    continue
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error generating trade decisions: {e}")
            return []
        else:
            return decisions

    # -------------------------
    # Caching
    # -------------------------

    async def _cache_trade_decisions(self, decisions: list[TradeDecision]):
        """Cache trade decisions"""
        try:
            self._set_json_with_ttl(
                "trade_decisions",
                [self._to_jsonable(d) for d in decisions],
                int(self.config["cache_ttl"]),
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error caching trade decisions: {e}")

    async def _cache_coin_states(self):
        """Cache coin states"""
        try:
            states_data = {symbol: self._to_jsonable(state) for symbol, state in self.coin_states.items()}
            self._set_json_with_ttl(
                "coin_states",
                states_data,
                int(self.config["cache_ttl"]),
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error caching coin states: {e}")

    # -------------------------
    # Engine loop
    # -------------------------

    async def run(self):
        """Main trade engine loop"""
        logger.info("Starting Unified Trade Engine...")
        self.is_running = True

        try:
            while self.is_running:
                try:
                    # Generate trade decisions
                    decisions = await self.generate_trade_decisions()

                    # Cache decisions and states
                    await self._cache_trade_decisions(decisions)
                    await self._cache_coin_states()

                    # Log significant decisions
                    for decision in decisions:
                        if decision.confidence > 0.8 and decision.action != TradeAction.HOLD:
                            logger.info(f"Strong signal: {decision.symbol} {decision.action.value} (confidence: {decision.confidence:.2f}) - {decision.reason}")

                    logger.debug(f"Generated {len(decisions)} trade decisions")

                    # Wait for next decision cycle (configurable interval)
                    await asyncio.sleep(float(self.config["decision_interval"]))

                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.exception(f"Error in trade engine loop: {e}")
                    # Wait before retry to avoid rapid error loops - using parent constant
                    await asyncio.sleep(10.0)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Fatal error in trade engine: {e}")
        finally:
            self.is_running = False

    def stop(self) -> None:
        """Signal the engine loop to stop after the current cycle."""
        self.is_running = False

    # -------------------------
    # Status & accessors
    # -------------------------

    def get_status(self) -> dict[str, Any]:
        """Get trade engine status"""
        return {
            "status": "running" if self.is_running else "stopped",
            "config": self.config,
            "coin_states_count": len(self.coin_states),
            "thresholds": self.thresholds,
        }

    async def get_coin_state(self, symbol: str) -> CoinState | None:
        """Get current state for a specific coin"""
        return self.coin_states.get(symbol)

    async def get_all_coin_states(self) -> dict[str, CoinState]:
        """Get all coin states"""
        return self.coin_states.copy()

    async def get_trade_decisions(self) -> list[dict[str, Any]]:
        """Get cached trade decisions"""
        try:
            raw = self.redis_client.get("trade_decisions")
            if not raw:
                return []
            text = self._decode_bytes(raw)
            if isinstance(text, list):
                return text  # already parsed
            return json.loads(text)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error getting trade decisions: {e}")
            return []
