import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

from backend.modules.ai.persistent_cache import get_persistent_cache
from backend.services.ai_service import get_ai_service
from backend.services.binance_rest_client import BinanceREST
from backend.services.market_data import MarketDataService
from backend.services.portfolio_service import PortfolioService
from backend.services.websocket_manager import WebSocketManager
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

logger = logging.getLogger(__name__)


class RealTimeProcessor:
    def __init__(self) -> None:
        self.running = False
        self.market_data_service = None
        self.ai_service = None
        self.portfolio_service = None
        logger.info("RealTimeProcessor initialized with live data connections")

    async def start(self):
        logger.info("RealTimeProcessor starting with live data services")
        try:
            # Initialize live data services - Use active service not deprecated
            self.market_data_service = MarketDataService.shared()
            self.ai_service = get_ai_service()
            self.portfolio_service = PortfolioService.shared()

            self.running = True
            logger.info("RealTimeProcessor started successfully")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to start RealTimeProcessor: {e}")
            self.running = False

    async def stop(self):
        logger.info("RealTimeProcessor stopping")
        self.running = False
        self.market_data_service = None
        self.ai_service = None
        self.portfolio_service = None

    async def process_market_data(self):
        """Process live market data"""
        if not self.running or not self.market_data_service:
            return

        try:
            # Get live market data
            symbols = await self.market_data_service.get_active_symbols()
            for symbol in symbols:
                coin_data = await self.market_data_service.get_coin_data(symbol)
                if coin_data:
                    await self.store_market_data(
                        {
                            "symbol": symbol,
                            "price": coin_data.current_price,
                            "volume": coin_data.volume_24h,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing market data: {e}")

    async def process_trade_signals(self):
        """Process live trade signals"""
        if not self.running or not self.ai_service:
            return

        try:
            # Get live AI signals
            predictions = await self.ai_service.get_predictions()
            if predictions and "predictions" in predictions:
                await self.store_trade_signals(predictions["predictions"])
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing trade signals: {e}")

    async def process_portfolio_updates(self):
        """Process live portfolio updates"""
        if not self.running or not self.portfolio_service:
            return

        try:
            # Get live portfolio data
            portfolio_summary = await self.portfolio_service.get_portfolio_summary()
            if portfolio_summary:
                await self.store_portfolio_data(portfolio_summary)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing portfolio updates: {e}")

    async def process_risk_alerts(self):
        """Process live risk alerts"""
        if not self.running:
            return

        try:
            # Calculate risk metrics
            risk_metrics = await self.calculate_risk_metrics()
            if risk_metrics:
                alerts = await self.check_risk_alerts(risk_metrics)
                if alerts:
                    await self.store_risk_alerts(alerts)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error processing risk alerts: {e}")

    async def fetch_live_market_data(self) -> dict[str, Any]:
        """Fetch live market data"""
        if not self.market_data_service:
            return {}

        try:
            symbols = await self.market_data_service.get_active_symbols()
            market_data: dict[str, Any] = {}
            for symbol in symbols:
                coin_data = await self.market_data_service.get_coin_data(symbol)
                if coin_data:
                    market_data[symbol] = {
                        "price": coin_data.current_price,
                        "volume": coin_data.volume_24h,
                        "change_24h": coin_data.change_24h,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error fetching live market data: {e}")
            return {}
        else:
            return market_data

    async def fetch_binance_data(self) -> dict[str, Any]:
        """Fetch live Binance data"""
        try:
            limiter = await BinanceWeightLimiter.create()
            client = BinanceREST(limiter)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error initializing Binance client: {e}")
            return {}

        # Get live ticker data - use first symbol from trading_universe (live data)
        if not TRADING_SYMBOLS:
            msg = "No trading symbols available - TRADING_SYMBOLS must be configured"
            raise RuntimeError(msg)
        try:
            symbol = TRADING_SYMBOLS[0]
            return await client.get_24h_ticker(symbol)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error fetching Binance data: {e}")
            return {}

    async def generate_trade_signals(self) -> list[dict[str, Any]]:
        """Generate live trade signals"""
        if not self.ai_service:
            return []

        try:
            predictions = await self.ai_service.get_predictions()
            result = predictions.get("predictions", []) if isinstance(predictions, dict) else []
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error generating trade signals: {e}")
            return []
        else:
            return result

    async def calculate_portfolio_metrics(self) -> dict[str, Any]:
        """Calculate live portfolio metrics"""
        if not self.portfolio_service:
            return {}

        try:
            return await self.portfolio_service.get_portfolio_summary()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error calculating portfolio metrics: {e}")
            return {}

    async def calculate_risk_metrics(self) -> dict[str, Any]:
        """Calculate live risk metrics"""
        try:
            # Basic risk calculation based on portfolio
            portfolio_metrics = await self.calculate_portfolio_metrics()
            result = (
                {
                    "total_value": portfolio_metrics.get("total_value", 0),
                    "risk_level": "medium",  # Could be calculated based on volatility
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if portfolio_metrics
                else {}
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error calculating risk metrics: {e}")
            return {}
        else:
            return result

    async def check_risk_alerts(self, risk_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Check for risk alerts based on live data"""
        alerts: list[dict[str, Any]] = []
        try:
            # Simple risk alert logic
            total_value = risk_data.get("total_value", 0)
            if total_value < 1000:  # Low portfolio value alert
                alerts.append(
                    {
                        "type": "low_portfolio",
                        "message": "Portfolio value is below threshold",
                        "value": total_value,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error checking risk alerts: {e}")
            return []
        else:
            return alerts

    async def store_market_data(self, data: dict[str, Any]):
        """Store live market data"""
        try:
            # Store in cache or database
            cache = get_persistent_cache()
            cache.store_market_data(data)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error storing market data: {e}")

    async def store_trade_signals(self, signals: list[dict[str, Any]]):
        """Store live trade signals"""
        try:
            cache = get_persistent_cache()
            for signal in signals:
                cache.store_signal(signal)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error storing trade signals: {e}")

    async def store_portfolio_data(self, data: dict[str, Any]):
        """Store live portfolio data"""
        try:
            cache = get_persistent_cache()
            cache.store_portfolio_data(data)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error storing portfolio data: {e}")

    async def store_risk_alerts(self, alerts: list[dict[str, Any]]):
        """Store live risk alerts"""
        try:
            cache = get_persistent_cache()
            for alert in alerts:
                cache.store_alert(alert)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error storing risk alerts: {e}")

    async def publish_market_updates(self, data: dict[str, Any]):
        """Publish live market updates"""
        try:
            # Publish to WebSocket or message queue
            ws_manager = WebSocketManager()
            await ws_manager.broadcast_market_data(data)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error publishing market updates: {e}")

    async def publish_trade_signals(self, signals: list[dict[str, Any]]):
        """Publish live trade signals"""
        try:
            ws_manager = WebSocketManager()
            await ws_manager.broadcast_trading_signals(signals)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error publishing trade signals: {e}")

    async def publish_portfolio_updates(self, data: dict[str, Any]):
        """Publish live portfolio updates"""
        try:
            ws_manager = WebSocketManager()
            await ws_manager.broadcast_portfolio_updates(data)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error publishing portfolio updates: {e}")

    async def publish_risk_alerts(self, alerts: list[dict[str, Any]]):
        """Publish live risk alerts"""
        try:
            ws_manager = WebSocketManager()
            await ws_manager.broadcast_risk_alerts(alerts)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error publishing risk alerts: {e}")


processor = RealTimeProcessor()


async def main():
    logger.info("RealTimeProcessor main() starting")
    await processor.start()

    # Run processing loops
    while processor.running:
        try:
            await processor.process_market_data()
            await processor.process_trade_signals()
            await processor.process_portfolio_updates()
            await processor.process_risk_alerts()
            await asyncio.sleep(5)  # Process every 5 seconds
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error in main processing loop: {e}")
            await asyncio.sleep(10)  # Wait longer on error


if __name__ == "__main__":
    asyncio.run(main())
