#!/usr/bin/env python3
"""
Trading Bots Startup Script
Runs Binance bots with proper error handling
"""

import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Any

# Import the bot manager
from bot_manager import BotManager

from backend.services.task_manager import task_manager

# Import rotated logging system
from backend.utils.log_rotation_manager import get_log_rotation_manager

# Configure logging with rotation
log_manager = get_log_rotation_manager()
logger = log_manager.setup_logger("trading_bots", "trading_bots.log")


class TradingBotsRunner:
    def __init__(self) -> None:
        self.bot_manager = BotManager()
        self.is_running = False

        # Set up signal handlers for graceful shutdown in the main thread
        try:
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            # In some environments (e.g., non-main threads or restricted contexts),
            # setting signal handlers may fail. We ignore failures here to keep
            # initialization robust.
            logger.debug("Could not register signal handlers in this environment")

        logger.info("Trading Bots Runner initialized")

    def signal_handler(self, signum: int, _frame: Any) -> None:
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False

        # If an asyncio loop is running, schedule cleanup there; otherwise run cleanup now.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Schedule the async cleanup in the running loop thread-safely
            try:
                loop.call_soon_threadsafe(lambda: task_manager.create_task_sync(self.cleanup(), name="run_bots:cleanup"))
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.debug(f"Failed to schedule cleanup on running loop: {e}")
        else:
            # No running loop — run cleanup synchronously to ensure resources are freed
            try:
                asyncio.run(self.cleanup())
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error running cleanup synchronously: {e}")

    async def run(self):
        """Main runner function"""
        logger.info("Starting Trading Bots Runner...")
        self.is_running = True

        try:
            # Start the bot manager
            await self.bot_manager.run()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Fatal error in Trading Bots Runner: {e}")
        finally:
            # Ensure cleanup is performed
            try:
                await self.cleanup()
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Error during final cleanup: {e}")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        self.is_running = False

        try:
            await self.bot_manager.stop_all_bots()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Error during cleanup: {e}")

        logger.info("Trading Bots Runner stopped")


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("MYSTIC TRADING BOTS")
    logger.info("=" * 60)
    logger.info(f"Starting at: {datetime.now(timezone.utc).isoformat()}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    try:
        runner = TradingBotsRunner()
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
