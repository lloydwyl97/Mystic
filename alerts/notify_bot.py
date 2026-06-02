#!/usr/bin/env python3
"""
Notification Bot - Discord & Telegram Alerts
Runs in a dedicated container for trading alerts
"""

import asyncio
import logging
import os

try:
    import httpx
except ImportError:
    httpx = None

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

# Add the current directory to Python path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DISCORD_SUCCESS_CODE = 204
TELEGRAM_SUCCESS_CODE = 200


class NotificationBot:
    """Main notification bot for trading alerts"""

    def __init__(self):
        self.running = False
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    async def initialize(self):
        """Initialize the notification bot"""
        logger.info("Initializing Notification Bot...")

        # Check configuration
        if not self.discord_webhook and not self.telegram_token:
            logger.warning("No notification channels configured")
        else:
            if self.discord_webhook:
                logger.info("Discord webhook configured")
            if self.telegram_token:
                logger.info("Telegram bot configured")

        logger.info("Notification Bot initialized")

    async def send_discord_alert(self, message: str):
        """Send alert to Discord"""
        if not self.discord_webhook:
            return

        if httpx is None:
            logger.error("httpx is not installed, cannot send Discord alerts")
            return

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                payload = {"content": message}
                r = await client.post(self.discord_webhook, json=payload)
                if r.status_code == DISCORD_SUCCESS_CODE:
                    logger.info("Discord alert sent")
                else:
                    logger.error(f"Discord alert failed: {r.status_code} - {r.text}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Discord alert error")

    async def send_telegram_alert(self, message: str):
        """Send alert to Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            return

        if httpx is None:
            logger.error("httpx is not installed, cannot send Telegram alerts")
            return

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(url, json=payload)
                if r.status_code == TELEGRAM_SUCCESS_CODE:
                    logger.info("Telegram alert sent")
                else:
                    logger.error(f"Telegram alert failed: {r.status_code} - {r.text}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Telegram alert error")

    async def send_alert(self, message: str):
        """Send alert to all configured channels"""
        await self.send_discord_alert(message)
        await self.send_telegram_alert(message)

    async def start(self):
        """Start the notification bot"""
        logger.info("Starting Notification Bot...")
        self.running = True

        await self.initialize()

        # Send startup notification
        await self.send_alert("Mystic Trading Notification Bot is online!")

        # Main loop - listen for Redis messages
        r = None
        if redis is None:
            logger.error("redis is not installed, cannot start notification bot")
            return

        try:
            redis_url = os.getenv("REDIS_URL")
            if not redis_url:
                logger.error("REDIS_URL not set; cannot start notification bot")
                return
            r = redis.from_url(redis_url, protocol=2)

            while self.running:
                try:
                    # Listen for notification messages
                    message = await r.blpop("notifications", timeout=1)
                    if message:
                        # message is expected to be a sequence like [key, value]
                        try:
                            _, alert_data = message
                        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                            # Fallback if a tuple/list with different shape
                            alert_data = message[1] if len(message) > 1 else message[0]

                        text = alert_data.decode("utf-8", errors="replace") if isinstance(alert_data, bytes) else str(alert_data)

                        await self.send_alert(text)

                except asyncio.CancelledError:
                    # Propagate cancellation to allow clean shutdown
                    raise
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    logger.exception("Error in notification loop")
                    await asyncio.sleep(5)

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Fatal error in notification bot")
            self.running = False
        finally:
            if r is not None:
                try:
                    # Close redis connection cleanly if supported
                    await r.close()
                    if hasattr(r, "connection_pool") and hasattr(r.connection_pool, "disconnect"):
                        r.connection_pool.disconnect()
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                    # ignore errors during cleanup
                    pass

    async def stop(self):
        """Stop the notification bot"""
        logger.info("Stopping Notification Bot...")
        self.running = False
        try:
            await self.send_alert("Mystic Trading Notification Bot is shutting down...")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            logger.exception("Error while sending shutdown alert")


async def main():
    """Main entry point"""
    bot = NotificationBot()

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        await bot.stop()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        logger.exception("Unexpected error")
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
