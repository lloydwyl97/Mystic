#!/usr/bin/env python3
"""
Mystic Auto-Withdraw System
Automatically withdraws funds to cold wallet when threshold is reached
Supports Binance US only with notifications and logging
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anyio
import httpx
from dotenv import load_dotenv

from backend.config import settings
from backend.config.trading_universe import EXCHANGE_ID
from backend.services.binance_rest_client import BinanceREST
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

# Load environment variables with override to ensure .env takes precedence
load_dotenv(dotenv_path=str(Path(__file__).parent / ".env"), override=True)

# Ensure logs directory exists before configuring logging handlers
Path("logs").mkdir(parents=True, exist_ok=True)


async def make_request(method: str, url: str, **kwargs) -> httpx.Response:
    """Make HTTP request using httpx"""
    async with httpx.AsyncClient() as client:
        return await client.request(method, url, **kwargs)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/auto_withdraw.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("auto_withdraw")


class AutoWithdrawSystem:
    """Unified auto-withdraw system for multiple exchanges"""

    def __init__(self) -> None:
        # All Live Data, No Fallback/Hardcoded Data
        self.exchange = os.getenv("EXCHANGE") or EXCHANGE_ID
        if not self.exchange:
            msg = "EXCHANGE environment variable is required - no fallback/hardcoded exchange"
            raise RuntimeError(msg)
        self.exchange = self.exchange.lower()
        self.cold_wallet_address = os.getenv("COLD_WALLET_ADDRESS")
        self.cold_wallet_threshold = float(os.getenv("COLD_WALLET_THRESHOLD", "250.00"))
        self.check_interval = int(os.getenv("CHECK_INTERVAL", "60"))  # seconds

        # API Keys
        self.binance_api_key = settings.exchange.binance_us_api_key
        self.binance_api_secret = settings.exchange.binance_us_secret_key

        # Notification settings
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK")
        self.telegram_token = os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # Validate configuration
        self._validate_config()

        # Statistics
        self.stats = {
            "total_withdrawals": 0,
            "total_amount_withdrawn": 0.0,
            "last_withdrawal": None,
            "last_check": None,
        }

        logger.info(f"Auto-withdraw system initialized for {self.exchange}")
        logger.info(f"Cold wallet threshold: ${self.cold_wallet_threshold}")
        logger.info(f"Check interval: {self.check_interval} seconds")

    def _validate_config(self) -> None:
        """Validate required configuration"""
        if not self.cold_wallet_address:
            msg = "COLD_WALLET_ADDRESS not configured"
            raise ValueError(msg)

        if self.exchange == EXCHANGE_ID:
            if not self.binance_api_key or not self.binance_api_secret:
                msg = "Binance_US API keys not configured"
                raise ValueError(msg)
        else:
            msg = f"Unsupported exchange: {self.exchange}"
            raise ValueError(msg)

    def _send_notification(self, message: str, _level: str = "INFO") -> None:
        """Send notification via Discord and/or Telegram"""
        try:
            # Discord notification
            if self.discord_webhook:
                discord_payload = {
                    "content": f"ðŸ” **Mystic Auto-Withdraw**\n{message}",
                    "username": "Mystic Trading Bot",
                }
                # anyio.run can run the async make_request directly with args
                try:
                    anyio.run(make_request, "POST", self.discord_webhook, json=discord_payload, timeout=10.0)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.debug(f"Discord notification failed: {e}")

            # Telegram notification
            if self.telegram_token and self.telegram_chat_id:
                telegram_url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                telegram_payload = {
                    "chat_id": self.telegram_chat_id,
                    "text": f"ðŸ” Mystic Auto-Withdraw\n{message}",
                    "parse_mode": "HTML",
                }
                try:
                    anyio.run(make_request, "POST", telegram_url, json=telegram_payload, timeout=10.0)
                except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                    logger.debug(f"Telegram notification failed: {e}")

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to send notification: {e}")

    def _log_withdrawal(
        self,
        amount: float,
        exchange: str,
        status: str,
        details: dict[str, Any],
    ):
        """Log withdrawal to database and file"""
        try:
            # Log to file
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "exchange": exchange,
                "amount": amount,
                "status": status,
                "details": details,
                "cold_wallet_address": (self.cold_wallet_address[:10] + "...") if self.cold_wallet_address else "",
            }

            withdrawals_path = Path("logs/withdrawals.json")
            withdrawals_path.parent.mkdir(parents=True, exist_ok=True)
            with withdrawals_path.open("a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Update statistics
            if status == "success":
                self.stats["total_withdrawals"] += 1
                self.stats["total_amount_withdrawn"] += amount
                self.stats["last_withdrawal"] = datetime.now(timezone.utc).isoformat()

            self.stats["last_check"] = datetime.now(timezone.utc).isoformat()

        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"Failed to log withdrawal: {e}")

    def binance_withdraw(self) -> dict[str, Any]:
        """Handle Binance withdrawal"""
        try:
            # Use centralized REST client for account info

            # Ensure timestamp is available for signed requests if needed
            timestamp = int(time.time() * 1000)

            async def _acct():
                limiter = await BinanceWeightLimiter.create()
                client = BinanceREST(limiter)
                return await client._request(
                    "GET",
                    "/api/v3/account",
                    params={"timestamp": timestamp},
                    signed=True,
                )

            data = anyio.run(_acct) or {}

            # Find USDT balance
            usdt_balance = 0.0
            for balance in data.get("balances", []):
                if balance.get("asset") == "USDT":
                    usdt_balance = float(balance.get("free", 0.0))
                    break

            logger.info(f"[BINANCE] Current USDT balance: ${usdt_balance:.2f}")

            if usdt_balance <= self.cold_wallet_threshold:
                logger.info(f"[BINANCE_US] Balance ${usdt_balance:.2f} below threshold ${self.cold_wallet_threshold}")
                return {"status": "below_threshold", "balance": usdt_balance}

            # Calculate withdrawal amount (leave threshold amount)
            withdrawal_amount = round(usdt_balance - self.cold_wallet_threshold, 2)

            # Execute withdrawal
            timestamp = int(time.time() * 1000)
            params = {
                "asset": "USDT",
                "address": self.cold_wallet_address,
                "amount": withdrawal_amount,
                "network": "ETH",  # Default to ETH network
                "timestamp": timestamp,
            }

            async def _wd():
                limiter = await BinanceWeightLimiter.create()
                client = BinanceREST(limiter)
                # sapi endpoint not pre-mapped; call generic
                return await client._request(
                    "POST",
                    "/sapi/v1/capital/withdraw/apply",
                    params=params,
                    signed=True,
                )

            result = anyio.run(_wd) or {}

            if result.get("id"):
                message = f"âœ… Binance_US withdrawal successful!\nAmount: ${withdrawal_amount:.2f}\nTransaction ID: {result['id']}"
                self._send_notification(message, "SUCCESS")
                self._log_withdrawal(withdrawal_amount, EXCHANGE_ID, "success", result)
                logger.info(f"[BINANCE_US] Withdrawal successful: ${withdrawal_amount:.2f}")
                return {
                    "status": "success",
                    "amount": withdrawal_amount,
                    "tx_id": result["id"],
                }
            message = f"âŒ Binance withdrawal failed: {result}"
            self._send_notification(message, "ERROR")
            self._log_withdrawal(withdrawal_amount, EXCHANGE_ID, "failed", result)
            logger.error(f"[BINANCE_US] Withdrawal failed: {result}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            error_msg = f"Binance withdrawal error: {e!s}"
            self._send_notification(error_msg, "ERROR")
            logger.exception(f"[BINANCE_US] {error_msg}")
            return {"status": "error", "error": str(e)}
        else:
            return {"status": "failed", "error": result}

    def binance_us_withdraw(self) -> dict[str, Any]:
        """Handle Binance US withdrawal"""
        try:
            base_url = "https://api.binance.us"

            def get_headers(_method: str, _request_path: str, _body: str = "") -> dict[str, str]:
                """Generate Binance US API headers"""
                return {
                    "X-MBX-APIKEY": self.binance_api_key,
                    "Content-Type": "application/json",
                }

            # Get account balance
            url = "/accounts"
            headers = get_headers("GET", url)

            async def _acct_get():
                resp = await make_request("GET", base_url + url, headers=headers, timeout=30.0)
                return resp.json()

            accounts = anyio.run(_acct_get) or []

            # Find USDT balance
            usdt_balance = 0.0
            for account in accounts:
                if account.get("currency") == "USDT":
                    usdt_balance = float(account.get("available", 0.0))
                    break

            logger.info(f"[BINANCE_US] Current USDT balance: ${usdt_balance:.2f}")

            if usdt_balance <= self.cold_wallet_threshold:
                logger.info(f"[BINANCE_US] Balance ${usdt_balance:.2f} below threshold ${self.cold_wallet_threshold}")
                return {"status": "below_threshold", "balance": usdt_balance}

            # Calculate withdrawal amount
            withdrawal_amount = round(usdt_balance - self.cold_wallet_threshold, 2)

            # Execute withdrawal
            url = "/withdrawals/crypto"
            body = {
                "amount": str(withdrawal_amount),
                "currency": "USDT",
                "crypto_address": self.cold_wallet_address,
            }
            body_str = json.dumps(body)

            headers = get_headers("POST", url, body_str)

            async def _wd_post():
                resp = await make_request(
                    "POST",
                    base_url + url,
                    headers=headers,
                    content=body_str,
                    timeout=30.0,
                )
                return resp.json()

            result = anyio.run(_wd_post) or {}

            if result.get("id"):
                message = f"âœ… Binance US withdrawal successful!\nAmount: ${withdrawal_amount:.2f}\nTransaction ID: {result['id']}"
                self._send_notification(message, "SUCCESS")
                self._log_withdrawal(withdrawal_amount, EXCHANGE_ID, "success", result)
                logger.info(f"[BINANCE_US] Withdrawal successful: ${withdrawal_amount:.2f}")
                return {
                    "status": "success",
                    "amount": withdrawal_amount,
                    "tx_id": result["id"],
                }
            message = f"âŒ  Binance_US withdrawal failed: {result}"
            self._send_notification(message, "ERROR")
            self._log_withdrawal(withdrawal_amount, EXCHANGE_ID, "failed", result)
            logger.error(f"[BINANCE_US] Withdrawal failed: {result}")
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            error_msg = f"Binance US withdrawal error: {e!s}"
            self._send_notification(error_msg, "ERROR")
            logger.exception(f"[BINANCE_US] {error_msg}")
            return {"status": "error", "error": str(e)}
        else:
            return {"status": "failed", "error": result}

    def get_statistics(self) -> dict[str, Any]:
        """Get withdrawal statistics"""
        return {
            "exchange": self.exchange,
            "cold_wallet_address": (self.cold_wallet_address[:10] + "...") if self.cold_wallet_address else "",
            "threshold": self.cold_wallet_threshold,
            "check_interval": self.check_interval,
            "statistics": self.stats,
        }

    def run_once(self) -> dict[str, Any]:
        """Run withdrawal check once"""
        logger.info(f"Checking {self.exchange} for withdrawal opportunity...")

        if self.exchange == EXCHANGE_ID:
            return self.binance_us_withdraw()
        # Only Binance US supported
        error_msg = f"Unsupported exchange: {self.exchange}"
        logger.error(error_msg)
        return {"status": "error", "error": error_msg}

    def run_continuous(self) -> None:
        """Run continuous withdrawal monitoring"""
        logger.info("Starting continuous auto-withdraw monitoring...")

        while True:
            try:
                result = self.run_once()

                if result.get("status") == "success":
                    logger.info(f"Withdrawal completed: {result}")
                elif result.get("status") == "below_threshold":
                    logger.info(f"Balance below threshold: {result}")
                else:
                    logger.warning(f"Withdrawal check result: {result}")

                # Wait before next check - sync sleep OK for standalone monitoring script
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("Auto-withdraw monitoring stopped by user")
                break
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.exception(f"Unexpected error in continuous monitoring: {e}")
                # Continue with same interval on error
                time.sleep(self.check_interval)


def main():
    """Main entry point"""
    try:
        # Create logs directory if it doesn't exist (safe to call again)
        Path("logs").mkdir(parents=True, exist_ok=True)

        # Initialize auto-withdraw system
        auto_withdraw = AutoWithdrawSystem()

        # Run continuous monitoring
        auto_withdraw.run_continuous()

    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
