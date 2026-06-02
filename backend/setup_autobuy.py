#!/usr/bin/env python3
"""
Binance US Autobuy Setup Script
Setup and configuration for autobuy system using trading_universe symbols
"""

import json
import logging
import sys
from pathlib import Path

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutobuySetup:
    """Setup and configuration for the autobuy system"""

    def __init__(self) -> None:
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.config_file = self.project_root / "autobuy_config.json"
        self.requirements_file = self.project_root / "requirements_autobuy.txt"

    def create_directories(self):
        directories = ["logs", "reports", "data", "backups"]
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        logger.info("[OK] Created directories")

    def create_env_template(self):
        env_template = """# Binance US Autobuy System Configuration
# ================================================

# Binance US API Configuration
BINANCE_US_API_KEY=your_binance_us_api_key_here
BINANCE_US_SECRET_KEY=your_binance_us_secret_key_here

# Trading Configuration
TRADING_ENABLED=true
BINANCE_TESTNET=false
USD_AMOUNT_PER_TRADE=50
MAX_CONCURRENT_TRADES=4

# Signal Configuration
MIN_VOLUME_INCREASE=1.5
MIN_PRICE_CHANGE=0.02
SIGNAL_COOLDOWN=300

# Risk Management
MAX_DAILY_TRADES=48
MAX_DAILY_VOLUME=2000.0
STOP_LOSS_PERCENTAGE=0.05
TAKE_PROFIT_PERCENTAGE=0.10

# Notification Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here

# Logging Configuration
LOG_LEVEL=INFO

# Performance Configuration
CYCLE_INTERVAL=30
DATA_CACHE_TTL=60

# Advanced Configuration
ENABLE_TECHNICAL_ANALYSIS=true
ENABLE_SENTIMENT_ANALYSIS=false
ENABLE_WHALE_TRACKING=true

# Market Hours (timezone.utc)
TRADING_START_HOUR=0
TRADING_END_HOUR=24

# Emergency Configuration
EMERGENCY_STOP=false
MAX_LOSS_PER_TRADE=10.0
"""
        if not self.env_file.exists():
            with self.env_file.open("w", encoding="utf-8") as f:
                f.write(env_template)
            logger.info("[OK] Created .env template")
        else:
            logger.info("[INFO] .env already exists")

    def create_config_file(self):
        # Use TRADING_SYMBOLS from trading_universe (live data)
        # Generate trading pairs config dynamically from trading_universe
        trading_pairs = {}
        # Default configuration values for all symbols
        default_configs = {
            "BTCUSDT": {"min_trade_amount": 50.0, "max_trade_amount": 500.0, "target_frequency": 30},
            "ETHUSDT": {"min_trade_amount": 50.0, "max_trade_amount": 400.0, "target_frequency": 20},
            "ADAUSDT": {"min_trade_amount": 15.0, "max_trade_amount": 150.0, "target_frequency": 15},
            "SOLUSDT": {"min_trade_amount": 25.0, "max_trade_amount": 200.0, "target_frequency": 15},
            "DOGEUSDT": {"min_trade_amount": 10.0, "max_trade_amount": 100.0, "target_frequency": 10},
            "XRPUSDT": {"min_trade_amount": 20.0, "max_trade_amount": 200.0, "target_frequency": 15},
            "BCHUSDT": {"min_trade_amount": 25.0, "max_trade_amount": 250.0, "target_frequency": 20},
            "LTCUSDT": {"min_trade_amount": 25.0, "max_trade_amount": 250.0, "target_frequency": 20},
            "AVAXUSDT": {"min_trade_amount": 25.0, "max_trade_amount": 200.0, "target_frequency": 15},
            "LINKUSDT": {"min_trade_amount": 25.0, "max_trade_amount": 200.0, "target_frequency": 15},
        }
        # Coin name mapping (for display purposes)
        coin_names = {
            "BTCUSDT": "Bitcoin",
            "ETHUSDT": "Ethereum",
            "ADAUSDT": "Cardano",
            "SOLUSDT": "Solana",
            "DOGEUSDT": "Dogecoin",
            "XRPUSDT": "XRP",
            "BCHUSDT": "Bitcoin Cash",
            "LTCUSDT": "Litecoin",
            "AVAXUSDT": "Avalanche",
            "LINKUSDT": "Chainlink",
        }
        # Generate config for each symbol in TRADING_SYMBOLS
        for symbol in TRADING_SYMBOLS:
            default_cfg = default_configs.get(symbol, {"min_trade_amount": 25.0, "max_trade_amount": 200.0, "target_frequency": 15})
            trading_pairs[symbol] = {
                "name": coin_names.get(symbol, symbol.replace("USDT", "")),
                "min_trade_amount": default_cfg["min_trade_amount"],
                "max_trade_amount": default_cfg["max_trade_amount"],
                "target_frequency": default_cfg["target_frequency"],
                "enabled": True,
            }
        config = {
            "trading_pairs": trading_pairs,
            "signal_config": {
                "min_confidence": 50.0,
                "min_volume_increase": 1.5,
                "min_price_change": 0.02,
                "max_price_change": 0.15,
                "volume_threshold": 1000000,
                "volatility_threshold": 0.05,
                "momentum_threshold": 0.03,
            },
            "risk_config": {
                "max_concurrent_trades": 4,
                "max_daily_trades": 48,
                "max_daily_volume": 2000.0,
                "stop_loss_percentage": 0.05,
                "take_profit_percentage": 0.10,
                "max_drawdown": 0.20,
            },
            "trading_hours": {
                "start_hour": 0,
                "end_hour": 24,
                "timezone": "timezone.utc",
            },
            "emergency_stop": False,
            "cycle_interval": 30,
            "signal_cooldown": 300,
        }
        with self.config_file.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("[OK] Created autobuy_config.json")

    def create_requirements_file(self):
        requirements = """# Binance US Autobuy System Requirements
# ================================================

httpx>=0.25.0
requests>=2.28.0
python-dotenv>=0.19.0
fastapi>=0.68.0
uvicorn>=0.15.0
websockets>=10.0
pandas>=1.5.0
numpy>=1.21.0
structlog>=21.5.0
"""
        with self.requirements_file.open("w", encoding="utf-8") as f:
            f.write(requirements)
        logger.info("[OK] Created requirements_autobuy.txt")

    def validate_setup(self) -> bool:
        required_files = [self.env_file, self.config_file, self.requirements_file]
        required_dirs = [self.project_root / d for d in ("logs", "reports", "data", "backups")]
        missing_files = [p.name for p in required_files if not p.exists()]
        missing_dirs = [p.name for p in required_dirs if not p.exists()]
        if missing_files or missing_dirs:
            logger.error("[FAIL] Setup validation failed")
            if missing_files:
                logger.error("Missing files: " + ", ".join(missing_files))
            if missing_dirs:
                logger.error("Missing directories: " + ", ".join(missing_dirs))
            return False
        logger.info("[OK] Setup validation passed")
        return True

    def run_setup(self):
        logger.info("[INIT] Binance US Autobuy Setup")
        try:
            self.create_directories()
            self.create_env_template()
            self.create_config_file()
            self.create_requirements_file()
            if self.validate_setup():
                logger.info("[DONE] Setup completed")
            else:
                return False
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"[FAIL] Setup error: {e}")
            return False
        return True


def main():
    setup = AutobuySetup()
    ok = setup.run_setup()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
