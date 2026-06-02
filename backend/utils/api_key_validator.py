"""
API Key Validator Utility
Provides validation and helpful messages for API key configuration
"""

import logging
import os

logger = logging.getLogger(__name__)


def validate_binance_us_keys() -> tuple[bool, list[str]]:
    """
    Validate Binance US API keys configuration
    Returns (is_valid, messages)
    """
    messages = []
    api_key = os.getenv("BINANCE_US_API_KEY", "").strip()
    secret_key = os.getenv("BINANCE_US_SECRET_KEY", "").strip()

    # Check if keys are set
    if not api_key or api_key == "your_binance_us_api_key_here":
        messages.append("WARNING: BINANCE_US_API_KEY not configured")

    if not secret_key or secret_key == "your_binance_us_secret_key_here":
        messages.append("WARNING: BINANCE_US_SECRET_KEY not configured")

    if not api_key or not secret_key or api_key.startswith("your_") or secret_key.startswith("your_"):
        messages.extend(
            [
                "INFO: To enable live trading and OHLCV data:",
                "   1. Create API keys at: https://www.binance.us/en/my/settings/api-management",
                "   2. Update your .env file with real API keys",
                "   3. Restart the backend",
                "NOTE: The backend works in simulation mode without API keys",
            ],
        )
        return False, messages

    # Basic format validation
    if len(api_key) < 20:
        messages.append("WARNING: BINANCE_US_API_KEY appears too short")

    if len(secret_key) < 20:
        messages.append("WARNING: BINANCE_US_SECRET_KEY appears too short")

    # Check for common test/placeholder values
    test_indicators = ["test", "demo", "placeholder", "example", "fake"]
    if any(indicator in api_key.lower() for indicator in test_indicators):
        messages.append("WARNING: BINANCE_US_API_KEY appears to be a test value")

    if any(indicator in secret_key.lower() for indicator in test_indicators):
        messages.append("WARNING: BINANCE_US_SECRET_KEY appears to be a test value")

    is_valid = len(messages) == 0
    if is_valid:
        messages.append("[SUCCESS] Binance US API keys appear to be configured correctly")

    return is_valid, messages


def get_api_status_summary() -> dict[str, any]:
    """Get a summary of API key status for monitoring/health checks"""
    is_valid, messages = validate_binance_us_keys()

    return {
        "binance_us_configured": is_valid,
        "api_key_set": bool(os.getenv("BINANCE_US_API_KEY", "").strip()),
        "secret_key_set": bool(os.getenv("BINANCE_US_SECRET_KEY", "").strip()),
        "validation_messages": messages,
        "simulation_mode": not is_valid,
    }


def log_api_key_status():
    """Log API key status on startup"""
    is_valid, messages = validate_binance_us_keys()

    if is_valid:
        logger.info("Binance US API keys configured - Live trading enabled")
    else:
        logger.warning("Binance US API keys not configured - Running in simulation mode")
        for message in messages:
            logger.info(f"    {message}")
