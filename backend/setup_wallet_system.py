#!/usr/bin/env python3
"""
Setup script for Mystic AI Wallet System
Initializes database tables and sample data for the real-time wallet panel
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_database() -> None:
    """Initialize database with required tables"""
    db_path = os.getenv("SIMULATION_TRADES_DB", "simulation_trades.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
    CREATE TABLE IF NOT EXISTS simulated_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        action TEXT NOT NULL,
        price REAL NOT NULL,
        confidence REAL,
        simulated_profit REAL,
        strategy TEXT,
        mystic_signals TEXT,
        wallet_source TEXT DEFAULT 'Main AI Trading'
    )
    """
        )
        cursor.execute(
            """
    CREATE TABLE IF NOT EXISTS wallet_allocations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wallet_name TEXT NOT NULL,
        allocation_percent REAL NOT NULL,
        current_balance REAL DEFAULT 0,
        last_updated TEXT,
        status TEXT DEFAULT 'active'
    )
    """
        )
        cursor.execute(
            """
    CREATE TABLE IF NOT EXISTS yield_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        provider TEXT NOT NULL,
        protocol TEXT NOT NULL,
        amount_deployed REAL NOT NULL,
        apy REAL NOT NULL,
        start_date TEXT,
        status TEXT DEFAULT 'active'
    )
    """
        )
        cursor.execute(
            """
    CREATE TABLE IF NOT EXISTS cold_wallet_syncs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL NOT NULL,
        timestamp TEXT NOT NULL,
        threshold_triggered REAL,
        status TEXT DEFAULT 'completed'
    )
    """
        )
        conn.commit()
    logger.info("[OK] Database tables created successfully")


def insert_live_data() -> None:
    """Initialize live data connections and empty tables for real-time data"""
    db_path = os.getenv("SIMULATION_TRADES_DB", "simulation_trades.db")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM wallet_allocations")
        cursor.execute("DELETE FROM yield_positions")
        cursor.execute("DELETE FROM cold_wallet_syncs")
        cursor.execute("DELETE FROM simulated_trades")
        # Initialize with empty live data structure - no sample data
        live_wallets = [
            ("Main AI Trading", 0.0, 0.0),  # Empty - will be populated by live system
            ("Backup AI Trading", 0.0, 0.0),  # Empty - will be populated by live system
            ("Cold Storage Vault", 0.0, 0.0),  # Empty - will be populated by live system
        ]
        cursor.executemany(
            """
    INSERT OR REPLACE INTO wallet_allocations (wallet_name, allocation_percent, current_balance, last_updated, status)
    VALUES (?, ?, ?, ?, 'waiting_for_live_data')
    """,
            [(w[0], w[1], w[2], _now_iso()) for w in live_wallets],
        )
        cursor.execute(
            """
    INSERT OR REPLACE INTO yield_positions (provider, protocol, amount_deployed, apy, start_date, status)
    VALUES ('Initializing', 'Live Data', 0.0, 0.0, ?, 'waiting_for_live_data')
    """,
            (_now_iso(),),
        )
        cursor.execute(
            """
    INSERT OR REPLACE INTO cold_wallet_syncs (amount, timestamp, threshold_triggered, status)
    VALUES (0.0, ?, 0.0, 'waiting_for_live_data')
    """,
            (_now_iso(),),
        )
        conn.commit()
    logger.info("[OK] Live data tables initialized - waiting for real-time data connections")
    logger.info("     Note: Data will be populated by live trading system and API connections")


def create_ai_model_state() -> None:
    """Create initial AI model state file"""
    model_state = {
        "version": 1,
        "mode": "training",
        "confidence_threshold": 0.75,
        "avg_profit_threshold": 0.5,
        "adjustment_count": 0,
        "last_update": _now_iso(),
        "performance_metrics": {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "total_profit": 0.0,
        },
    }
    model_state_path = Path("ai_model_state.json")
    with model_state_path.open("w", encoding="utf-8") as f:
        json.dump(model_state, f, indent=2)
    logger.info("[OK] AI model state file created")


def create_config_files() -> None:
    """Create configuration files for the system"""
    env_template = """
# Mystic AI Wallet System Configuration
SIM_DB_PATH=simulation_trades.db
MODEL_STATE_PATH=ai_model_state.json

# Discord/Telegram Notifications (optional)
DISCORD_WEBHOOK=your_discord_webhook_url
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Cold Wallet Configuration
COLD_WALLET_THRESHOLD=250.00
COLD_WALLET_ADDRESS=your_cold_wallet_address

# Yield Rotation Settings
YIELD_ROTATION_THRESHOLD=0.005
MAX_YIELD_ALLOCATION=0.4
    """.strip()
    env_example_path = Path("env.example")
    with env_example_path.open("w", encoding="utf-8") as f:
        f.write(env_template)
    logger.info("[OK] Configuration files created")


def main() -> bool:
    """Main setup function"""
    logger.info("[INIT] Setting up Mystic AI Wallet System...")
    try:
        setup_database()
        insert_live_data()
        create_ai_model_state()
        create_config_files()
        logger.info("\n[DONE] Setup completed successfully")
        logger.info("\n[NEXT] Steps:")
        logger.info("1. Copy env.example to .env and configure your settings")
        logger.info("2. Start the backend: uvicorn main:app --host 127.0.0.1 --port 8000")
        logger.info("3. Start the UI: npm start")
        logger.info("4. Dashboard: http://127.0.0.1:3501")
        logger.info("5. API docs: http://127.0.0.1:8000/docs")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"[FAIL] Setup failed: {e!s}")
        return False
    return True


if __name__ == "__main__":
    main()
