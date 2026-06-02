import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RISK_FILE = "./config/risk.json"
INTERVAL = 900
PING_FILE = "./logs/risk_optimizer.ping"
DB_PATH = os.getenv("SIM_DB_PATH", "simulation_trades.db")
LOOKBACK_TRADES = int(os.getenv("RISK_LOOKBACK_TRADES", "500"))

Path("./config").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)


def create_ping_file(risk_level: str, winrate: float) -> None:
    try:
        ping_path = Path(PING_FILE)
        ping_path.parent.mkdir(parents=True, exist_ok=True)
        with ping_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "online",
                    "last_update": datetime.now(timezone.utc).isoformat(),
                    "risk_level": risk_level,
                    "winrate": winrate,
                },
                f,
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"ping file error: {e}")


def load_risk_config() -> dict:
    try:
        risk_path = Path(RISK_FILE)
        if risk_path.exists():
            with risk_path.open(encoding="utf-8") as f:
                return json.load(f)
        cfg = {
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "max_position_size": 0.1,
            "max_daily_loss": 0.05,
            "recent_winrate": 0.6,
            "total_trades": 0,
            "profitable_trades": 0,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        save_risk_config(cfg)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"load risk config error: {e}")
        return {}
    else:
        return cfg


def save_risk_config(config: dict) -> None:
    try:
        config["last_updated"] = datetime.now(timezone.utc).isoformat()
        risk_path = Path(RISK_FILE)
        risk_path.parent.mkdir(parents=True, exist_ok=True)
        with risk_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"save risk config error: {e}")


def fetch_trades_from_db(limit: int) -> list[float]:
    if not Path(DB_PATH).exists():
        return []
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT simulated_profit
                FROM simulated_trades
                WHERE simulated_profit IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [float(r[0]) for r in rows if r and r[0] is not None]
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"db fetch error: {e}")
        return []


def calculate_winrate_live() -> tuple[float, int, int]:
    profits = fetch_trades_from_db(LOOKBACK_TRADES)
    if not profits:
        return 0.6, 0, 0
    total = len(profits)
    profitable = sum(1 for p in profits if p > 0)
    winrate = profitable / total if total else 0.0
    return winrate, total, profitable


def adjust_risk() -> None:
    try:
        cfg = load_risk_config()
        winrate, total, profitable = calculate_winrate_live()
        if total > 0:
            cfg["total_trades"] = total
            cfg["profitable_trades"] = profitable
        cfg["recent_winrate"] = winrate

        if winrate < 0.40:
            cfg["stop_loss_pct"] = 0.01
            risk_level = "conservative"
        elif winrate < 0.50:
            cfg["stop_loss_pct"] = 0.015
            risk_level = "moderate"
        elif winrate < 0.60:
            cfg["stop_loss_pct"] = 0.02
            risk_level = "standard"
        else:
            cfg["stop_loss_pct"] = 0.03
            risk_level = "aggressive"

        if winrate < 0.50:
            cfg["max_position_size"] = 0.05
        elif winrate > 0.70:
            cfg["max_position_size"] = 0.15
        else:
            cfg["max_position_size"] = 0.10

        save_risk_config(cfg)

        logger.info(f"[RISK] winrate={winrate:.3f} stop_loss={cfg['stop_loss_pct']:.3f} pos_size={cfg['max_position_size']:.3f} level={risk_level}")
        create_ping_file(risk_level, winrate)

        risk_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "winrate": winrate,
            "stop_loss_pct": cfg["stop_loss_pct"],
            "max_position_size": cfg["max_position_size"],
            "risk_level": risk_level,
            "lookback_trades": total,
        }
        risk_log_path = Path("./logs/risk_log.jsonl")
        risk_log_path.parent.mkdir(parents=True, exist_ok=True)
        with risk_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(risk_log) + "\n")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"risk adjust error: {e}")


def main() -> None:
    logger.info("[RISK] Risk Optimizer started")
    logger.info(f"[RISK] Optimization interval: {INTERVAL} seconds")
    while True:
        try:
            adjust_risk()
            time.sleep(INTERVAL)
        except KeyboardInterrupt:
            logger.info("[RISK] Shutting down...")
            break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            logger.exception(f"main loop error: {e}")


if __name__ == "__main__":
    main()
