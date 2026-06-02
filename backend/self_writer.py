import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from ai_mutation_feedback import fetch_recent_strategy_stats

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import TRADING_SYMBOLS from trading_universe: {e}"
    raise RuntimeError(msg) from e

logger = logging.getLogger(__name__)

# All Live Data, No Fallback/Hardcoded Data
ALLOWED_SYMBOLS = list(TRADING_SYMBOLS)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_float(val, default: float = 0.0) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


def _maybe_float(val):
    try:
        if val is None:
            return None
        return float(val)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


def _maybe_int(val):
    try:
        if val is None:
            return None
        # Support values like "10.0" by converting via float first
        return int(float(val))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return None


class SelfWriter:
    """Autonomous parameter generator"""

    def __init__(self) -> None:
        self.generated_modules: list[str] = []
        self.mutation_count = 0
        Path("generated_modules").mkdir(parents=True, exist_ok=True)

    def generate_module_blueprint(self, strategy_stats: list[dict]) -> tuple[str | None, str | None]:
        if not strategy_stats:
            return None, None
        top = max(strategy_stats, key=lambda s: _to_float(s.get("total_profit", 0.0)))
        base = _safe_name(str(top.get("strategy", "UnknownStrategy")))
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        evolved_name = f"{base}_params_{ts}"
        params = self._derive_parameters(top)
        payload = {
            "name": evolved_name,
            "version": 1,
            "generated_at": _now_iso(),
            "source_strategy": base,
            "source_snapshot": {
                "win_rate": _to_float(top.get("win_rate", 0.0)),
                "total_profit": _to_float(top.get("total_profit", 0.0)),
                "avg_profit": _maybe_float(top.get("avg_profit")) if "avg_profit" in top else None,
                "max_drawdown": _maybe_float(top.get("max_drawdown")) if "max_drawdown" in top else None,
                "trades": _maybe_int(top.get("trades")) if "trades" in top else None,
            },
            "pairs": ALLOWED_SYMBOLS,
            "parameters": params,
        }
        return evolved_name + ".json", json.dumps(payload, indent=2)

    def _derive_parameters(self, s: dict) -> dict:
        win_rate = _to_float(s.get("win_rate", 0.5))
        total_profit = _to_float(s.get("total_profit", 0.0))
        max_dd = _to_float(s.get("max_drawdown", 0.0))
        trades = _to_float(s.get("trades", 0.0))
        profit_score = math.tanh(total_profit / 1000.0)
        dd_penalty = _clamp(1.0 - max(0.0, max_dd) / 0.3, 0.0, 1.0)
        base_aggr = _clamp(0.5 * win_rate + 0.4 * profit_score + 0.1 * dd_penalty, 0.0, 1.0)
        pos_size = _clamp(0.01 + 0.04 * base_aggr, 0.01, 0.05)
        stop_loss = _clamp(0.08 - 0.06 * base_aggr, 0.02, 0.08)
        take_profit = _clamp(0.06 + 0.14 * base_aggr, 0.06, 0.20)
        cooldown = int(_clamp(600.0 - 4.0 * trades, 60.0, 600.0))
        min_conf = _clamp(0.55 - 0.1 * (base_aggr - 0.5), 0.45, 0.65)
        slip_limit = _clamp(0.015 - 0.01 * base_aggr, 0.002, 0.015)
        vol_floor_usd = 500000.0 - 300000.0 * base_aggr
        return {
            "risk": {
                "position_size": round(pos_size, 4),
                "max_drawdown_limit": round(max(0.05, min(0.25, max_dd or 0.1)), 4),
                "stop_loss": round(stop_loss, 4),
                "take_profit": round(take_profit, 4),
            },
            "execution": {
                "signal_cooldown_sec": cooldown,
                "min_confidence": round(min_conf, 3),
                "max_slippage": round(slip_limit, 4),
                "min_quote_volume_24h": max(100000, int(vol_floor_usd)),
            },
            "filters": {
                "enable_rsi_filter": True,
                "enable_macd_filter": True,
                "enable_orderflow_filter": True,
            },
            "weights": {
                "momentum": round(0.4 + 0.3 * base_aggr, 3),
                "mean_reversion": round(0.4 - 0.2 * base_aggr, 3),
                "orderflow": round(0.2, 3),
            },
        }

    def save_module(self, file_name: str, code: str) -> str:
        full_path = Path("generated_modules") / file_name
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with full_path.open("w", encoding="utf-8") as f:
            f.write(code)
        self.generated_modules.append(full_path)
        logger.info(f"[SELF-WRITER] New parameters saved -> {full_path}")
        return full_path

    def auto_write_loop(self) -> str | None:
        logger.info("[SELF-WRITER] Starting autonomous parameter generation")
        stats = fetch_recent_strategy_stats(hours_back=12)
        if not stats:
            logger.info("[SELF-WRITER] No strategy data available")
            return None
        file_name, code = self.generate_module_blueprint(stats)
        if file_name and code:
            path = self.save_module(file_name, code)
            self.mutation_count += 1
            logger.info(f"[SELF-WRITER] Generated set {self.mutation_count}: {file_name}")
            return path
        return None

    def get_generation_stats(self) -> dict:
        return {
            "total_modules": len(self.generated_modules),
            "mutation_count": self.mutation_count,
            "latest_modules": self.generated_modules[-5:] if self.generated_modules else [],
        }


def auto_write_loop() -> str | None:
    writer = SelfWriter()
    return writer.auto_write_loop()


if __name__ == "__main__":
    path = auto_write_loop()
    if path:
        logger.info(f"[SELF-WRITER] Successfully generated: {path}")
    else:
        logger.info("[SELF-WRITER] No parameters generated")
