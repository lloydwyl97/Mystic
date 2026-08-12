"""Scalp configuration — separate from Mystic DAY / portfolio_engine."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=False)


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ScalpConfig:
    repo_root: str
    database_path: str
    redis_url: str
    redis_key_prefix: str
    products: tuple[str, ...]
    scalp_paper_enabled: bool
    scalp_paper_auto_arm: bool
    scalp_live: bool
    fee_model_verified: bool
    max_open_positions: int
    max_notional_paper: float
    allow_repair_add: bool
    allow_market_orders: bool
    strategy_id: str
    exchange: str
    calibration_mode: bool
    calibration_profile: str
    disabled_strategies: frozenset[str]
    daily_loss_limit_pct: float
    max_consecutive_losses: int
    scalp_live_armed: bool
    scalp_live_max_notional: float
    scalp_live_max_open: int
    symbol_notional_caps: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> ScalpConfig:
        products_raw = os.getenv("SCALP_PRODUCTS", "BTCUSDT,ETHUSDT")
        products = tuple(p.strip().upper() for p in products_raw.split(",") if p.strip())
        prefix = (os.getenv("SCALP_REDIS_PREFIX", "scalp") or "scalp").strip().rstrip(":")
        db = os.getenv(
            "DATABASE_PATH",
            os.path.join(_REPO_ROOT, "mystic_trading.db"),
        )
        symbol_caps: dict[str, float] = {}
        raw_caps = (os.getenv("SCALP_SYMBOL_NOTIONAL_CAPS_JSON") or "").strip()
        if raw_caps:
            try:
                parsed = json.loads(raw_caps)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        sym = str(k or "").strip().upper().replace("/", "")
                        try:
                            cap = float(v)
                        except (TypeError, ValueError):
                            continue
                        if sym and cap > 0:
                            symbol_caps[sym] = cap
            except Exception:
                symbol_caps = {}
        return cls(
            repo_root=_REPO_ROOT,
            database_path=db,
            redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            redis_key_prefix=prefix,
            products=products,
            scalp_paper_enabled=_bool("SCALP_PAPER_ENABLED", False),
            scalp_paper_auto_arm=_bool(
                "SCALP_PAPER_AUTO_ARM",
                _bool("SCALP_PAPER_ENABLED", False),
            ),
            scalp_live=_bool("SCALP_LIVE", False),
            fee_model_verified=_bool("SCALP_FEE_MODEL_VERIFIED", False),
            max_open_positions=int(os.getenv("SCALP_MAX_OPEN_POSITIONS", "1")),
            max_notional_paper=float(os.getenv("SCALP_MAX_NOTIONAL_PAPER", "150")),
            allow_repair_add=_bool("SCALP_ALLOW_REPAIR_ADD", False),
            allow_market_orders=_bool("SCALP_ALLOW_MARKET_ORDERS", False),
            strategy_id="scalp",
            exchange="binance_us",
            calibration_mode=_bool("SCALP_CALIBRATION_MODE", False),
            calibration_profile=(os.getenv("SCALP_CALIBRATION_PROFILE", "moderate") or "moderate").strip().lower(),
            disabled_strategies=frozenset(
                s.strip().lower()
                for s in (
                    os.getenv(
                        "SCALP_DISABLED_STRATEGIES",
                        # Ocean paper proof: only setups that ever produced genuine
                        # fills. Others were 100% STRATEGY_NO_SIGNAL / soft-rank noise.
                        "breakout_momentum,orderbook_tape_scalp,failed_breakdown_reversal,compression_breakout,volume_impulse_continuation,trend_pullback_micro,failed_breakout_reversal",
                    )
                    or ""
                ).split(",")
                if s.strip()
            ),
            daily_loss_limit_pct=float(os.getenv("SCALP_DAILY_LOSS_LIMIT_PCT", "0.05")),
            max_consecutive_losses=int(os.getenv("SCALP_MAX_CONSECUTIVE_LOSSES", "5")),
            scalp_live_armed=_bool("SCALP_LIVE_ARMED", False),
            scalp_live_max_notional=float(os.getenv("SCALP_LIVE_MAX_NOTIONAL", "50.0")),
            scalp_live_max_open=int(os.getenv("SCALP_LIVE_MAX_OPEN", "2")),
            symbol_notional_caps=symbol_caps,
        )

    def notional_cap_for_symbol(self, symbol: str) -> float:
        """Global paper notional capped by optional per-symbol override."""
        sym = (symbol or "").strip().upper().replace("/", "")
        per = float(self.symbol_notional_caps.get(sym) or 0.0)
        if per > 0:
            return min(float(self.max_notional_paper), per)
        return float(self.max_notional_paper)

    def assert_no_live_trading(self) -> None:
        """Hard block: raises if live trading attempted without proper arming.

        - SCALP_LIVE=false  →  always safe (paper mode, no-op).
        - SCALP_LIVE=true + SCALP_LIVE_ARMED=false  →  raises (misconfigured live attempt).
        - SCALP_LIVE=true + SCALP_LIVE_ARMED=true   →  passes (engine was explicitly armed).
        """
        if self.scalp_live and not os.getenv("SCALP_LIVE_ARMED", "false").lower() == "true":
            raise RuntimeError("SCALP_LIVE=true but SCALP_LIVE_ARMED is not set. Set SCALP_LIVE_ARMED=true and arm the engine explicitly before live trading.")
        if self.allow_market_orders and not self.scalp_live:
            raise RuntimeError("SCALP_ALLOW_MARKET_ORDERS must remain false in paper mode.")
        if self.calibration_mode and self.scalp_live:
            raise RuntimeError("SCALP_CALIBRATION_MODE requires SCALP_LIVE=false")


def get_scalp_config() -> ScalpConfig:
    return ScalpConfig.from_env()
