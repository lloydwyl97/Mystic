"""Scalp-specific economics — isolated from Mystic DAY trading_economics."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _optional_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    return float(raw)


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ScalpEconomics:
    maker_fee_pct: float
    taker_fee_pct: float
    slippage_buffer_pct: float
    spread_cap_pct: float
    paper_spread_caps: dict[str, float] | None
    impact_cap_pct: float
    min_net_edge_pct: float
    net_profit_target_pct: float
    entry_edge_buffer_pct: float
    entry_required_gross_edge_pct_env: float | None
    min_projected_surplus_pct: float
    stale_scalp_timeout_sec: int
    fee_model_verified: bool
    use_maker_only: bool

    @classmethod
    def from_env(cls) -> ScalpEconomics:
        # Reference Mystic Binance.US defaults without importing DAY sell thresholds.
        default_taker = float(os.getenv("BINANCE_US_TAKER_FEE_PCT", "0.0002"))
        default_maker = float(os.getenv("BINANCE_US_MAKER_FEE_PCT", "0.0"))
        return cls(
            maker_fee_pct=float(os.getenv("SCALP_MAKER_FEE_PCT", str(default_maker))),
            taker_fee_pct=float(os.getenv("SCALP_TAKER_FEE_PCT", str(default_taker))),
            slippage_buffer_pct=float(os.getenv("SCALP_SLIPPAGE_BUFFER_PCT", "0.0001")),
            spread_cap_pct=float(os.getenv("SCALP_SPREAD_CAP_PCT", "0.0005")),
            paper_spread_caps=None,
            impact_cap_pct=float(os.getenv("SCALP_IMPACT_CAP_PCT", "0.0005")),
            min_net_edge_pct=float(os.getenv("SCALP_MIN_NET_EDGE_PCT", "0.0015")),
            net_profit_target_pct=float(os.getenv("SCALP_NET_PROFIT_TARGET_PCT", "0.0025")),
            entry_edge_buffer_pct=float(os.getenv("SCALP_ENTRY_EDGE_BUFFER_PCT", "0.001")),
            entry_required_gross_edge_pct_env=_optional_float("SCALP_ENTRY_REQUIRED_GROSS_EDGE_PCT"),
            min_projected_surplus_pct=float(os.getenv("SCALP_MIN_PROJECTED_SURPLUS_PCT", "0.0005")),
            stale_scalp_timeout_sec=int(os.getenv("SCALP_STALE_TIMEOUT_SEC", "300")),
            fee_model_verified=_bool("SCALP_FEE_MODEL_VERIFIED", False),
            use_maker_only=_bool("SCALP_USE_MAKER_ONLY", False),
        )

    @property
    def roundtrip_fee_pct(self) -> float:
        return self.entry_fee_pct() + self.exit_fee_pct()

    def entry_fee_pct(self, *, entry_maker: bool | None = None) -> float:
        if entry_maker is None:
            entry_maker = self.use_maker_only
        return self.maker_fee_pct if entry_maker else self.taker_fee_pct

    def exit_fee_pct(self, *, exit_maker: bool | None = None) -> float:
        if exit_maker is None:
            exit_maker = self.use_maker_only
        return self.maker_fee_pct if exit_maker else self.taker_fee_pct

    def roundtrip_fee_for_mode(self, *, entry_maker: bool, exit_maker: bool) -> float:
        return self.entry_fee_pct(entry_maker=entry_maker) + self.exit_fee_pct(exit_maker=exit_maker)

    def break_even_move_pct(
        self,
        spread_pct: float,
        buy_impact_pct: float,
        sell_impact_pct: float,
        *,
        entry_maker: bool | None = None,
        exit_maker: bool | None = None,
    ) -> float:
        if entry_maker is None:
            entry_maker = self.use_maker_only
        if exit_maker is None:
            exit_maker = self.use_maker_only
        return self.roundtrip_fee_for_mode(entry_maker=entry_maker, exit_maker=exit_maker) + spread_pct + buy_impact_pct + sell_impact_pct + self.slippage_buffer_pct

    def required_gross_move_for_min_edge_pct(
        self,
        spread_pct: float,
        buy_impact_pct: float,
        sell_impact_pct: float,
        *,
        entry_maker: bool | None = None,
        exit_maker: bool | None = None,
    ) -> float:
        return (
            self.break_even_move_pct(
                spread_pct,
                buy_impact_pct,
                sell_impact_pct,
                entry_maker=entry_maker,
                exit_maker=exit_maker,
            )
            + self.min_net_edge_pct
        )

    def is_fee_model_verified(self) -> bool:
        return self.fee_model_verified

    def spread_cap_for_symbol(self, symbol: str) -> float:
        """Uniform cap unless paper_spread_caps attached (paper/calibration only)."""
        sym = str(symbol).strip().upper()
        if self.paper_spread_caps and sym in self.paper_spread_caps:
            return self.paper_spread_caps[sym]
        return self.spread_cap_pct

    def entry_required_gross_edge_pct(
        self,
        spread_pct: float,
        buy_impact_pct: float,
        sell_impact_pct: float,
    ) -> float:
        """net_profit_target + roundtrip costs + buffer (Phase 3c entry gate)."""
        computed = self.net_profit_target_pct + self.roundtrip_cost_pct(spread_pct, buy_impact_pct, sell_impact_pct) + self.entry_edge_buffer_pct
        if self.entry_required_gross_edge_pct_env is not None:
            return max(computed, self.entry_required_gross_edge_pct_env)
        return computed

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "maker_fee_pct": self.maker_fee_pct,
            "taker_fee_pct": self.taker_fee_pct,
            "roundtrip_fee_pct_active": self.roundtrip_fee_pct,
            "slippage_buffer_pct": self.slippage_buffer_pct,
            "spread_cap_pct": self.spread_cap_pct,
            "impact_cap_pct": self.impact_cap_pct,
            "min_net_edge_pct": self.min_net_edge_pct,
            "net_profit_target_pct": self.net_profit_target_pct,
            "entry_edge_buffer_pct": self.entry_edge_buffer_pct,
            "entry_required_gross_edge_pct_env": self.entry_required_gross_edge_pct_env,
            "min_projected_surplus_pct": self.min_projected_surplus_pct,
            "stale_scalp_timeout_sec": self.stale_scalp_timeout_sec,
            "fee_model_verified": self.fee_model_verified,
            "use_maker_only": self.use_maker_only,
        }

    def roundtrip_cost_pct(
        self,
        spread_pct: float,
        buy_impact_pct: float,
        sell_impact_pct: float,
    ) -> float:
        return self.break_even_move_pct(spread_pct, buy_impact_pct, sell_impact_pct)

    def executable_exit_net_pct(
        self,
        entry_price: float,
        sell_fill_price: float,
        *,
        entry_buy_impact_pct: float,
        exit_sell_impact_pct: float,
    ) -> float:
        """
        Net move for closing an open scalp using executable sell fill.
        Entry spread is embedded in entry_price (ask-side); do not re-walk buy book.
        """
        if entry_price <= 0:
            return -1.0
        gross = (sell_fill_price - entry_price) / entry_price
        costs = self.entry_fee_pct() + self.exit_fee_pct() + self.slippage_buffer_pct * 2.0 + entry_buy_impact_pct + exit_sell_impact_pct
        return gross - costs

    def projected_entry_edge_pct(self, spread_pct: float, order_book_imbalance: float) -> float:
        """Rule-based scalp edge proxy — no ML; uses book imbalance only."""
        imb = max(0.0, float(order_book_imbalance))
        projected = imb * spread_pct * 8.0
        if imb >= 0.12:
            projected += spread_pct * 0.25
        return projected
