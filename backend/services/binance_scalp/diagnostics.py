"""Binance.US scalp diagnostics — economics audit and book-walk reports."""

from __future__ import annotations

from typing import Any

from backend.services.binance_scalp.config import ScalpConfig, get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.protected_preflight import run_scalp_preflight


def _mode_viable(
    econ: ScalpEconomics,
    spread: float,
    buy_impact: float,
    sell_impact: float,
    *,
    entry_maker: bool,
    exit_maker: bool,
) -> dict[str, Any]:
    break_even = econ.break_even_move_pct(spread, buy_impact, sell_impact, entry_maker=entry_maker, exit_maker=exit_maker)
    required = break_even + econ.min_net_edge_pct
    immediate = spread + buy_impact + sell_impact
    return {
        "entry_maker": entry_maker,
        "exit_maker": exit_maker,
        "roundtrip_fee_pct": econ.roundtrip_fee_for_mode(entry_maker=entry_maker, exit_maker=exit_maker),
        "break_even_move_pct": break_even,
        "required_gross_move_for_min_edge_pct": required,
        "immediate_microstructure_edge_pct": immediate,
        "economically_viable": required <= immediate,
    }


def build_fee_diagnostic(
    econ: ScalpEconomics,
    *,
    spread_pct: float = 0.0,
    buy_impact_pct: float = 0.0,
    sell_impact_pct: float = 0.0,
) -> dict[str, Any]:
    break_even = econ.break_even_move_pct(spread_pct, buy_impact_pct, sell_impact_pct)
    return {
        **econ.as_dict(),
        "spread_pct_observed": spread_pct,
        "buy_impact_pct_observed": buy_impact_pct,
        "sell_impact_pct_observed": sell_impact_pct,
        "total_break_even_move_pct": break_even,
        "total_required_move_for_min_edge_pct": econ.required_gross_move_for_min_edge_pct(spread_pct, buy_impact_pct, sell_impact_pct),
        "env_knobs": {
            "SCALP_MAKER_FEE_PCT": econ.maker_fee_pct,
            "SCALP_TAKER_FEE_PCT": econ.taker_fee_pct,
            "SCALP_MIN_NET_EDGE_PCT": econ.min_net_edge_pct,
            "SCALP_SLIPPAGE_BUFFER_PCT": econ.slippage_buffer_pct,
            "SCALP_SPREAD_CAP_PCT": econ.spread_cap_pct,
            "SCALP_IMPACT_CAP_PCT": econ.impact_cap_pct,
            "SCALP_FEE_MODEL_VERIFIED": econ.fee_model_verified,
            "SCALP_USE_MAKER_ONLY": econ.use_maker_only,
        },
    }


def build_pair_book_walk(
    symbol: str,
    config: ScalpConfig | None = None,
    econ: ScalpEconomics | None = None,
) -> dict[str, Any]:
    cfg = config or get_scalp_config()
    economics = econ or ScalpEconomics.from_env()
    reader = ScalpMarketReader(cfg)
    snap = reader.read(symbol)
    if snap is None:
        return {"symbol": symbol, "error": "NO_MARKET_DATA"}

    notional = cfg.max_notional_paper
    buy_pf = run_scalp_preflight(snap, economics, cfg, side="BUY", notional_usd=notional, check_paper_enabled=False)
    sell_qty = notional / snap.mid if snap.mid > 0 else 0.0
    sell_pf = run_scalp_preflight(
        snap,
        economics,
        cfg,
        side="SELL",
        quantity=sell_qty,
        check_paper_enabled=False,
    )

    walks: dict[str, Any] = {}
    for label, n in (("$5", 5.0), ("$10", 10.0), ("$25", 25.0)):
        pf = run_scalp_preflight(
            snap,
            economics,
            cfg,
            side="BUY",
            notional_usd=n,
            check_paper_enabled=False,
        )
        walks[label] = {
            "depth_sufficient": pf.depth_sufficient,
            "expected_avg_fill": pf.expected_avg_fill,
            "buy_impact_pct": pf.buy_impact_pct,
            "sell_impact_pct": pf.sell_impact_pct,
            "levels_consumed": pf.levels_consumed,
            "impact_cap_pass": pf.buy_impact_pct <= economics.impact_cap_pct,
            "preflight_pass": pf.passed,
            "reject_reason": pf.reject_reason,
        }

    return {
        "symbol": snap.symbol_bus,
        "best_bid": snap.best_bid,
        "best_ask": snap.best_ask,
        "spread_pct": snap.spread_pct,
        "redis_orderbook_spread": snap.redis_spread_pct,
        "book_source": snap.book_source,
        "buy_impact_notional_25": buy_pf.buy_impact_pct,
        "sell_impact_qty_25usd": sell_pf.sell_impact_pct,
        "buy_walks": walks,
        "preflight_buy_25": buy_pf.as_dict(),
        "mode_comparison": {
            "taker_taker": _mode_viable(
                economics,
                snap.spread_pct,
                buy_pf.buy_impact_pct,
                sell_pf.sell_impact_pct,
                entry_maker=False,
                exit_maker=False,
            ),
            "maker_maker": _mode_viable(
                economics,
                snap.spread_pct,
                buy_pf.buy_impact_pct,
                sell_pf.sell_impact_pct,
                entry_maker=True,
                exit_maker=True,
            ),
            "maker_taker": _mode_viable(
                economics,
                snap.spread_pct,
                buy_pf.buy_impact_pct,
                sell_pf.sell_impact_pct,
                entry_maker=True,
                exit_maker=False,
            ),
        },
    }


def build_economics_audit(
    config: ScalpConfig | None = None,
    econ: ScalpEconomics | None = None,
) -> dict[str, Any]:
    cfg = config or get_scalp_config()
    economics = econ or ScalpEconomics.from_env()
    reader = ScalpMarketReader(cfg)
    products: dict[str, Any] = {}
    spreads: list[float] = []
    buy_imps: list[float] = []
    sell_imps: list[float] = []

    for sym in cfg.products:
        snap = reader.read(sym)
        if snap is None:
            products[sym] = {"error": "NO_MARKET_DATA"}
            continue
        pf = run_scalp_preflight(
            snap,
            economics,
            cfg,
            side="BUY",
            notional_usd=cfg.max_notional_paper,
            check_paper_enabled=False,
        )
        sell_qty = cfg.max_notional_paper / snap.mid if snap.mid > 0 else 0.0
        spf = run_scalp_preflight(
            snap,
            economics,
            cfg,
            side="SELL",
            quantity=sell_qty,
            check_paper_enabled=False,
        )
        spreads.append(snap.spread_pct)
        buy_imps.append(pf.buy_impact_pct)
        sell_imps.append(spf.sell_impact_pct)
        products[sym] = {
            "fee_diagnostic": build_fee_diagnostic(
                economics,
                spread_pct=snap.spread_pct,
                buy_impact_pct=pf.buy_impact_pct,
                sell_impact_pct=spf.sell_impact_pct,
            ),
            "mode_comparison": build_pair_book_walk(sym, cfg, economics).get("mode_comparison", {}),
            "preflight_buy": pf.as_dict(),
        }

    n = len(spreads) or 1
    summary = build_fee_diagnostic(
        economics,
        spread_pct=sum(spreads) / n,
        buy_impact_pct=sum(buy_imps) / n,
        sell_impact_pct=sum(sell_imps) / n,
    )
    any_viable = any(
        products[s].get("mode_comparison", {}).get(mode, {}).get("economically_viable") for s in products if "error" not in products[s] for mode in ("taker_taker", "maker_maker", "maker_taker")
    )
    return {
        "summary": {
            "fee_model_verified": economics.is_fee_model_verified(),
            "scalp_paper_enabled": cfg.scalp_paper_enabled,
            "scalp_live": cfg.scalp_live,
            "paper_scalper_blocked_reason": ("FEE_MODEL_UNVERIFIED" if not economics.is_fee_model_verified() else "SCALP_PAPER_DISABLED" if not cfg.scalp_paper_enabled else None),
            "scalping_viable_under_current_fees": any_viable,
            "global_fee_diagnostic": summary,
        },
        "products": products,
    }


def build_book_walk_report(config: ScalpConfig | None = None) -> dict[str, Any]:
    cfg = config or get_scalp_config()
    econ = ScalpEconomics.from_env()
    return {
        "impact_cap_pct": econ.impact_cap_pct,
        "spread_cap_pct": econ.spread_cap_pct,
        "max_notional_paper": cfg.max_notional_paper,
        "fee_model_verified": econ.is_fee_model_verified(),
        "products": {sym: build_pair_book_walk(sym, cfg, econ) for sym in cfg.products},
    }
