"""
Orderbook spread vs protected preflight ceiling — explains SPREAD_TOO_WIDE idle capital.

Observation only unless DAY_PAPER_ALIGN_SPREAD_WITH_BAR=true (paper sim).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from backend.config.protected_execution import (
    MAX_ORDERBOOK_SPREAD_PCT,
    bar_rank_max_spread_fraction,
    day_paper_align_spread_with_bar_enabled,
    effective_max_orderbook_spread_pct,
)
from backend.config.trading_universe import DAY_TRADE_SYMBOLS

logger = logging.getLogger(__name__)


def _decode_map(raw: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw.items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


def _spread_from_fields(data: dict[str, str]) -> tuple[float, float, float] | None:
    try:
        if data.get("bid_ask_spread") not in (None, ""):
            spread = float(data["bid_ask_spread"])
            return 0.0, 0.0, spread
        bid = float(data.get("bid") or data.get("best_bid") or 0)
        ask = float(data.get("ask") or data.get("best_ask") or 0)
        if bid <= 0 or ask <= 0 or ask < bid:
            return None
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / mid if mid > 0 else 1.0
        return bid, ask, spread
    except (TypeError, ValueError):
        return None


def build_spread_preflight_snapshot() -> dict[str, Any]:
    """Per top-4 symbol: live spread vs execution preflight cap vs bar-rank cap."""
    rows: list[dict[str, Any]] = []
    blocked_exec = 0
    blocked_bar = 0
    max_exec = MAX_ORDERBOOK_SPREAD_PCT
    max_bar = bar_rank_max_spread_fraction()
    max_paper = effective_max_orderbook_spread_pct(live_capable=False)

    try:
        from backend.config.redis_config import get_redis_client

        r = get_redis_client()
    except Exception:
        r = None

    for sym in DAY_TRADE_SYMBOLS:
        bus = sym.strip().upper().replace("/", "")
        base = bus.replace("USDT", "")
        spread_pct: float | None = None
        bid = ask = 0.0
        source = "missing"

        if r:
            for key in (f"orderbook:{base}", f"orderbook:{bus}"):
                try:
                    raw = r.hgetall(key)
                    if raw:
                        parsed = _spread_from_fields(_decode_map(raw))
                        if parsed:
                            bid, ask, spread_pct = parsed
                            source = key
                            break
                except Exception:
                    continue
            if spread_pct is None:
                try:
                    mraw = r.get(f"market:{bus}")
                    if mraw:
                        mtxt = mraw.decode() if isinstance(mraw, bytes) else str(mraw)
                        mdata = json.loads(mtxt) if mtxt.strip().startswith("{") else {}
                        if isinstance(mdata, dict):
                            parsed = _spread_from_fields(
                                {k: str(v) for k, v in mdata.items()}
                            )
                            if parsed:
                                bid, ask, spread_pct = parsed
                                source = f"market:{bus}"
                except Exception:
                    pass

        exec_ok = spread_pct is not None and spread_pct <= max_paper + 1e-15
        bar_ok = spread_pct is not None and spread_pct <= max_bar + 1e-15
        if spread_pct is not None and not exec_ok:
            blocked_exec += 1
        if spread_pct is not None and not bar_ok:
            blocked_bar += 1

        rows.append(
            {
                "symbol": bus,
                "spread_pct": round(spread_pct, 6) if spread_pct is not None else None,
                "spread_bps": round(spread_pct * 10000, 2) if spread_pct is not None else None,
                "best_bid": bid,
                "best_ask": ask,
                "source": source,
                "preflight_ok_paper": exec_ok if spread_pct is not None else None,
                "bar_rank_ok": bar_ok if spread_pct is not None else None,
            }
        )

    spread_passing = [r["symbol"] for r in rows if r.get("preflight_ok_paper") is True]

    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_orderbook_spread_pct": max_exec,
        "max_bar_spread_pct": max_bar,
        "effective_paper_spread_pct": max_paper,
        "effective_paper_spread_bps": round(max_paper * 10000, 2),
        "paper_align_with_bar": day_paper_align_spread_with_bar_enabled(),
        "symbols": rows,
        "spread_passing_symbols": spread_passing,
        "spread_passing_count": len(spread_passing),
        "blocked_by_exec_spread_count": blocked_exec,
        "blocked_by_bar_spread_count": blocked_bar,
        "telemetry_only": not day_paper_align_spread_with_bar_enabled(),
        "note": "Bar rank uses penalty-only; execute_buy preflight hard-blocks on effective_paper_spread_pct",
    }


__all__ = ["build_spread_preflight_snapshot"]
