#!/usr/bin/env python3
"""One-shot: close ETH legacy inventory with LEGACY_INVENTORY_CLEANUP_EXIT."""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> int:
    from backend.services.portfolio_engine import initialize_portfolio_engine

    engine = await initialize_portfolio_engine()
    pos = engine.open_positions.get("ETH/USDT")
    if not pos:
        for k in list(engine.open_positions.keys()):
            if "ETH" in k.upper():
                pos = engine.open_positions[k]
                sym = k
                break
        else:
            print(json.dumps({"ok": False, "error": "no_eth_position"}))
            return 1
    else:
        sym = "ETH/USDT"

    entry = float(pos.entry_price or 0)
    entry_ts = float(pos.entry_time or 0)
    qty = float(pos.quantity or 0)
    result = await engine.execute_legacy_inventory_cleanup(sym)
    if not result:
        print(json.dumps({"ok": False, "error": "cleanup_failed", "symbol": sym}))
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "symbol": sym,
                "entry_price": entry,
                "entry_time": entry_ts,
                "sell_price": result.get("price"),
                "realized_pnl": result.get("realized_pnl"),
                "exit_reason": result.get("exit_reason") or "LEGACY_INVENTORY_CLEANUP_EXIT",
                "quantity": qty,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
