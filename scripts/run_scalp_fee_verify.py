#!/usr/bin/env python3
"""Verify Binance.US scalp fee model and print audit summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.services.binance_scalp.diagnostics import build_economics_audit
    from backend.services.binance_scalp.economics import ScalpEconomics

    econ = ScalpEconomics.from_env()
    report = build_economics_audit()
    fd = report["summary"]["global_fee_diagnostic"]
    out = {
        "maker_fee_pct": fd["maker_fee_pct"],
        "taker_fee_pct": fd["taker_fee_pct"],
        "roundtrip_fee_pct": fd["roundtrip_fee_pct_active"],
        "slippage_buffer_pct": fd["slippage_buffer_pct"],
        "min_net_edge_pct": fd["min_net_edge_pct"],
        "total_break_even_move_pct": fd["total_break_even_move_pct"],
        "total_required_move_for_min_edge_pct": fd[
            "total_required_move_for_min_edge_pct"
        ],
        "fee_model_verified_env": econ.is_fee_model_verified(),
        "reference": "Binance.US spot defaults via SCALP_* / BINANCE_US_* env",
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
