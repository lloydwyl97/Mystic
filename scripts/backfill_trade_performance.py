#!/usr/bin/env python3
"""Backfill trade_performance from canonical paper_trades SELL rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.trade_performance_tracker import backfill_trade_performance_from_paper_trades


if __name__ == "__main__":
    result = backfill_trade_performance_from_paper_trades()
    print(json.dumps(result, indent=2))
