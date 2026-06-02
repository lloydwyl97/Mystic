"""
Structured decision trace for BUY/SELL/HOLD auditing.

Logs action, symbol, and a context dict (scores, thresholds, sizing, reason_code)
so live runs can be inspected for "why BUY/SELL/HOLD" and sizing behavior.
Enable with env: DECISION_TRACE_ENABLED=true (default false).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

DECISION_TRACE_ENABLED = os.getenv("DECISION_TRACE_ENABLED", "false").lower() == "true"


def log_decision_trace(action: str, symbol: str, context_dict: dict[str, Any]) -> None:
    """
    Emit a structured decision trace log line.

    action: "BUY" | "SELL" | "HOLD"
    symbol: e.g. "BTC/USDT"
    context_dict: e.g. signal_score, confidence, threshold, cash, equity,
                  target_notional, final_notional, qty, reason_code, pnl_pct, net_pnl_pct, etc.
    """
    if not DECISION_TRACE_ENABLED:
        return
    payload = {
        "ts": time.time(),
        "action": action,
        "symbol": symbol,
        **context_dict,
    }
    with contextlib.suppress(Exception):
        logger.info("DECISION_TRACE %s", json.dumps(payload, default=str))
