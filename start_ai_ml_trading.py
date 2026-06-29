#!/usr/bin/env python3
"""
RETIRED — not part of the supported DAY stack.

DAY ML signals are produced by start_ai_signal_generator.py (per-coin RF .pkl)
and consumed by start_portfolio_engine_integration.py.

Use: ./start_mystic.sh full
See: CANONICAL_SYSTEM.md
"""

from __future__ import annotations

import sys

print(
    "ERROR: start_ai_ml_trading.py is retired. Use ./start_mystic.sh full (start_ai_signal_generator + portfolio integration).",
    file=sys.stderr,
)
sys.exit(1)
