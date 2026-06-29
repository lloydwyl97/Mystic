#!/usr/bin/env python3
"""
DISABLED — not part of the current DAY trading engine.

backend/services/ai_position_tracker.py is not shipped. Exits are handled by
portfolio_engine.monitor_all_positions / _check_exit_conditions only.
"""

from __future__ import annotations

import sys

print(
    "ERROR: start_ai_position_tracker.py is disabled (backend.services.ai_position_tracker not present; not part of DAY engine).",
    file=sys.stderr,
)
sys.exit(1)
