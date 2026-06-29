#!/usr/bin/env python3
"""
DISABLED — not part of the current DAY trading engine.

backend/services/ai_outcome_bridge.py is not shipped. Learning uses
ai_training_pipeline / trade_learning_outcomes without this bridge.
"""

from __future__ import annotations

import sys

print(
    "ERROR: start_ai_outcome_bridge.py is disabled (backend.services.ai_outcome_bridge not present; not part of DAY engine).",
    file=sys.stderr,
)
sys.exit(1)
