#!/usr/bin/env python3
"""
RETIRED — legacy agent orchestrator is not part of the supported Mystic stack.

Use: ./start_mystic.sh core
See: CANONICAL_SYSTEM.md
"""

from __future__ import annotations

import sys

print(
    "ERROR: start_agent_orchestrator.py is retired. Use ./start_mystic.sh core (DAY + scalp paper).",
    file=sys.stderr,
)
sys.exit(1)
