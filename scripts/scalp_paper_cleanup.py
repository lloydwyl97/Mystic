#!/usr/bin/env python3
"""One-shot paper scalp safe-state cleanup."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.session_cleanup import full_teardown


def main() -> int:
    db = os.getenv("DATABASE_PATH", str(REPO / "mystic_trading.db"))
    result = full_teardown(db, close_reason="SCALP_TEST_ORPHAN_RESET")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("safe_state_verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
