#!/usr/bin/env python3
"""Binance.US scalp book-walk check — read-only diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.binance_scalp.diagnostics import build_book_walk_report
    from backend.services.binance_scalp.schema import init_scalp_schema

    cfg = get_scalp_config()
    cfg.assert_no_live_trading()
    init_scalp_schema(cfg.database_path)
    print(json.dumps(build_book_walk_report(cfg), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
