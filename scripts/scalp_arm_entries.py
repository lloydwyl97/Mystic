#!/usr/bin/env python3
"""Arm or disarm scalp paper entries via Redis control key."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import redis

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.scalp_control import (
    is_entry_armed,
    set_entry_armed,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["arm", "disarm", "status"])
    args = parser.parse_args()
    cfg = get_scalp_config()
    if cfg.scalp_live:
        print("SCALP_LIVE must be false", file=sys.stderr)
        return 1
    client = redis.from_url(cfg.redis_url, decode_responses=True)
    prefix = cfg.redis_key_prefix
    if args.action == "arm":
        set_entry_armed(client, prefix=prefix, armed=True)
    elif args.action == "disarm":
        set_entry_armed(client, prefix=prefix, armed=False)
    print("entry_armed", is_entry_armed(client, prefix=prefix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
