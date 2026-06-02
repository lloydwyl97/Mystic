#!/usr/bin/env python3
"""
Subnet config builder (Windows-friendly)
- Validates inputs
- Writes UTF-8 JSON with indentation
- Returns the config dict
- CLI: python subnet_config.py --chain-name MysticSubnet --validators 5 --out subnet_config.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class SubnetConfig(TypedDict):
    chain: str
    consensus: str
    validators: int
    ai_controlled: bool


def build_subnet_config(
    chain_name: str = "MysticSubnet",
    validators: int = 5,
    out_path: str | Path = "subnet_config.json",
) -> SubnetConfig:
    """
    Build and save a subnet configuration JSON file.

    Args:
        chain_name: Name of the subnet/chain (non-empty string)
        validators: Number of validators (>= 1)
        out_path: Output file path for the JSON

    Returns:
        The configuration dict.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not isinstance(chain_name, str) or not chain_name.strip():
        msg = "chain_name must be a non-empty string"
        raise ValueError(msg)

    # Accept ints or int-like
    if not isinstance(validators, int):
        try:
            validators = int(validators)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            msg = "validators must be an integer"
            raise ValueError(msg) from e

    if validators < 1:
        msg = "validators must be >= 1"
        raise ValueError(msg)

    config: SubnetConfig = {
        "chain": chain_name.strip(),
        "consensus": "PoS",
        "validators": validators,
        "ai_controlled": True,
    }

    out = Path(out_path)
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info(f"[SUBNET] Config saved for {config['chain']} -> {out.resolve()}")
    return config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build subnet_config.json")
    parser.add_argument("--chain-name", default="MysticSubnet", help="Subnet/chain name")
    parser.add_argument("--validators", type=int, default=5, help="Number of validators (>=1)")
    parser.add_argument("--out", default="subnet_config.json", help="Output JSON file path")
    args = parser.parse_args()

    build_subnet_config(chain_name=args.chain_name, validators=args.validators, out_path=args.out)
