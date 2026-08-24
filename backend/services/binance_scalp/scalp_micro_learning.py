"""Learn from executable markouts — rank/size only, never eligibility.

Buckets: symbol × OFI tercile × queue-imbalance tercile × adverse tercile.
Reward positive forward executable net; penalize negative.
Requires sample size + recency; missing data is a zero adjustment.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL = 20.0
MIN_N = 8
RANK_CAP = 0.035
SIZE_LO = 0.70
SIZE_HI = 1.15


def _bucket_key(symbol: str, ofi: float, obi: float, adverse: float) -> str:
    def terc(x: float) -> str:
        if x < -0.15:
            return "neg"
        if x > 0.15:
            return "pos"
        return "neu"

    adv = "hi" if adverse >= 0.45 else ("mid" if adverse >= 0.20 else "lo")
    return f"{symbol.upper()}|{terc(ofi)}|{terc(obi)}|{adv}"


def _load_stats(db_path: str) -> dict[str, Any]:
    now = time.time()
    hit = _CACHE.get(db_path)
    if hit and hit[0] > now:
        return hit[1]
    stats: dict[str, Any] = {}
    try:
        with sqlite3.connect(db_path, timeout=1.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT symbol, extra_json, points_json, t0
                FROM scalp_micro_markouts
                WHERE kind='entry' AND t0 > ?
                ORDER BY t0 DESC LIMIT 4000
                """,
                (now - 14 * 86400,),
            ).fetchall()
    except sqlite3.OperationalError:
        _CACHE[db_path] = (now + 5.0, {})
        return {}
    except Exception:
        _CACHE[db_path] = (now + 5.0, {})
        return {}
    for row in rows:
        extra = {}
        points = {}
        try:
            extra = json.loads(row["extra_json"] or "{}")
            points = json.loads(row["points_json"] or "{}")
        except Exception:
            continue
        net = None
        for h in ("10", "5", "30", 10, 5, 30):
            pt = points.get(str(h)) or points.get(h)
            if isinstance(pt, dict) and "executable_net_markout" in pt:
                net = float(pt["executable_net_markout"])
                break
        if net is None:
            continue
        key = _bucket_key(
            str(row["symbol"]),
            float(extra.get("ofi_5s") or 0.0),
            float(extra.get("obi_l5") or extra.get("obi_l1") or 0.0),
            float(extra.get("adverse_selection_score") or 0.0),
        )
        rec = stats.setdefault(key, {"n": 0, "sum": 0.0, "pos": 0})
        rec["n"] += 1
        rec["sum"] += net
        rec["pos"] += int(net > 0)
    _CACHE[db_path] = (now + _CACHE_TTL, stats)
    return stats


def micro_learning_adjustments(
    db_path: str,
    *,
    symbol: str,
    ofi_5s: float = 0.0,
    obi_l5: float = 0.0,
    adverse_selection_score: float = 0.0,
) -> dict[str, Any]:
    """Return rank_delta and size_mult. Never a hard block."""
    stats = _load_stats(db_path)
    key = _bucket_key(symbol, ofi_5s, obi_l5, adverse_selection_score)
    rec = stats.get(key) or {}
    n = int(rec.get("n") or 0)
    if n < MIN_N:
        return {
            "consumed": False,
            "n": n,
            "rank_delta": 0.0,
            "size_mult": 1.0,
            "bucket": key,
            "eligibility": False,
        }
    exp = float(rec["sum"]) / n
    conf = min(1.0, n / 40.0)
    rank_delta = max(-RANK_CAP, min(RANK_CAP, 80.0 * exp * conf))
    size_mult = max(SIZE_LO, min(SIZE_HI, 1.0 + 40.0 * exp * conf))
    return {
        "consumed": True,
        "n": n,
        "expectancy": round(exp, 8),
        "rank_delta": round(rank_delta, 6),
        "size_mult": round(size_mult, 4),
        "bucket": key,
        "eligibility": False,
    }


def reset_learning_cache() -> None:
    _CACHE.clear()


__all__ = ["micro_learning_adjustments", "reset_learning_cache"]
