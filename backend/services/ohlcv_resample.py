"""Resample CCXT-style OHLCV rows [ts_ms, o, h, low, c, v] onto a fixed bar size in seconds."""

from __future__ import annotations


def resample_ohlcv_to_seconds(ohlcv_1m: list[list], bucket_seconds: int) -> list[list]:
    """
    Aggregate consecutive 1m bars into ``bucket_seconds`` buckets (floor bucket start in UTC epoch seconds).

    Incomplete leading/trailing partial buckets are dropped if they have no rows.
    """
    if not ohlcv_1m or bucket_seconds < 60:
        return list(ohlcv_1m)
    buckets: dict[int, list[list]] = {}
    for row in ohlcv_1m:
        if not isinstance(row, (list, tuple)) or len(row) < 6:
            continue
        ts = int(float(row[0]))
        sec = ts // 1000
        b0 = (sec // bucket_seconds) * bucket_seconds
        b_ms = b0 * 1000
        buckets.setdefault(b_ms, []).append([float(row[0])] + [float(x) for x in row[1:6]])
    out: list[list] = []
    for b_ms in sorted(buckets.keys()):
        rows = buckets[b_ms]
        o = float(rows[0][1])
        h = max(float(r[2]) for r in rows)
        low = min(float(r[3]) for r in rows)
        c = float(rows[-1][4])
        v = sum(float(r[5]) for r in rows)
        out.append([b_ms, o, h, low, c, v])
    return out


__all__ = ["resample_ohlcv_to_seconds"]
