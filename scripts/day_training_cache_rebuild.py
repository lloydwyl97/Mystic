#!/usr/bin/env python3
"""Rebuild DAY training-cache vectors point-in-time at their 4h anchors (dry run unless --apply).

--apply copies every ``*_day_latest.json`` into --backup-dir first and needs the
learning process stopped (it holds the cache in memory and rewrites the files).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.canonical_candle_intervals import interval_ms
from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES, fetch_limit_for_day_tf
from backend.services.day_training_cache_rebuild import anchor_asof_ms, rebuild_cache_samples


async def _klines(symbol: str, tf: str, start_ms: int, end_ms: int) -> list[list]:
    from backend.services.canonical_candle_pipeline import CanonicalCandlePipeline

    pipe = CanonicalCandlePipeline()
    width = interval_ms(tf)
    rows: dict[int, list] = {}
    cursor = start_ms
    while cursor <= end_ms:
        page = await pipe.fetch_binance(symbol, tf, start_ms=cursor, end_ms=min(end_ms, cursor + width * 999), limit=1000)
        if not page:
            break
        for k in page:
            rows[int(k[0])] = [int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])]
        nxt = int(page[-1][0]) + width
        if nxt <= cursor:
            break
        cursor = nxt
    return [rows[k] for k in sorted(rows)]


def main() -> int:
    from backend.utils.path_helpers import ensure_model_directories

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--backup-dir", default="")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    args.dir = args.dir or str(ensure_model_directories()["training_data"])
    if args.apply and not args.backup_dir:
        print("--apply requires --backup-dir", file=sys.stderr)
        return 2

    reports: dict[str, object] = {}
    now_ms = int(time.time() * 1000)
    for path in sorted(Path(args.dir).glob("*_day_latest.json")):
        bus = path.name.split("_day_latest.json")[0]
        samples = json.loads(path.read_text())
        anchors = [int(s["label_anchor_4h_open_ms"]) for s in samples if s.get("label_anchor_4h_open_ms") is not None]
        if not anchors:
            continue
        end = min(now_ms, anchor_asof_ms(max(anchors)))

        async def _load(bus: str = bus, first: int = min(anchors), end: int = end) -> dict[str, list[list]]:
            hist: dict[str, list[list]] = {}
            for tf in DAY_ACTIVE_TIMEFRAMES:
                hist[tf] = await _klines(bus, tf, first - (fetch_limit_for_day_tf(tf) + 2) * interval_ms(tf), end)
            return hist

        history = asyncio.run(_load())
        ccxt = f"{bus[:-4]}/USDT" if bus.endswith("USDT") else bus
        rebuilt, rep = rebuild_cache_samples(samples, history, ccxt)
        rep["history_bars"] = {tf: len(v) for tf, v in history.items()}
        rep["first_anchor_ms"] = min(anchors)
        rep["last_anchor_ms"] = max(anchors)
        reports[path.name] = rep
        if args.apply and rep["rebuilt"]:
            bdir = Path(args.backup_dir)
            bdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bdir / path.name)
            tmp = path.with_suffix(".json.rebuild_tmp")
            tmp.write_text(json.dumps(rebuilt))
            tmp.replace(path)
            rep["written"] = True
        del history
    text = json.dumps({"mode": "apply" if args.apply else "dry_run", "files": reports}, indent=2)
    if args.report:
        Path(args.report).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
