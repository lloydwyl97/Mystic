"""Replay captured live L2/aggTrade sample when present."""

from __future__ import annotations

import json
from pathlib import Path

import backend.services.microstructure_engine as m
from backend.services.binance_scalp.scalp_micro_replay import replay_events

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "scalp_l2_sample.jsonl"


def _events_from_fixture(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        stream = str(rec.get("stream") or "")
        data = rec.get("data") or {}
        ts = float(rec.get("ts") or 0.0)
        if "depth" in stream:
            bids = [[float(p), float(q)] for p, q in (data.get("bids") or [])[:20]]
            asks = [[float(p), float(q)] for p, q in (data.get("asks") or [])[:20]]
            if bids and asks:
                events.append(
                    {
                        "kind": "snapshot",
                        "ts": ts,
                        "bids": bids,
                        "asks": asks,
                        "last_update_id": data.get("lastUpdateId"),
                    }
                )
        elif "aggTrade" in stream or "trade" in stream:
            qty = data.get("q") or data.get("qty")
            ibm = data.get("m")
            if qty is not None:
                events.append(
                    {
                        "kind": "trade",
                        "ts": ts,
                        "qty": float(qty),
                        "is_buyer_maker": bool(ibm),
                    }
                )
    return events


def test_captured_sample_replays_when_present():
    if not FIXTURE.exists():
        return
    events = _events_from_fixture(FIXTURE)
    assert len(events) >= 4
    m._STATE.clear()
    # Replay BTC-shaped events only if mixed — filter by first snapshot book.
    out = replay_events(events[:80], symbol="BTCUSDT")
    assert out["n_events"] >= 4
    assert "healthy" in out["book"]
    m._STATE.clear()
