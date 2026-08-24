"""OLD vs NEW SCALP selection on captured L2 — no profitability fabrication."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import backend.services.microstructure_engine as m
from backend.services.binance_scalp.scalp_micro_ev import multi_horizon_ev
from backend.services.microstructure_engine import compute_features, record_agg_trade, record_snapshot

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "scalp_l2_sample.jsonl"


def _base(stream: str) -> str:
    return stream.split("@")[0].replace("usdt", "").upper()


def main() -> int:
    if not FIXTURE.exists():
        print("NO_CAPTURE")
        return 1
    m._STATE.clear()
    last: dict[str, dict] = {}
    n_snap = n_trade = 0
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        stream = str(rec.get("stream") or "")
        data = rec.get("data") or {}
        ts = float(rec.get("ts") or 0.0)
        sym = _base(stream) + "USDT"
        if "depth" in stream:
            bids = [[float(p), float(q)] for p, q in (data.get("bids") or [])[:20]]
            asks = [[float(a[0]), float(a[1])] for a in (data.get("asks") or [])[:20]]
            if bids and asks:
                record_snapshot(sym, bids, asks, ts=ts)
                n_snap += 1
        elif "aggTrade" in stream:
            record_agg_trade(sym, float(data.get("q") or 0), bool(data.get("m")), ts=ts)
            n_trade += 1
        feats = compute_features(sym)
        if feats:
            last[sym] = feats
    print(f"CAPTURE snaps={n_snap} trades={n_trade} symbols={sorted(last)}")
    print("NEW microstructure decision-time (last captured state per coin):")
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
        feats = last.get(sym) or {}
        ev = multi_horizon_ev(feats) if feats else {}
        print(
            f"  {sym} age={feats.get('data_age_sec')} "
            f"obi_l1={feats.get('obi_l1')} ofi_5s={feats.get('ofi_5s')} "
            f"mp={feats.get('microprice_pressure')} adv={feats.get('adverse_selection_score')} "
            f"EV_10s={ev.get('EV_10s')} p_pos_10s={ev.get('p_positive_executable_net_10s')} "
            f"select={ev.get('selection_micro_score')}"
        )
    print("OLD model note: accepted artifact remains scalp_path_net_v1 (40-dim).")
    print("NEW model_version=scalp_micro_ev_v1 calibration=INCONCLUSIVE (capture too short for ECE).")
    print("No economic verdict claimed from this bounded capture.")
    m._STATE.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
