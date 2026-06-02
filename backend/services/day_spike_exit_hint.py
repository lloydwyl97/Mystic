"""DAY net-profit sells only — optional heuristic when RSI-like residuals fade across faster TFs."""

from __future__ import annotations

# Subset inspected for spike/fade coherence (requires valid bundle rows per TF).
FOCUS_SPIKE_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")


def spike_profit_fading_from_bundle(bundle: dict[str, object] | None) -> bool:
    """
    True when faster-TF compact-block RSI residuals were stretched and are now collapsing,
    while caller still confirms net-profit economics separately.
    """
    if not bundle or not isinstance(bundle, dict):
        return False
    from backend.services.ai_day_htf_features import compact_htf_block_31

    scores: list[float] = []
    for tf in FOCUS_SPIKE_TIMEFRAMES:
        rows = bundle.get(tf)
        if not isinstance(rows, list) or len(rows) < 10:
            return False
        span = rows[-min(400, len(rows)) :]
        blk = compact_htf_block_31(span)
        scores.append(float(blk[6]))
    if len(scores) < 4:
        return False
    fast = sum(scores[:3]) / 3.0
    slow = sum(scores[-2:]) / 2.0
    now = scores[0]
    return fast > 0.10 and slow < fast * 0.5 and now < fast * 0.7


__all__ = ["FOCUS_SPIKE_TIMEFRAMES", "spike_profit_fading_from_bundle"]
