import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_vote(
    proposal_text: str,
    impact_score: float,
    risk_score: float,
    score_bounds: tuple[float, float] = (0.0, 1.0),
    min_text_len: int = 3,
    max_text_len: int = 500,
) -> dict[str, Any]:
    def clamp(v: float, lo: float, hi: float) -> float:
        try:
            v = float(v)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            v = 0.0
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    text = " ".join((proposal_text or "").split()).strip()
    if len(text) < min_text_len:
        text = text.ljust(min_text_len, ".")
    if len(text) > max_text_len:
        text = text[:max_text_len]

    lo, hi = score_bounds
    impact = clamp(impact_score, lo, hi)
    risk = clamp(risk_score, lo, hi)

    decision = "YES" if impact > risk else "NO"
    margin = round(impact - risk, 6)

    logger.info(f"[GOVERNANCE] Proposal: {text}")
    logger.info(f"[GOVERNANCE] Vote: {decision} (impact={impact:.4f}, risk={risk:.4f}, margin={margin:+.4f})")

    return {
        "proposal": text,
        "decision": decision,
        "impact_score": impact,
        "risk_score": risk,
        "margin": margin,
        "bounds": {"min": lo, "max": hi},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
