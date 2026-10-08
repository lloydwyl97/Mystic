"""Research-only SCALP separability and queue-supported execution. Not live authority.

A passive bid fills only when later seller-initiated prints consume the size
that was already displayed ahead of it, or when a print trades through that
price. Cancels and a quiet tape are not fills. Maker results are a separate
target from the taker policy net and never enter accounting.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FILL_KIND = "QUEUE_SUPPORTED_PASSIVE"


def passive_fill(
    queue_ahead: float,
    our_qty: float,
    prints: list[tuple[float, float, bool]],
    posted_price: float,
    *,
    hit_bid: bool,
) -> dict[str, Any]:
    """Walk prints after the order is posted.

    ``hit_bid`` is true for a buy posted on the bid: only seller-aggressor
    prints can fill it. A buy posted as a passive ask uses buyer-aggressor
    prints. ``prints`` are ``(price, qty, seller_was_aggressor)``.
    """
    empty = {"filled_qty": 0.0, "fraction": 0.0, "full": False, "through": False, "kind": FILL_KIND}
    if our_qty <= 0.0 or posted_price <= 0.0 or not math.isfinite(queue_ahead) or queue_ahead < 0.0:
        return empty
    ahead = float(queue_ahead)
    filled = 0.0
    through = False
    for price, qty, seller in prints:
        if qty <= 0.0 or not math.isfinite(price):
            continue
        aggressor_hits_us = seller if hit_bid else (not seller)
        if not aggressor_hits_us:
            continue
        trades_through = price < posted_price if hit_bid else price > posted_price
        at_price = (not trades_through) and abs(price - posted_price) <= posted_price * 1e-8
        if trades_through:
            filled = our_qty
            through = True
            break
        if not at_price:
            continue
        remaining = qty
        if ahead > 0.0:
            consumed = min(ahead, remaining)
            ahead -= consumed
            remaining -= consumed
        if ahead <= 0.0 and remaining > 0.0:
            filled = min(our_qty, filled + remaining)
            if filled >= our_qty:
                break
    fraction = 0.0 if our_qty <= 0.0 else filled / our_qty
    return {"filled_qty": filled, "fraction": fraction, "full": 0.0 < our_qty <= filled, "through": through, "kind": FILL_KIND}


def candidate_net(filled_fraction: float, roundtrip_net: float) -> float:
    """Unfilled size earns nothing. A fill is never invented to avoid a loser."""
    fraction = min(1.0, max(0.0, float(filled_fraction)))
    return fraction * float(roundtrip_net)


def maker_entry_taker_exit(entry_bid: float, exit_bid: float, *, maker_fee: float, taker_fee: float, slippage: float) -> float | None:
    if entry_bid <= 0.0 or exit_bid <= 0.0:
        return None
    return (float(exit_bid) - float(entry_bid)) / float(entry_bid) - float(maker_fee) - float(taker_fee) - float(slippage)


def fit_first_component(past: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """First principal direction of already-resolved rows. The row being scored is not included."""
    if past.ndim != 2 or past.shape[0] < 3 or past.shape[1] < 1:
        return None
    mu = past.mean(axis=0)
    sd = past.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    centered = (past - mu) / sd
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return mu, sd, vt[0]


def project_component(row: np.ndarray, fitted: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> float:
    if fitted is None:
        return 0.0
    mu, sd, direction = fitted
    return float(((row - mu) / sd) @ direction)


def point_biserial(values: list[float], labels: list[int]) -> float | None:
    """Correlation of one causal feature with a resolved group label. The label is not a gate."""
    if len(values) != len(labels) or len(values) < 4:
        return None
    if len(set(labels)) < 2:
        return None
    xs = np.array(values, dtype=float)
    ys = np.array(labels, dtype=float)
    if float(xs.std()) <= 0.0 or float(ys.std()) <= 0.0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def cohens_d(positive: list[float], negative: list[float]) -> float | None:
    if len(positive) < 2 or len(negative) < 2:
        return None
    a = np.array(positive, dtype=float)
    b = np.array(negative, dtype=float)
    pooled = math.sqrt(((len(a) - 1) * float(a.var(ddof=1)) + (len(b) - 1) * float(b.var(ddof=1))) / (len(a) + len(b) - 2))
    if pooled <= 1e-12:
        return None
    return float((a.mean() - b.mean()) / pooled)


def range_overlap(positive: list[float], negative: list[float]) -> float | None:
    """Share of the combined range that both groups occupy. 1 means the same span."""
    if not positive or not negative:
        return None
    lo = min(*positive, *negative)
    hi = max(*positive, *negative)
    if hi <= lo:
        return 1.0
    overlap_lo = max(min(positive), min(negative))
    overlap_hi = min(max(positive), max(negative))
    return max(0.0, overlap_hi - overlap_lo) / (hi - lo)


def information_value(groups: list[dict[str, Any]]) -> dict[str, float | None]:
    """Economic value of knowing the group, the coin, or both. Research only."""
    if not groups:
        return {"perfect_both": None, "perfect_group_current_rank": None, "current_group_perfect_rank": None, "always_best": None}
    both, group_then_rank, rank_given_live, always = [], [], [], []
    for group in groups:
        nets = group["nets"]
        live = group.get("live") or {}
        coins = list(nets)
        best_coin = max(coins, key=lambda coin: float(nets[coin]))
        best = float(nets[best_coin])
        always.append(best)
        both.append(max(best, 0.0))
        if best > 0.0 and all(live.get(coin) is not None for coin in coins):
            ranked = max(coins, key=lambda coin: float(live[coin]))
            group_then_rank.append(float(nets[ranked]))
        else:
            group_then_rank.append(0.0)
        live_values = [live.get(coin) for coin in coins]
        if all(value is not None for value in live_values) and max(float(value) for value in live_values) > 0.0:
            rank_given_live.append(best)
        else:
            rank_given_live.append(0.0)

    def avg(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "perfect_both": avg(both),
        "perfect_group_current_rank": avg(group_then_rank),
        "current_group_perfect_rank": avg(rank_given_live),
        "always_best": avg(always),
    }
