"""DAY outcome bandit — promote winning arms, starve losing arms.

Thompson sampling over arms keyed by (symbol, setup, regime).
Updated on every closed DAY sell from realized net PnL.
Used as the primary ranking key for BUY selection.

Risk/safety gates (spread, max open, duplicate symbol, paper/live) stay elsewhere.
The bandit itself never hard-blocks; toxic arms get near-zero size + bottom rank.

Exception: the opt-in env var DAY_BLOCK_SETUP_REGIME_PAIRS (comma-separated
"SETUP:regime" pairs, e.g. "HTF_TREND_PULLBACK:range") applies an explicit
hard-block on named (setup, regime) combinations. Used to cull structural
losers whose bandit starvation is too slow to prevent ongoing bleed. Empty by
default (no blocks); safe to set/unset via .env with a restart.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
MIN_OBS_FOR_STARVE = 3
STARVE_MEAN_MAX = 0.35
STARVE_SIZE_FLOOR = 0.08
EXPLORE_SIZE_FLOOR = 0.35
WIN_PNL_SCALE = 12.0  # usd → extra alpha weight
LOSS_PNL_SCALE = 12.0
MAX_WEIGHT = 3.0
SLIDING_WINDOW = 40
BANDIT_BLEND_PRIMARY = 0.72  # share of selection score from bandit sample
BANDIT_BLEND_SECONDARY = 0.28  # residual from existing final_selection_score

# Batch 7: widened size curve.
# Fresh arms (< N_OBS_FOR_UPSIZE) are capped at MAX_SIZE_FRESH so the bandit
# does not upsize before it has enough data to know if the arm is truly a
# winner. Well-observed arms can scale up to MAX_SIZE_TOP with strong mean
# so proven winners actually get more capital. Starved arms keep the tiny
# STARVE_SIZE_FLOOR unchanged. This replaces the old hard cap of 1.35 with
# a data-gated curve.
N_OBS_FOR_UPSIZE = int(os.getenv("DAY_BANDIT_N_OBS_FOR_UPSIZE", "8") or "8")
MAX_SIZE_FRESH = float(os.getenv("DAY_BANDIT_MAX_SIZE_FRESH", "1.15") or "1.15")
MAX_SIZE_TOP = float(os.getenv("DAY_BANDIT_MAX_SIZE_TOP", "2.00") or "2.00")

# Hierarchical (partial-pooling) prior: when an arm has zero observations we
# borrow evidence from peer arms with the same (setup, regime) family across
# other symbols, then also from the (symbol, regime) family across other
# setups, blended and shrunk toward the uninformative Beta(1,1). This makes
# a fresh (BTC/USDT, VWAP_REVERSION, bull) arm start closer to the empirical
# behavior of VWAP_REVERSION in bull rather than at 50/50, so early trades
# still ride real signal instead of coin-flipping prior mass.
HIERARCHICAL_PRIOR_ENABLED_DEFAULT = True
HIERARCHICAL_PRIOR_MAX_WEIGHT = 2.5  # max effective peer weight added to α+β
HIERARCHICAL_PRIOR_SETUP_REGIME_WEIGHT = 0.6
HIERARCHICAL_PRIOR_SYMBOL_REGIME_WEIGHT = 0.4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS day_outcome_bandit_arms (
    arm_key TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    setup TEXT NOT NULL,
    regime TEXT NOT NULL,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0.0,
    last_pnl REAL,
    last_exit_reason TEXT,
    last_updated REAL NOT NULL,
    n_obs INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_day_bandit_symbol ON day_outcome_bandit_arms(symbol);
"""


def bandit_enabled() -> bool:
    return os.getenv("DAY_OUTCOME_BANDIT_ENABLED", "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def hierarchical_prior_enabled() -> bool:
    default = "true" if HIERARCHICAL_PRIOR_ENABLED_DEFAULT else "false"
    return os.getenv("DAY_BANDIT_HIERARCHICAL_PRIOR_ENABLED", default).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper().replace("-", "/")
    if "/" not in s and s.endswith("USDT"):
        s = s[:-4] + "/USDT"
    return s


def _normalize_setup(setup: str) -> str:
    s = str(setup or "").strip().upper()
    if s == "TREND_PULLBACK":
        return "HTF_TREND_PULLBACK"
    if s.startswith("BREAKOUT") and s != "BREAKOUT_CONTINUATION":
        return "BREAKOUT_CONTINUATION"
    return s or "UNKNOWN"


def _normalize_regime(regime: str) -> str:
    r = str(regime or "").strip().lower()
    if r in ("neutral", "chop", "sideways"):
        return "range"
    if r in ("bull", "bear", "range"):
        return r
    if "bull" in r or "trend_up" in r or "trending_up" in r:
        return "bull"
    if "bear" in r or "trend_down" in r or "trending_down" in r:
        return "bear"
    if "range" in r:
        return "range"
    return "range"


def arm_key(symbol: str, setup: str, regime: str) -> str:
    return f"{_normalize_symbol(symbol)}|{_normalize_setup(setup)}|{_normalize_regime(regime)}"


def ensure_bandit_schema(db_path: str | Path) -> None:
    with sqlite3.connect(str(db_path), timeout=15) as conn:
        conn.executescript(_SCHEMA)
        conn.commit()


def _is_win(pnl: float, exit_reason: str) -> bool:
    er = str(exit_reason or "").upper()
    if "NET_PROFIT" in er and pnl > 0:
        return True
    if pnl > 0.5:  # clear green after costs
        return True
    return False


def _weight(pnl: float) -> float:
    return float(min(MAX_WEIGHT, 1.0 + abs(float(pnl)) / (WIN_PNL_SCALE if pnl >= 0 else LOSS_PNL_SCALE)))


def record_bandit_outcome(
    *,
    symbol: str,
    setup: str,
    regime: str,
    pnl_usd: float,
    exit_reason: str,
    db_path: str | Path,
    trade_id: str | None = None,
) -> dict[str, Any]:
    """Update Thompson posterior for the arm from a closed DAY trade."""
    if not bandit_enabled():
        return {"applied": False, "reason": "disabled"}
    ensure_bandit_schema(db_path)
    sym = _normalize_symbol(symbol)
    st = _normalize_setup(setup)
    reg = _normalize_regime(regime)
    key = arm_key(sym, st, reg)
    pnl = float(pnl_usd or 0.0)
    win = _is_win(pnl, exit_reason)
    w = _weight(pnl)
    now = time.time()

    with sqlite3.connect(str(db_path), timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT alpha, beta, wins, losses, total_pnl, n_obs FROM day_outcome_bandit_arms WHERE arm_key=?",
            (key,),
        ).fetchone()
        if row is None:
            alpha, beta, wins, losses, total_pnl, n_obs = PRIOR_ALPHA, PRIOR_BETA, 0, 0, 0.0, 0
        else:
            alpha = float(row["alpha"] or PRIOR_ALPHA)
            beta = float(row["beta"] or PRIOR_BETA)
            wins = int(row["wins"] or 0)
            losses = int(row["losses"] or 0)
            total_pnl = float(row["total_pnl"] or 0.0)
            n_obs = int(row["n_obs"] or 0)

        if win:
            alpha += w
            wins += 1
        else:
            beta += w
            losses += 1
        total_pnl += pnl
        n_obs += 1

        # Soft decay when window exceeded (keep prior mass).
        if n_obs > SLIDING_WINDOW:
            decay = 0.92
            alpha = PRIOR_ALPHA + (alpha - PRIOR_ALPHA) * decay
            beta = PRIOR_BETA + (beta - PRIOR_BETA) * decay

        conn.execute(
            """
            INSERT INTO day_outcome_bandit_arms (
                arm_key, symbol, setup, regime, alpha, beta, wins, losses,
                total_pnl, last_pnl, last_exit_reason, last_updated, n_obs
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arm_key) DO UPDATE SET
                alpha=excluded.alpha,
                beta=excluded.beta,
                wins=excluded.wins,
                losses=excluded.losses,
                total_pnl=excluded.total_pnl,
                last_pnl=excluded.last_pnl,
                last_exit_reason=excluded.last_exit_reason,
                last_updated=excluded.last_updated,
                n_obs=excluded.n_obs
            """,
            (
                key,
                sym,
                st,
                reg,
                float(alpha),
                float(beta),
                wins,
                losses,
                float(total_pnl),
                float(pnl),
                str(exit_reason or "")[:120],
                now,
                n_obs,
            ),
        )
        conn.commit()

    mean = float(alpha / (alpha + beta)) if (alpha + beta) > 0 else 0.5
    logger.info(
        "DAY_BANDIT_UPDATE arm=%s win=%s pnl=%.3f α=%.3f β=%.3f mean=%.3f n=%d trade=%s",
        key,
        win,
        pnl,
        alpha,
        beta,
        mean,
        n_obs,
        str(trade_id or "")[:40],
    )
    return {
        "applied": True,
        "arm_key": key,
        "win": win,
        "alpha": alpha,
        "beta": beta,
        "mean": mean,
        "n_obs": n_obs,
    }


def _peer_prior_alpha_beta(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    setup: str,
    regime: str,
) -> tuple[float, float, int]:
    """Compute a hierarchical (partial-pooling) prior α, β for a fresh arm.

    Blends two peer families:
    1. Same (setup, regime), different symbol — how does this setup behave in
       this regime elsewhere in the universe?
    2. Same (symbol, regime), different setup — how does this symbol behave in
       this regime across setups?

    Each family contributes at most HIERARCHICAL_PRIOR_MAX_WEIGHT / 2 units of
    effective observation mass. Result is added to Beta(1,1) so an arm with no
    peer support still falls back to the uninformative prior.
    """
    peer_alpha = 0.0
    peer_beta = 0.0
    peer_used = 0
    for label, weight, query, params in (
        (
            "setup_regime",
            HIERARCHICAL_PRIOR_SETUP_REGIME_WEIGHT,
            """
            SELECT alpha, beta, n_obs FROM day_outcome_bandit_arms
            WHERE setup = ? AND regime = ? AND symbol != ?
            """,
            (setup, regime, symbol),
        ),
        (
            "symbol_regime",
            HIERARCHICAL_PRIOR_SYMBOL_REGIME_WEIGHT,
            """
            SELECT alpha, beta, n_obs FROM day_outcome_bandit_arms
            WHERE symbol = ? AND regime = ? AND setup != ?
            """,
            (symbol, regime, setup),
        ),
    ):
        del label  # kept for grepability during debugging
        rows = conn.execute(query, params).fetchall()
        fam_a = 0.0
        fam_b = 0.0
        fam_n = 0
        for r in rows:
            fam_a += max(0.0, float(r["alpha"] or 0.0) - PRIOR_ALPHA)
            fam_b += max(0.0, float(r["beta"] or 0.0) - PRIOR_BETA)
            fam_n += int(r["n_obs"] or 0)
        if fam_n <= 0:
            continue
        # Shrink family mass to at most (weight * MAX_WEIGHT) effective obs
        # so a huge peer set does not overwhelm the true prior.
        cap = HIERARCHICAL_PRIOR_MAX_WEIGHT * weight
        scale = min(1.0, cap / (fam_a + fam_b + 1e-9))
        peer_alpha += fam_a * scale
        peer_beta += fam_b * scale
        peer_used += fam_n
    return peer_alpha, peer_beta, peer_used


def get_arm_stats(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str | Path,
) -> dict[str, Any]:
    ensure_bandit_schema(db_path)
    sym = _normalize_symbol(symbol)
    st = _normalize_setup(setup)
    reg = _normalize_regime(regime)
    key = arm_key(sym, st, reg)
    with sqlite3.connect(str(db_path), timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM day_outcome_bandit_arms WHERE arm_key=?",
            (key,),
        ).fetchone()
        if row is None:
            alpha = PRIOR_ALPHA
            beta = PRIOR_BETA
            peer_n = 0
            prior_source = "uniform"
            if hierarchical_prior_enabled():
                p_a, p_b, peer_n = _peer_prior_alpha_beta(conn, symbol=sym, setup=st, regime=reg)
                if peer_n > 0:
                    alpha = PRIOR_ALPHA + p_a
                    beta = PRIOR_BETA + p_b
                    prior_source = "hierarchical"
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            return {
                "arm_key": key,
                "alpha": alpha,
                "beta": beta,
                "mean": mean,
                "n_obs": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "starved": False,
                "prior_source": prior_source,
                "peer_n_obs": peer_n,
            }
    alpha = float(row["alpha"] or PRIOR_ALPHA)
    beta = float(row["beta"] or PRIOR_BETA)
    mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
    n_obs = int(row["n_obs"] or 0)
    starved = n_obs >= MIN_OBS_FOR_STARVE and mean < STARVE_MEAN_MAX and int(row["losses"] or 0) > int(row["wins"] or 0)
    return {
        "arm_key": key,
        "alpha": alpha,
        "beta": beta,
        "mean": mean,
        "n_obs": n_obs,
        "wins": int(row["wins"] or 0),
        "losses": int(row["losses"] or 0),
        "total_pnl": float(row["total_pnl"] or 0.0),
        "starved": starved,
        "prior_source": "empirical",
        "peer_n_obs": 0,
    }


def sample_arm(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str | Path,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    stats = get_arm_stats(symbol, setup, regime, db_path=db_path)
    r = rng or random.Random()
    alpha = max(1e-6, float(stats["alpha"]))
    beta = max(1e-6, float(stats["beta"]))
    try:
        sample = r.betavariate(alpha, beta)
    except ValueError:
        sample = 0.5
    starved = bool(stats["starved"])
    n_obs = int(stats.get("n_obs") or 0)
    peer_n = int(stats.get("peer_n_obs") or 0)
    mean = float(stats["mean"])
    if starved:
        size_factor = STARVE_SIZE_FLOOR
        # Pull sample toward zero so ranking buries the arm when better peers exist.
        sample = min(sample, mean * 0.5)
    elif n_obs == 0 and peer_n == 0:
        # No evidence at all → mid-range explore. Cap below MAX_SIZE_FRESH so
        # the bandit does not oversize a truly-unknown arm.
        size_factor = min(1.0, MAX_SIZE_FRESH)
    elif n_obs < N_OBS_FOR_UPSIZE:
        # Fresh arm (including peer-informed): use empirical mean but cap
        # size at MAX_SIZE_FRESH so an early lucky streak cannot bet the farm.
        raw = 0.4 + mean * (MAX_SIZE_FRESH - 0.4) * 2.0  # map mean [0,1] into [0.4, MAX_SIZE_FRESH+small]
        raw = min(raw, MAX_SIZE_FRESH)
        size_factor = max(EXPLORE_SIZE_FLOOR, raw)
    else:
        # Well-observed arm: allow size to reach MAX_SIZE_TOP for genuine
        # winners. Curve is linear in mean from the floor up to MAX_SIZE_TOP
        # so a mean of 0.5 sits at ~ (floor+top)/2.
        raw = EXPLORE_SIZE_FLOOR + mean * (MAX_SIZE_TOP - EXPLORE_SIZE_FLOOR)
        size_factor = max(EXPLORE_SIZE_FLOOR, min(MAX_SIZE_TOP, raw))
    return {
        **stats,
        "sample": float(sample),
        "size_factor": float(size_factor),
        "hard_block": False,
        "candidate_eligible": True,
        "size_factor_cap": ("starve" if starved else "fresh" if n_obs < N_OBS_FOR_UPSIZE else "top"),
    }


def apply_bandit_to_decision_data(
    decision_data: dict[str, Any],
    symbol: str,
    *,
    db_path: str | Path,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Stamp bandit fields and recompute final_selection_score with bandit primary."""
    dd = dict(decision_data or {})
    if not bandit_enabled():
        dd["day_bandit_enabled"] = False
        return dd

    from backend.services.day_trade_thesis import resolve_setup_identity

    identity = resolve_setup_identity(dd)
    setup = identity.get("setup_type_canonical") or str(dd.get("setup_type") or dd.get("entry_thesis") or "UNKNOWN")
    regime = identity.get("day_route_regime") or str(dd.get("day_route_regime") or dd.get("regime") or "range")

    # Opt-in explicit hard-block on named (setup, regime) pairs. Empty env = no
    # blocks. Format: "SETUP:regime,SETUP:regime" (case-insensitive on both
    # sides). Used when structural losers need to stop trading immediately
    # rather than wait for slow bandit starvation.
    _block_env = os.getenv("DAY_BLOCK_SETUP_REGIME_PAIRS", "").strip()
    if _block_env:
        _block_set = {p.strip().upper() for p in _block_env.split(",") if p.strip() and ":" in p}
        _key = f"{str(setup).upper()}:{str(regime).upper()}"
        if _key in _block_set:
            dd["day_bandit_enabled"] = True
            dd["day_bandit_hard_block_reason"] = f"SETUP_REGIME_BLOCKED:{setup}:{regime}"
            dd["hard_block"] = True
            dd["candidate_eligible"] = False
            dd["final_selection_score"] = 0.0
            dd["selection_score"] = 0.0
            logger.info(
                "DAY_SETUP_REGIME_HARD_BLOCK symbol=%s setup=%s regime=%s key=%s",
                symbol,
                setup,
                regime,
                _key,
            )
            return dd

    sampled = sample_arm(symbol, setup, regime, db_path=db_path, rng=rng)
    prior_fss = 0.0
    try:
        prior_fss = float(dd.get("final_selection_score") or dd.get("selection_score") or 0.0)
    except (TypeError, ValueError):
        prior_fss = 0.0
    # Compress prior FSS into ~[-0.5, 0.5] then blend with bandit sample in [0,1].
    prior_norm = math.tanh(prior_fss)
    bandit_score = BANDIT_BLEND_PRIMARY * float(sampled["sample"]) + BANDIT_BLEND_SECONDARY * (0.5 + 0.5 * prior_norm)
    if sampled["starved"]:
        bandit_score *= 0.35

    dd["day_bandit_enabled"] = True
    dd["day_bandit_arm_key"] = sampled["arm_key"]
    dd["day_bandit_alpha"] = round(float(sampled["alpha"]), 5)
    dd["day_bandit_beta"] = round(float(sampled["beta"]), 5)
    dd["day_bandit_mean"] = round(float(sampled["mean"]), 5)
    dd["day_bandit_sample"] = round(float(sampled["sample"]), 5)
    dd["day_bandit_n_obs"] = int(sampled["n_obs"])
    dd["day_bandit_wins"] = int(sampled["wins"])
    dd["day_bandit_losses"] = int(sampled["losses"])
    dd["day_bandit_total_pnl"] = round(float(sampled["total_pnl"]), 4)
    dd["day_bandit_starved"] = bool(sampled["starved"])
    dd["day_bandit_size_factor"] = round(float(sampled["size_factor"]), 4)
    dd["day_bandit_score"] = round(float(bandit_score), 6)
    dd["day_bandit_prior_source"] = str(sampled.get("prior_source") or "")
    dd["day_bandit_peer_n_obs"] = int(sampled.get("peer_n_obs") or 0)
    dd["final_selection_score_pre_bandit"] = prior_fss
    dd["final_selection_score"] = round(float(bandit_score), 6)
    dd["selection_score"] = dd["final_selection_score"]
    dd["hard_block"] = False
    dd["candidate_eligible"] = True
    dd["outcome_penalty_hard_block"] = False
    # Size path: multiply existing thesis size factor if present.
    try:
        prev_size = float(dd.get("thesis_size_factor") or 1.0)
    except (TypeError, ValueError):
        prev_size = 1.0
    dd["thesis_size_factor"] = round(max(STARVE_SIZE_FLOOR, prev_size * float(sampled["size_factor"])), 4)
    return dd


def bootstrap_bandit_from_paper_trades(
    db_path: str | Path,
    *,
    lookback: int = 240,
) -> int:
    """One-shot hydrate from recent DAY sells if arms table is empty."""
    ensure_bandit_schema(db_path)
    with sqlite3.connect(str(db_path), timeout=30) as conn:
        n_arms = conn.execute("SELECT COUNT(*) FROM day_outcome_bandit_arms").fetchone()[0]
        if int(n_arms or 0) > 0:
            return 0
        rows = conn.execute(
            """
            SELECT symbol, pnl, exit_reason, explainability_json
            FROM paper_trades
            WHERE UPPER(side)='SELL'
              AND COALESCE(strategy_id, 'day') = 'day'
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(lookback),),
        ).fetchall()
    import json

    count = 0
    skipped_unknown = 0
    # Oldest first so chronology is natural
    for symbol, pnl, exit_reason, ex_json in reversed(list(rows)):
        setup = "UNKNOWN"
        regime = "range"
        if ex_json:
            try:
                ex = json.loads(ex_json) if isinstance(ex_json, str) else dict(ex_json or {})
                setup = str(ex.get("setup_type_canonical") or ex.get("setup_type") or ex.get("entry_thesis") or "UNKNOWN")
                regime = str(ex.get("day_route_regime") or ex.get("regime") or "range")
            except Exception:
                pass
        # Skip legacy rows where the setup was never labeled — those arms muddy
        # the pool because they mix multiple real setups under one bucket.
        if str(setup or "").strip().upper() in ("", "UNKNOWN"):
            skipped_unknown += 1
            continue
        record_bandit_outcome(
            symbol=str(symbol),
            setup=setup,
            regime=regime,
            pnl_usd=float(pnl or 0.0),
            exit_reason=str(exit_reason or ""),
            db_path=db_path,
        )
        count += 1
    if count or skipped_unknown:
        logger.info(
            "DAY_BANDIT_BOOTSTRAP hydrated %d sells into arms (skipped_unknown=%d)",
            count,
            skipped_unknown,
        )
    return count


__all__ = [
    "apply_bandit_to_decision_data",
    "arm_key",
    "bandit_enabled",
    "bootstrap_bandit_from_paper_trades",
    "ensure_bandit_schema",
    "get_arm_stats",
    "record_bandit_outcome",
    "sample_arm",
]
