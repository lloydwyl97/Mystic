"""DAY outcome bandit — promote winning arms, starve losing arms.

Thompson sampling over arms keyed by (symbol, setup, regime).
Updated on every closed DAY sell from realized net PnL.
Used as the primary ranking key for BUY selection.

Risk/safety gates (spread, max open, duplicate symbol, paper/live) stay elsewhere.
This module never hard-blocks; toxic arms get near-zero size + bottom rank.
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


def get_arm_stats(
    symbol: str,
    setup: str,
    regime: str,
    *,
    db_path: str | Path,
) -> dict[str, Any]:
    ensure_bandit_schema(db_path)
    key = arm_key(symbol, setup, regime)
    with sqlite3.connect(str(db_path), timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM day_outcome_bandit_arms WHERE arm_key=?",
            (key,),
        ).fetchone()
    if row is None:
        return {
            "arm_key": key,
            "alpha": PRIOR_ALPHA,
            "beta": PRIOR_BETA,
            "mean": 0.5,
            "n_obs": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "starved": False,
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
    if starved:
        size_factor = STARVE_SIZE_FLOOR
        # Pull sample toward zero so ranking buries the arm when better peers exist.
        sample = min(sample, float(stats["mean"]) * 0.5)
    elif int(stats["n_obs"] or 0) == 0:
        size_factor = 1.0
    else:
        # Map mean [0,1] → size [floor, 1.35]
        size_factor = max(EXPLORE_SIZE_FLOOR, min(1.35, 0.4 + float(stats["mean"]) * 1.2))
    return {
        **stats,
        "sample": float(sample),
        "size_factor": float(size_factor),
        "hard_block": False,
        "candidate_eligible": True,
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
    setup = identity.get("setup_type_canonical") or str(
        dd.get("setup_type") or dd.get("entry_thesis") or "UNKNOWN"
    )
    regime = identity.get("day_route_regime") or str(
        dd.get("day_route_regime") or dd.get("regime") or "range"
    )
    sampled = sample_arm(symbol, setup, regime, db_path=db_path, rng=rng)
    prior_fss = 0.0
    try:
        prior_fss = float(dd.get("final_selection_score") or dd.get("selection_score") or 0.0)
    except (TypeError, ValueError):
        prior_fss = 0.0
    # Compress prior FSS into ~[-0.5, 0.5] then blend with bandit sample in [0,1].
    prior_norm = math.tanh(prior_fss)
    bandit_score = (
        BANDIT_BLEND_PRIMARY * float(sampled["sample"])
        + BANDIT_BLEND_SECONDARY * (0.5 + 0.5 * prior_norm)
    )
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
    lookback: int = 120,
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
        record_bandit_outcome(
            symbol=str(symbol),
            setup=setup,
            regime=regime,
            pnl_usd=float(pnl or 0.0),
            exit_reason=str(exit_reason or ""),
            db_path=db_path,
        )
        count += 1
    if count:
        logger.info("DAY_BANDIT_BOOTSTRAP hydrated %d sells into arms", count)
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
