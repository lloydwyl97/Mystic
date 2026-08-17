"""
DAY position health & opportunity-cost telemetry — observation only.

No sells, no rotation, no gate changes. Persists to operational_state + optional missed-opp rows.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from backend.config.trading_economics import (
    ESTIMATED_ROUNDTRIP_COST,
    MIN_NET_PROFIT_TO_SELL,
    is_net_profit_acceptable,
)
from backend.config.trading_universe import DAY_TRADE_SYMBOLS
from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)

OPERATIONAL_KEY = "day_position_health"
LAST_EXEC_BLOCK_KEY = "day_last_execution_block"
OBS_COOLDOWN_PREFIX = "day_health_obs:"
MIN_NOTIONAL_DEFAULT = 11.0
OBS_EMIT_COOLDOWN_SEC = 600

STATE_PROFIT_READY = "PROFIT_READY"
STATE_UNDERWATER = "UNDERWATER_HOLD"
STATE_LAGGING = "LAGGING_VS_BASKET"
STATE_HEALTHY = "TREND_HEALTHY_HOLD"


def _norm(sym: str) -> str:
    s = (sym or "").strip().upper().replace("/", "")
    return s if s.endswith("USDT") else f"{s}USDT"


def _api_to_ccxt(sym: str) -> str:
    s = _norm(sym)
    return f"{s[:-4]}/USDT" if s.endswith("USDT") else sym


def _decode_sig_str(raw_sig: dict, key: str) -> str:
    v = raw_sig.get(key) or raw_sig.get(key.encode() if isinstance(key, str) else key)
    if v is None:
        return ""
    if isinstance(v, bytes):
        try:
            return v.decode()
        except Exception:
            return ""
    return str(v)


def _redis_prices_and_signals() -> tuple[dict[str, float], list[dict[str, Any]]]:
    prices: dict[str, float] = {}
    signals: list[dict[str, Any]] = []
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if r is None:
            return prices, signals
        for sym in DAY_TRADE_SYMBOLS:
            base = sym.replace("USDT", "")
            raw_px = r.hget(f"price:{base}", "v")
            if raw_px:
                try:
                    prices[_api_to_ccxt(sym)] = float(raw_px.decode() if isinstance(raw_px, bytes) else raw_px)
                except (TypeError, ValueError):
                    pass
            raw_sig = r.hgetall(f"ai_signal:day:{sym}") or {}
            if not raw_sig:
                continue

            def _f(k: str, _sig: dict = raw_sig) -> float:
                v = _sig.get(k) or _sig.get(k.encode() if isinstance(k, str) else k)
                if v is None:
                    return 0.0
                try:
                    return float(v.decode() if isinstance(v, bytes) else v)
                except (TypeError, ValueError):
                    return 0.0

            conf = _f("winner_probability") or _f("confidence")
            rs = _f("ctx_rs_btc") or _f("ctx_rs_eth")
            side = _decode_sig_str(raw_sig, "side") or _decode_sig_str(raw_sig, "prediction") or _decode_sig_str(raw_sig, "argmax_action")
            side = str(side).upper()
            # buy_margin is the execution gate; winner_probability alone is often P(HOLD).
            buy_margin = _f("buy_margin")
            if buy_margin == 0.0 and (_f("prob_buy") or _f("prob_hold") or _f("prob_sell")):
                buy_margin = _f("prob_buy") - max(_f("prob_hold"), _f("prob_sell"))
            ts = _f("timestamp")
            age_sec = max(0.0, time.time() - ts) if ts > 0 else None
            ctx_age = _f("ctx_age_sec")
            signals.append(
                {
                    "symbol": _api_to_ccxt(sym),
                    "confidence": conf,
                    "winner_probability": conf,
                    "buy_margin": buy_margin,
                    "ctx_rs_btc": rs,
                    "side": side,
                    "action": side,
                    "prob_buy": _f("prob_buy"),
                    "prob_hold": _f("prob_hold"),
                    "prob_sell": _f("prob_sell"),
                    "regime": _decode_sig_str(raw_sig, "regime_label") or _decode_sig_str(raw_sig, "regime"),
                    "ctx_market_regime": _decode_sig_str(raw_sig, "ctx_market_regime"),
                    "signal_age_sec": age_sec,
                    "ctx_age_sec": ctx_age if ctx_age else None,
                    "stale": bool(age_sec is not None and age_sec > 180.0),
                    "fresh": bool(age_sec is None or age_sec <= 180.0),
                }
            )
        signals.sort(key=lambda x: (x.get("ctx_rs_btc", 0), x.get("confidence", 0)), reverse=True)
    except Exception as exc:
        logger.debug("day_position_health redis snapshot failed: %s", exc)
    return prices, signals


def _rs_rank(symbol: str, signals: list[dict[str, Any]]) -> int:
    sym = _api_to_ccxt(symbol)
    for i, s in enumerate(signals):
        if _api_to_ccxt(str(s.get("symbol", ""))) == sym:
            return i + 1
    return len(DAY_TRADE_SYMBOLS)


def _best_alternate(symbol: str, signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    sym = _api_to_ccxt(symbol)
    for s in signals:
        other = _api_to_ccxt(str(s.get("symbol", "")))
        if other != sym and str(s.get("side", "")).upper() in ("BUY", "STRONG_BUY", "1"):
            return s
        if other != sym and float(s.get("confidence") or 0) >= 0.55:
            return s
    return signals[0] if signals else None


def net_pct(entry: float, mark: float) -> float:
    if entry <= 0 or mark <= 0:
        return 0.0
    return (mark - entry) / entry - ESTIMATED_ROUNDTRIP_COST


def classify_position(
    *,
    symbol: str,
    entry_price: float,
    entry_epoch: float,
    quantity: float,
    mark: float,
    signals: list[dict[str, Any]],
) -> dict[str, Any]:
    npct = net_pct(entry_price, mark)
    net_usd = npct * entry_price * quantity
    profit_ready = is_net_profit_acceptable(npct, net_usd)
    hold_days = max(0.0, (time.time() - float(entry_epoch or time.time())) / 86400.0)
    rank = _rs_rank(symbol, signals)
    alt = _best_alternate(symbol, signals)
    alt_sym = _api_to_ccxt(str(alt.get("symbol", ""))) if alt else None

    if profit_ready:
        state = STATE_PROFIT_READY
        reason = "net_profit_floor_met"
    elif npct < 0 and rank >= 3 and hold_days >= 1.0:
        state = STATE_LAGGING
        reason = "underwater_lagging_rs_rank"
    elif npct < 0:
        state = STATE_UNDERWATER
        reason = "underwater_awaiting_profit_or_recovery"
    else:
        state = STATE_HEALTHY
        reason = "green_below_profit_floor"

    return {
        "symbol": _api_to_ccxt(symbol),
        "entry_price": entry_price,
        "mark": mark,
        "quantity": quantity,
        "net_pct": round(npct, 6),
        "net_usd": round(net_usd, 4),
        "profit_ready": profit_ready,
        "profit_floor_pct": MIN_NET_PROFIT_TO_SELL,
        "hold_days": round(hold_days, 2),
        "rs_rank": rank,
        "best_alternate_symbol": alt_sym,
        "best_alternate_confidence": float(alt.get("confidence") or 0) if alt else None,
        "state": state,
        "reason": reason,
        "trapped": npct < 0 and hold_days >= 1.0,
    }


def persist_last_execution_block(
    *,
    symbol: str,
    reject_reason: str,
    filter_name: str = "PROTECTED_PREFLIGHT",
    detail: dict[str, Any] | None = None,
    db_path: str = DATABASE_PATH,
) -> None:
    try:
        blob = json.dumps(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "reject_reason": reject_reason,
                "filter_name": filter_name,
                "detail": detail or {},
            },
            separators=(",", ":"),
        )
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO operational_state(key, value_json, updated_ts)
                VALUES(?, ?, strftime('%s','now'))
                ON CONFLICT(key) DO UPDATE SET
                  value_json=excluded.value_json,
                  updated_ts=excluded.updated_ts
                """,
                (LAST_EXEC_BLOCK_KEY, blob),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("persist_last_execution_block failed: %s", exc)


def load_last_execution_block(db_path: str = DATABASE_PATH) -> dict[str, Any] | None:
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT value_json FROM operational_state WHERE key=?",
                (LAST_EXEC_BLOCK_KEY,),
            ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def build_entry_reject_summary(db_path: str = DATABASE_PATH, limit: int = 8) -> list[dict[str, Any]]:
    """Top BUY reject reasons (observation for idle-capital diagnosis)."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT reason, COUNT(*) AS cnt
                FROM portfolio_engine_rejects
                WHERE UPPER(COALESCE(side, '')) = 'BUY'
                  AND datetime(ts) >= datetime('now', '-7 days')
                GROUP BY reason
                ORDER BY cnt DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [{"reject_reason": str(r[0]), "count": int(r[1])} for r in rows if r and r[0]]
    except Exception:
        return []


def build_portfolio_health(
    *,
    open_positions: dict[str, Any],
    current_prices: dict[str, float],
    cash_balance: float,
    max_open_positions: int,
    min_notional: float = MIN_NOTIONAL_DEFAULT,
    last_bar_skip_reason: str | None = None,
) -> dict[str, Any]:
    _, signals = _redis_prices_and_signals()
    active = [p for p in open_positions.values() if getattr(p, "status", "ACTIVE") != "DUST_PENDING"]
    open_count = len(active)
    slots_free = max(0, int(max_open_positions) - open_count)
    deployable = cash_balance >= min_notional and slots_free > 0

    positions_health: list[dict[str, Any]] = []
    trapped_days = 0.0
    for sym, pos in open_positions.items():
        if getattr(pos, "status", "ACTIVE") == "DUST_PENDING":
            continue
        mark = float(current_prices.get(sym) or current_prices.get(normalize_symbol(sym)) or 0)
        if mark <= 0:
            mark = float(getattr(pos, "entry_price", 0) or 0)
        ph = classify_position(
            symbol=sym,
            entry_price=float(pos.entry_price),
            entry_epoch=float(getattr(pos, "entry_time", 0) or time.time()),
            quantity=float(pos.quantity),
            mark=mark,
            signals=signals,
        )
        ep = float(getattr(pos, "entry_price", 0) or 0)
        hi = float(getattr(pos, "highest_price", 0) or 0)
        lo = float(getattr(pos, "lowest_price", 0) or 0)
        ph["setup"] = str(getattr(pos, "entry_thesis", "") or "")
        ph["setup_type"] = ph["setup"]
        ph["hold_minutes"] = round(max(0.0, (time.time() - float(getattr(pos, "entry_time", 0) or time.time())) / 60.0), 2)
        ph["mfe_pct"] = round((hi - ep) / ep, 6) if ep > 0 and hi > 0 else None
        ph["mae_pct"] = round((lo - ep) / ep, 6) if ep > 0 and lo > 0 else None
        positions_health.append(ph)
        if ph.get("trapped"):
            trapped_days = max(trapped_days, float(ph.get("hold_days") or 0))

    signal_freshness: dict[str, Any] = {}
    entry_quality: dict[str, Any] = {}
    try:
        from backend.config.trading_universe import DAY_TRADE_SYMBOLS
        from backend.services.ai_context_freshness_sync import build_freshness_snapshot

        signal_freshness = build_freshness_snapshot(list(DAY_TRADE_SYMBOLS))
    except Exception:
        pass
    try:
        from backend.services.day_entry_quality_gate import build_entry_quality_telemetry

        entry_quality = build_entry_quality_telemetry()
    except Exception:
        pass

    entry_reject_summary = build_entry_reject_summary()
    spread_preflight: dict[str, Any] = {}
    try:
        from backend.services.day_spread_preflight_telemetry import build_spread_preflight_snapshot

        spread_preflight = build_spread_preflight_snapshot()
    except Exception:
        pass
    last_execution_block = load_last_execution_block()

    idle_reason = None
    capital_idle_diagnosis: dict[str, Any] = {}
    if deployable:
        if last_bar_skip_reason:
            idle_reason = last_bar_skip_reason
        elif not signals:
            idle_reason = "NO_TOP4_SIGNALS_IN_REDIS"
        else:
            buy_signals = [s for s in signals if float(s.get("confidence") or 0) >= 0.5]
            if not buy_signals:
                idle_reason = "NO_HIGH_CONFIDENCE_BUY_SIGNAL"
            else:
                idle_reason = "CAPITAL_AVAILABLE_AWAITING_BAR_ENTRY"

        spread_passing = list(spread_preflight.get("spread_passing_symbols") or [])
        spread_blocked = int(spread_preflight.get("blocked_by_exec_spread_count") or 0)
        capital_idle_diagnosis = {
            "spread_passing_symbols": spread_passing,
            "spread_blocked_count": spread_blocked,
            "spread_cap_bps": spread_preflight.get("effective_paper_spread_bps"),
            "paper_align_with_bar": spread_preflight.get("paper_align_with_bar"),
            "last_execution_block": (last_execution_block or {}).get("reject_reason"),
            "trapped_lagging": any(p.get("state") == STATE_LAGGING for p in positions_health),
        }
        if idle_reason == "CAPITAL_AVAILABLE_AWAITING_BAR_ENTRY" and spread_blocked >= 3 and len(spread_passing) <= 1:
            if spread_passing:
                idle_reason = f"SPREAD_PREFLIGHT_TIGHT_CAP_ONLY_{spread_passing[0]}_PASSES"
            else:
                idle_reason = "SPREAD_PREFLIGHT_BLOCKS_ALL_SYMBOLS"
        elif last_execution_block and str(last_execution_block.get("reject_reason") or "").startswith("SPREAD"):
            idle_reason = f"LAST_EXEC_BLOCKED_{last_execution_block.get('reject_reason')}"

    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "open_positions_count": open_count,
        "max_open_positions": max_open_positions,
        "slots_available": slots_free,
        "cash_balance": round(float(cash_balance), 4),
        "capital_deployable": deployable,
        "capital_idle_reason": idle_reason,
        "capital_idle_diagnosis": capital_idle_diagnosis,
        "trapped_position_days_max": round(trapped_days, 2),
        "basket_signals": signals,
        "positions": positions_health,
        "signal_freshness": signal_freshness,
        "entry_quality": entry_quality,
        "entry_reject_summary_7d": entry_reject_summary,
        "spread_preflight": spread_preflight,
        "last_execution_block": last_execution_block,
        "last_bar_skip_reason": last_bar_skip_reason,
        "telemetry_only": True,
        "no_auto_sell": True,
        "no_rotation": True,
    }


def normalize_symbol(symbol: str) -> str:
    return _api_to_ccxt(symbol)


def persist_health(payload: dict[str, Any], db_path: str = DATABASE_PATH) -> None:
    try:
        blob = json.dumps(payload, separators=(",", ":"))
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO operational_state(key, value_json, updated_ts)
                VALUES(?, ?, strftime('%s','now'))
                ON CONFLICT(key) DO UPDATE SET
                  value_json=excluded.value_json,
                  updated_ts=excluded.updated_ts
                """,
                (OPERATIONAL_KEY, blob),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("persist day_position_health failed: %s", exc)


def load_health(db_path: str = DATABASE_PATH) -> dict[str, Any] | None:
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT value_json FROM operational_state WHERE key=?",
                (OPERATIONAL_KEY,),
            ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def _observation_emit_allowed(obs_key: str, *, cooldown_sec: int = OBS_EMIT_COOLDOWN_SEC, db_path: str = DATABASE_PATH) -> bool:
    """Rate-limit repeated observation rows (exit monitor runs frequently)."""
    key = f"{OBS_COOLDOWN_PREFIX}{obs_key}"
    now = int(time.time())
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT updated_ts FROM operational_state WHERE key=?",
                (key,),
            ).fetchone()
            if row and row[0]:
                try:
                    if now - int(row[0]) < int(cooldown_sec):
                        return False
                except (TypeError, ValueError):
                    pass
            conn.execute(
                """
                INSERT INTO operational_state(key, value_json, updated_ts)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET updated_ts=excluded.updated_ts
                """,
                (key, "{}", now),
            )
            conn.commit()
        return True
    except Exception:
        return True


def record_capital_idle_observation(
    *,
    reason: str,
    details: dict[str, Any] | None = None,
    db_path: str = DATABASE_PATH,
) -> None:
    """Extend missed-opportunity log for idle-capital / trapped-hold cases."""
    if not _observation_emit_allowed(f"capital_idle:{reason}", db_path=db_path):
        return
    try:
        from backend.services.ai_missed_opportunity_observer import record_missed_opportunity_observation

        record_missed_opportunity_observation(
            block_reason=f"DAY_CAPITAL_IDLE:{reason}",
            attempted_symbol=(details or {}).get("attempted_symbol"),
            active_positions=(details or {}).get("open_positions_count"),
            max_positions=(details or {}).get("max_open_positions"),
            db_path=db_path,
        )
    except Exception as exc:
        logger.debug("record_capital_idle_observation failed: %s", exc)


def update_telemetry(
    engine: Any,
    current_prices: dict[str, float],
    *,
    last_bar_skip_reason: str | None = None,
) -> dict[str, Any]:
    """Called from exit monitor / bar processor — best effort."""
    try:
        from backend.services.portfolio_engine import MAX_OPEN_POSITIONS

        max_pos = int(MAX_OPEN_POSITIONS)
    except Exception:
        max_pos = 4
    payload = build_portfolio_health(
        open_positions=getattr(engine, "open_positions", {}) or {},
        current_prices=current_prices,
        cash_balance=float(getattr(engine, "cash_balance", 0)),
        max_open_positions=max_pos,
        last_bar_skip_reason=last_bar_skip_reason,
    )
    with contextlib.suppress(Exception):
        cap = engine.get_trading_capability_status()
        payload["failsafe_active"] = bool(cap.get("failsafe_active"))
        payload["day_entry_enabled"] = bool(cap.get("day_entry_enabled"))
        payload["no_trade_reason"] = cap.get("no_trade_reason")
        if cap.get("failsafe_active") or not cap.get("day_entry_enabled"):
            reason = str(cap.get("no_trade_reason") or cap.get("kill_switch_reason") or "")
            if reason:
                payload["capital_idle_reason"] = reason
    persist_health(payload, db_path=str(getattr(engine, "db_path", DATABASE_PATH)))
    if payload.get("capital_deployable") and payload.get("trapped_position_days_max", 0) >= 1.0:
        record_capital_idle_observation(
            reason=str(payload.get("capital_idle_reason") or "TRAPPED_WITH_IDLE_CAPITAL"),
            details={
                "open_positions_count": payload.get("open_positions_count"),
                "max_open_positions": payload.get("max_open_positions"),
                "trapped_days": payload.get("trapped_position_days_max"),
            },
        )
    _record_lagging_opportunity_cost(payload, db_path=str(getattr(engine, "db_path", DATABASE_PATH)))
    return payload


def _record_lagging_opportunity_cost(payload: dict[str, Any], *, db_path: str = DATABASE_PATH) -> None:
    """Observation when underwater lagging position coexists with deployable capital."""
    if not payload.get("capital_deployable"):
        return
    lagging = [p for p in (payload.get("positions") or []) if p.get("state") == STATE_LAGGING]
    if not lagging:
        return
    for pos in lagging:
        alt = pos.get("best_alternate_symbol")
        sym = str(pos.get("symbol") or "")
        if not _observation_emit_allowed(f"lagging:{sym}", db_path=db_path):
            continue
        try:
            from backend.services.ai_missed_opportunity_observer import record_missed_opportunity_observation

            record_missed_opportunity_observation(
                block_reason=(f"DAY_OPPORTUNITY_COST:LAGGING_VS_BASKET:{pos.get('symbol')} rank={pos.get('rs_rank')} alt={alt}"),
                attempted_symbol=str(alt or pos.get("symbol") or ""),
                active_positions=payload.get("open_positions_count"),
                max_positions=payload.get("max_open_positions"),
                db_path=db_path,
            )
        except Exception as exc:
            logger.debug("lagging opportunity cost observation failed: %s", exc)
