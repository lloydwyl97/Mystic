"""Binance.US scalp paper engine — paper fills only, no live orders."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone

import redis

from backend.services.binance_scalp.calibration_profiles import economics_for_config
from backend.services.binance_scalp.exit_manager import (
    DECISION_SELL,
    PositionTrack,
    evaluate_exit,
    track_from_row,
)
from backend.services.binance_scalp.config import ScalpConfig, get_scalp_config
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter
from backend.services.binance_scalp.strategies.kline_cache import KlineCache
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import MarketSnapshot, ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.protected_preflight import (
    FEE_MODEL_UNVERIFIED,
    NET_PROFIT_TARGET_NOT_MET,
    SCALP_PAPER_DISABLED,
    run_scalp_preflight,
)
from backend.services.binance_scalp.redis_keys import (
    assert_key_allowed,
    market_key,
    position_key,
    signal_key,
)
from backend.services.binance_scalp.scalp_control import (
    clear_entry_armed,
    is_entry_armed,
    set_entry_armed,
)
from backend.services.binance_scalp.schema import init_scalp_schema

logger = logging.getLogger(__name__)
WOULD_ENTER_NOT_ARMED = "WOULD_ENTER_NOT_ARMED"


class BinanceScalpPaperEngine:
    """Paper-only scalp loop — isolated from Mystic DAY portfolio_engine."""

    def __init__(
        self,
        config: ScalpConfig | None = None,
        econ: ScalpEconomics | None = None,
    ) -> None:
        self.config = config or get_scalp_config()
        self.config.assert_no_live_trading()
        if self.config.allow_repair_add:
            raise RuntimeError("SCALP_ALLOW_REPAIR_ADD must remain false")
        self.econ = econ or economics_for_config(self.config)
        self.reader = ScalpMarketReader(self.config)
        self._momentum = MomentumTracker()
        self._klines = KlineCache()
        self._router = ScalpStrategyRouter(
            config=self.config,
            econ=self.econ,
            reader=self.reader,
            momentum=self._momentum,
            klines=self._klines,
        )
        self._redis = redis.from_url(self.config.redis_url, decode_responses=True)
        init_scalp_schema(self.config.database_path)
        set_entry_armed(
            self._redis, prefix=self.config.redis_key_prefix, armed=False
        )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.config.database_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _now(self) -> tuple[str, float]:
        dt = datetime.now(timezone.utc)
        return dt.isoformat(), dt.timestamp()

    def _ledger(self, conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM scalp_paper_ledger WHERE id = 1").fetchone()
        if row is None:
            raise RuntimeError("scalp_paper_ledger missing")
        return row

    def _open_positions(self, conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return list(
            conn.execute("SELECT * FROM scalp_paper_positions WHERE status = 'OPEN'")
        )

    def _position_strategy_id(self, row: sqlite3.Row) -> str:
        try:
            sid = row["strategy_id"]
            if sid:
                return str(sid)
        except (KeyError, IndexError, TypeError):
            pass
        return self.config.strategy_id

    def _record_reject(
        self,
        conn: sqlite3.Connection,
        symbol: str,
        side: str,
        reason: str,
        detail: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO scalp_rejects (symbol, exchange, strategy_id, side, reason, detail)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                self.config.exchange,
                self.config.strategy_id,
                side,
                reason,
                detail,
            ),
        )

    def _write_signal(self, symbol_bus: str, payload: dict) -> None:
        key = signal_key(self.config.redis_key_prefix, symbol_bus)
        assert_key_allowed(key, prefix=self.config.redis_key_prefix)
        self._redis.setex(key, 120, json.dumps(payload))

    def _write_market_cache(self, snap: MarketSnapshot) -> None:
        key = market_key(self.config.redis_key_prefix, snap.symbol_bus)
        assert_key_allowed(key, prefix=self.config.redis_key_prefix)
        self._redis.setex(
            key,
            120,
            json.dumps(
                {
                    "symbol": snap.symbol_bus,
                    "best_bid": snap.best_bid,
                    "best_ask": snap.best_ask,
                    "mid": snap.mid,
                    "spread_pct": snap.spread_pct,
                    "order_book_imbalance": snap.order_book_imbalance,
                    "source": "scalp_paper_engine",
                    "strategy_id": self.config.strategy_id,
                }
            ),
        )

    def _write_position_cache(
        self, symbol_bus: str, payload: dict | None
    ) -> None:
        key = position_key(self.config.redis_key_prefix, symbol_bus)
        assert_key_allowed(key, prefix=self.config.redis_key_prefix)
        if payload is None:
            self._redis.delete(key)
        else:
            self._redis.setex(key, 300, json.dumps(payload))

    def _audit(
        self,
        conn: sqlite3.Connection,
        *,
        trade_id: str,
        action: str,
        symbol: str,
        qty: float | None,
        price: float | None,
        pre: dict,
        post: dict,
        reason: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO scalp_trade_audit
            (trade_id, action, symbol, exchange, strategy_id, qty, price,
             pre_ledger_json, post_ledger_json, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                action,
                symbol,
                self.config.exchange,
                self.config.strategy_id,
                qty,
                price,
                json.dumps(pre),
                json.dumps(post),
                reason,
            ),
        )

    def _record_momentum(self, snap: MarketSnapshot) -> None:
        _, epoch = self._now()
        self._momentum.record(snap.symbol_bus, epoch, snap.best_bid, snap.mid)

    def _entry_armed_ok(self) -> bool:
        """Calibration mode auto-arms paper entries; normal mode requires operator arm."""
        if self.config.calibration_mode:
            return True
        return is_entry_armed(self._redis, prefix=self.config.redis_key_prefix)

    def _entry_candidates(
        self, conn: sqlite3.Connection
    ) -> list[tuple[str, MarketSnapshot, object]]:
        if not self.config.scalp_paper_enabled:
            return []

        _, epoch = self._now()
        notional = min(self.config.max_notional_paper, float(self._ledger(conn)["cash_balance"]))
        ranked = self._router.evaluate_all(epoch=epoch, notional_usd=notional)

        for row in ranked:
            sym = row["symbol"]
            snap = row["snap"]
            self._write_market_cache(snap)
            self._record_momentum(snap)
            sig = row["signal"]
            self._write_signal(
                sym,
                {
                    "symbol": sym,
                    "strategy_id": self.config.strategy_id,
                    "exchange": self.config.exchange,
                    "side": "BUY",
                    "setup": sig.as_dict(),
                    "all_setups": row["all_signals"],
                    "paper_only": True,
                    "live_blocked": True,
                },
            )

        if not ranked:
            for sym in self.config.products:
                snap = self.reader.read(sym)
                if snap is None:
                    continue
                _, all_sigs = self._router.evaluate_symbol(sym, epoch=epoch, notional_usd=notional, snap=snap)
                best_reject = next((s for s in all_sigs if not s.passed), None)
                if best_reject:
                    self._record_reject(
                        conn, sym, "BUY", best_reject.reject_reason or "NO_VALID_SETUP",
                        json.dumps({"signals": [s.as_dict() for s in all_sigs]}),
                    )
            return []

        best = ranked[0]
        sym, snap, sig = best["symbol"], best["snap"], best["signal"]
        self._last_ranking_meta = {
            "selection_reason": f"{sig.setup_name} score={sig.score:.2f} {sig.entry_reason}",
            "selected_symbol": sym,
            "ranking": [r["signal"].as_dict() for r in ranked],
        }
        logger.info("SCALP_STRATEGY_PICK %s", self._last_ranking_meta["selection_reason"])

        if not self._entry_armed_ok():
            self._record_reject(
                conn, sym, "BUY", WOULD_ENTER_NOT_ARMED,
                json.dumps({"setup": sig.as_dict(), "entry_armed": False}),
            )
            return []

        return [(sym, snap, sig)]

    def _try_entry(self, conn: sqlite3.Connection) -> None:
        open_count = conn.execute(
            "SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchone()[0]
        if open_count >= self.config.max_open_positions:
            return

        candidates = self._entry_candidates(conn)
        if not candidates:
            return

        sym, snap, sig = candidates[0]
        ledger = self._ledger(conn)
        notional = min(self.config.max_notional_paper, float(ledger["cash_balance"]))
        if notional < 1.0:
            self._record_reject(
                conn, sym, "BUY", "INSUFFICIENT_CASH", f"cash={ledger['cash_balance']}"
            )
            return

        limit_buy = sig.limit_buy_price
        qty = notional / limit_buy
        fee = notional * self.econ.taker_fee_pct
        slip = notional * self.econ.slippage_buffer_pct
        trade_id = f"scalp_paper_{sym}_{int(time.time() * 1000)}"
        ts, epoch = self._now()
        pre = dict(ledger)

        ranking_meta = getattr(self, "_last_ranking_meta", {}) or {}
        from backend.services.day_trade_thesis import scalp_strategy_to_thesis

        thesis_fields = scalp_strategy_to_thesis(sig.setup_name, sig.setup_context or {})
        entry_diag = {
            "setup_name": sig.setup_name,
            "setup_context": sig.setup_context,
            "setup_signal": sig.as_dict(),
            "selected_symbol": sym,
            "symbol_ranking": ranking_meta,
            "review_lows": [],
            "session_low_bid": limit_buy,
            **thesis_fields,
        }

        conn.execute(
            """
            INSERT INTO scalp_paper_trades
            (trade_id, symbol, exchange, strategy_id, side, quantity, price, notional,
             fee_usd, slippage_usd, diagnostics_json)
            VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                sym,
                self.config.exchange,
                sig.setup_name,
                qty,
                limit_buy,
                notional,
                fee,
                slip,
                json.dumps({"setup": sig.as_dict(), "paper_limit": True, "selected_symbol": sym}),
            ),
        )
        conn.execute(
            """
            INSERT INTO scalp_paper_positions
            (symbol, exchange, strategy_id, quantity, entry_price, entry_time,
             entry_time_epoch, trade_id, paper_order_id, status, state,
             max_favorable_pct, max_adverse_pct, stale_review_count,
             session_low_bid, diagnostics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', 'OPEN', 0, 0, 0, ?, ?)
            """,
            (
                sym,
                self.config.exchange,
                sig.setup_name,
                qty,
                limit_buy,
                ts,
                epoch,
                trade_id,
                f"paper_order_{trade_id}",
                limit_buy,
                json.dumps(entry_diag),
            ),
        )
        new_cash = float(ledger["cash_balance"]) - notional - fee - slip
        pos_val = qty * limit_buy
        conn.execute(
            """
            UPDATE scalp_paper_ledger SET
              cash_balance = ?,
              positions_value = positions_value + ?,
              total_equity = ? + positions_value + ?,
              updated_at = datetime('now')
            WHERE id = 1
            """,
            (new_cash, pos_val, new_cash, pos_val),
        )
        post = dict(self._ledger(conn))
        self._audit(
            conn,
            trade_id=trade_id,
            action="BUY",
            symbol=sym,
            qty=qty,
            price=limit_buy,
            pre=pre,
            post=post,
            reason=f"SCALP_PAPER_ENTRY_{sig.setup_name}",
        )
        self._write_position_cache(
            sym,
            {
                "symbol": sym,
                "quantity": qty,
                "entry_price": limit_buy,
                "setup_name": sig.setup_name,
                "trade_id": trade_id,
                "strategy_id": sig.setup_name,
                "paper_only": True,
            },
        )
        logger.info("SCALP_PAPER_BUY %s setup=%s qty=%.8f price=%.4f", sym, sig.setup_name, qty, limit_buy)
        set_entry_armed(self._redis, prefix=self.config.redis_key_prefix, armed=False)

    def shutdown(self) -> None:
        clear_entry_armed(self._redis, prefix=self.config.redis_key_prefix)

    def _record_position_review(
        self,
        conn: sqlite3.Connection,
        *,
        trade_id: str,
        sym: str,
        review_diag: dict,
    ) -> None:
        d = review_diag
        conn.execute(
            """
            INSERT INTO scalp_position_reviews
            (trade_id, symbol, hold_seconds, current_bid, entry_price,
             executable_net_pct, max_favorable_pct, max_adverse_pct,
             recovery_from_low_pct, bid_change_15s, bid_change_30s, bid_change_60s,
             higher_lows, spread_pct, decision, state, reason, diagnostics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                sym,
                float(d.get("hold_seconds") or 0),
                float(d.get("current_bid") or 0),
                float(d.get("entry_price") or 0),
                float(d.get("executable_net_pct") or 0),
                float(d.get("max_favorable_pct") or 0),
                float(d.get("max_adverse_pct") or 0),
                float(d.get("recovery_from_low_pct") or 0),
                float(d.get("bid_change_15s") or 0),
                float(d.get("bid_change_30s") or 0),
                float(d.get("bid_change_60s") or 0),
                1 if d.get("higher_lows") else 0,
                float(d.get("spread_pct") or 0),
                str(d.get("decision") or ""),
                str(d.get("state") or ""),
                str(d.get("reason") or ""),
                json.dumps(d),
            ),
        )

    def _persist_position_track(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        track: PositionTrack,
        *,
        ts: str,
        reason: str,
        review_lows: tuple[float, ...],
    ) -> None:
        raw_diag = row["diagnostics_json"]
        diag = json.loads(raw_diag) if raw_diag else {}
        diag["review_lows"] = list(review_lows)
        diag["session_low_bid"] = track.session_low_bid
        conn.execute(
            """
            UPDATE scalp_paper_positions SET
              state = ?,
              max_favorable_pct = ?,
              max_adverse_pct = ?,
              stale_review_count = ?,
              session_low_bid = ?,
              last_review_ts = ?,
              last_state_reason = ?,
              diagnostics_json = ?
            WHERE id = ?
            """,
            (
                track.state,
                track.max_favorable_pct,
                track.max_adverse_pct,
                track.stale_review_count,
                track.session_low_bid,
                ts,
                reason,
                json.dumps(diag),
                row["id"],
            ),
        )

    def _execute_sell(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        sym: str,
        entry: float,
        qty: float,
        exit_price: float,
        net_pct: float,
        net_usd: float,
        reason: str,
        pf_dict: dict,
        exit_gate: dict,
    ) -> None:
        trade_id = str(row["trade_id"])
        sell_tid = f"{trade_id}_SELL"
        strategy_id = self._position_strategy_id(row)
        notional = qty * exit_price
        fee = notional * self.econ.taker_fee_pct
        slip = notional * self.econ.slippage_buffer_pct
        ledger = self._ledger(conn)
        pre = dict(ledger)

        conn.execute(
            """
            INSERT INTO scalp_paper_trades
            (trade_id, symbol, exchange, strategy_id, side, quantity, price, notional,
             fee_usd, slippage_usd, pnl_usd, pnl_pct, entry_price, exit_reason, diagnostics_json)
            VALUES (?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sell_tid,
                sym,
                self.config.exchange,
                strategy_id,
                qty,
                exit_price,
                notional,
                fee,
                slip,
                net_usd,
                net_pct,
                entry,
                reason,
                json.dumps(
                    {
                        "preflight": pf_dict,
                        "paper_limit": True,
                        "exit_gate": exit_gate,
                    }
                ),
            ),
        )
        conn.execute(
            "UPDATE scalp_paper_positions SET status='CLOSED', state=? WHERE id=?",
            (reason, row["id"]),
        )
        pos_cost = entry * qty
        new_cash = float(ledger["cash_balance"]) + notional - fee - slip
        conn.execute(
            """
            UPDATE scalp_paper_ledger SET
              cash_balance = ?,
              positions_value = MAX(0, positions_value - ?),
              realized_pnl = realized_pnl + ?,
              total_equity = cash_balance + positions_value,
              updated_at = datetime('now')
            WHERE id = 1
            """,
            (new_cash, pos_cost, net_usd),
        )
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        win = 1 if net_usd > 0 else 0
        loss = 0 if net_usd > 0 else 1
        conn.execute(
            """
            INSERT INTO scalp_scoreboard_daily (day, trades, wins, losses, net_pnl)
            VALUES (?, 1, ?, ?, ?)
            ON CONFLICT(day) DO UPDATE SET
              trades = trades + 1,
              wins = wins + ?,
              losses = losses + ?,
              net_pnl = net_pnl + ?,
              updated_at = datetime('now')
            """,
            (day, win, loss, net_usd, win, loss, net_usd),
        )
        post = dict(self._ledger(conn))
        self._audit(
            conn,
            trade_id=sell_tid,
            action="SELL",
            symbol=sym,
            qty=qty,
            price=exit_price,
            pre=pre,
            post=post,
            reason=reason,
        )
        self._write_position_cache(sym, None)
        logger.info("SCALP_PAPER_SELL %s pnl=%.4f reason=%s", sym, net_usd, reason)

    def _try_exit(self, conn: sqlite3.Connection, row: sqlite3.Row) -> None:
        sym = str(row["symbol"])
        snap = self.reader.read(sym)
        if snap is None:
            return
        self._write_market_cache(snap)
        self._record_momentum(snap)

        entry = float(row["entry_price"])
        qty = float(row["quantity"])
        ts, now_epoch = self._now()
        age = now_epoch - float(row["entry_time_epoch"])
        trade_id = str(row["trade_id"])

        entry_buy_impact = 0.0
        target_pct = self.econ.net_profit_target_pct
        pos_diag: dict = {}
        raw_diag = row["diagnostics_json"]
        if raw_diag:
            try:
                pos_diag = json.loads(raw_diag)
                setup_sig = pos_diag.get("setup_signal") or {}
                entry_buy_impact = float(setup_sig.get("impact_pct") or 0.0)
                target_pct = float(setup_sig.get("required_target_pct") or target_pct)
            except (json.JSONDecodeError, TypeError, ValueError):
                entry_buy_impact = 0.0

        pf = run_scalp_preflight(
            snap,
            self.econ,
            self.config,
            side="SELL",
            entry_price=entry,
            entry_buy_impact_pct=entry_buy_impact,
            quantity=qty,
            check_paper_enabled=True,
        )
        self._write_signal(
            sym,
            {
                "symbol": sym,
                "strategy_id": self._position_strategy_id(row),
                "side": "SELL",
                "preflight": pf.as_dict(),
                "paper_only": True,
            },
        )

        exit_price = pf.expected_avg_fill if pf.expected_avg_fill > 0 else pf.limit_sell_price
        net_pct = pf.expected_net_edge_pct
        net_usd = (exit_price - entry) * qty - (
            exit_price * qty * self.econ.taker_fee_pct
            + exit_price * qty * self.econ.slippage_buffer_pct
            + entry * qty * self.econ.taker_fee_pct
        )
        profit_hit = net_pct >= target_pct
        exit_spread_ok = pf.reject_reason != "SPREAD_TOO_WIDE"

        _, epoch = self._now()
        mom = self._momentum.diagnostics(sym, epoch, snap.best_bid, snap.mid)
        track = track_from_row(row, pos_diag)
        review_interval = int(os.getenv("SCALP_REVIEW_INTERVAL_SEC", "30"))
        last_review_epoch = 0.0
        try:
            lrt = row["last_review_ts"]
            if lrt:
                last_review_epoch = datetime.fromisoformat(str(lrt).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            last_review_epoch = 0.0
        in_review_phase = age >= self.econ.stale_scalp_timeout_sec
        perform_review = in_review_phase and (
            track.stale_review_count == 0
            or (epoch - last_review_epoch) >= review_interval
        )
        review = evaluate_exit(
            track=track,
            snap=snap,
            mom=mom,
            econ=self.econ,
            config=self.config,
            trade_id=trade_id,
            hold_sec=age,
            executable_net_pct=net_pct,
            profit_hit=profit_hit,
            exit_spread_ok=exit_spread_ok,
            perform_review=perform_review,
        )

        if perform_review or review.decision == DECISION_SELL:
            self._record_position_review(conn, trade_id=trade_id, sym=sym, review_diag=review.diagnostics)

        self._persist_position_track(
            conn,
            row,
            review.updated_track,
            ts=ts,
            reason=review.reason,
            review_lows=review.updated_track.review_lows,
        )

        exit_gate = {
            "current_bid": snap.best_bid,
            "expected_sell_fill": exit_price,
            "net_pct": net_pct,
            "target_pct": self.econ.net_profit_target_pct,
            "sell_preflight_pass": profit_hit,
            "reject_reason": pf.reject_reason or None,
            "spread_pct": snap.spread_pct,
            "entry_buy_impact_pct": entry_buy_impact,
            "exit_sell_impact_pct": pf.sell_impact_pct,
            "age_sec": age,
            "exit_review": review.diagnostics,
            "exit_state": review.state,
        }

        if review.decision != DECISION_SELL or not review.exit_reason:
            return

        self._execute_sell(
            conn,
            row,
            sym=sym,
            entry=entry,
            qty=qty,
            exit_price=exit_price,
            net_pct=net_pct,
            net_usd=net_usd,
            reason=review.exit_reason,
            pf_dict=pf.as_dict(),
            exit_gate=exit_gate,
        )

    def tick(self) -> None:
        self.config.assert_no_live_trading()
        with self._conn() as conn:
            if not self.config.scalp_paper_enabled:
                for sym in self.config.products:
                    self._write_signal(
                        sym,
                        {
                            "symbol": sym,
                            "strategy_id": self.config.strategy_id,
                            "side": "SCAN",
                            "paper_enabled": False,
                            "reject_reason": SCALP_PAPER_DISABLED,
                        },
                    )
                conn.commit()
                return

            if not self.econ.is_fee_model_verified():
                for sym in self.config.products:
                    self._record_reject(
                        conn, sym, "BUY", FEE_MODEL_UNVERIFIED, "fee model not verified"
                    )
                conn.commit()
                return

            open_rows = self._open_positions(conn)
            if open_rows:
                for row in open_rows:
                    self._try_exit(conn, row)
            elif conn.execute(
                "SELECT COUNT(*) FROM scalp_paper_positions WHERE status='OPEN'"
            ).fetchone()[0] < self.config.max_open_positions:
                self._try_entry(conn)
            conn.commit()

    def run_loop(self, interval_sec: float = 5.0) -> None:
        if not self.config.scalp_paper_enabled:
            logger.error("Scalp paper blocked: SCALP_PAPER_ENABLED=false")
            raise SystemExit(2)
        if not self.econ.is_fee_model_verified():
            logger.error("Scalp paper blocked: SCALP_FEE_MODEL_VERIFIED=false")
            raise SystemExit(3)
        self.config.assert_no_live_trading()
        armed = is_entry_armed(self._redis, prefix=self.config.redis_key_prefix)
        logger.info(
            "Binance scalp paper loop products=%s max_open=%s interval=%ss "
            "paper_only=True live_blocked=True calibration=%s profile=%s entry_armed=%s",
            self.config.products,
            self.config.max_open_positions,
            interval_sec,
            self.config.calibration_mode,
            self.config.calibration_profile if self.config.calibration_mode else "strict",
            armed if not self.config.calibration_mode else "auto",
        )
        try:
            while True:
                try:
                    self.tick()
                except Exception as exc:
                    logger.exception("scalp paper tick error: %s", exc)
                time.sleep(interval_sec)
        finally:
            self.shutdown()
