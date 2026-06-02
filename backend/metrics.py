from __future__ import annotations

from typing import Any

try:
    from prometheus_client import (  # type: ignore[import-not-found]
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
except (ImportError, ModuleNotFoundError, AttributeError):

    class _NoopMetric:  # type: ignore[misc]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def labels(self, *_args: Any, **_kwargs: Any) -> _NoopMetric:
            return self

        def inc(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def observe(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def set(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class Counter(_NoopMetric):  # type: ignore[misc]
        pass

    class Histogram(_NoopMetric):  # type: ignore[misc]
        pass

    class Gauge(_NoopMetric):  # type: ignore[misc]
        pass

    class CollectorRegistry:  # type: ignore[misc]
        pass

    def generate_latest(_: Any = None) -> bytes:  # type: ignore[misc]
        return b""

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

decision_total = Counter("ai_decision_total", "Total AI decisions processed", ["symbol", "result"])
decision_skipped = Counter("ai_decision_skipped_total", "AI decisions skipped by reason", ["symbol", "reason"])
order_type_total = Counter("ai_order_type_total", "Order type chosen for buys", ["symbol", "type"])
exit_triggers_total = Counter("ai_exit_triggers_total", "Exit triggers emitted", ["symbol", "reason"])
decision_latency_seconds = Histogram("ai_decision_latency_seconds", "Per-symbol decision processing latency", ["symbol"])
exit_executed_total = Counter("ai_exit_executed_total", "Exit orders executed", ["symbol", "result", "reason"])
overrides_changes_total = Counter("ai_overrides_changes_total", "Overrides changes applied", ["key", "symbol"])
kill_switch_toggles_total = Counter("ai_overrides_kill_switch_toggles_total", "Kill switch toggled", ["state"])
symbol_kill_toggles_total = Counter(
    "ai_overrides_symbol_kill_toggles_total",
    "Per-symbol kill toggled",
    ["symbol", "state"],
)

limiter_tokens = Gauge("binance_limiter_tokens", "Tokens remaining in the 1m bucket")
limiter_circuit_open = Gauge("binance_circuit_open", "Circuit breaker state (1=open, 0=closed)")
limiter_consume_wait_seconds = Histogram("binance_consume_wait_seconds", "Wait time to acquire tokens", ["path"])
limiter_denied_total = Counter(
    "binance_consume_denied_total",
    "Limiter consume denials/timeouts",
    ["path", "reason"],
)
limiter_consumes_total = Counter("binance_consumes_total", "Limiter successful token consumes", ["path"])

rest_requests_total = Counter("binance_rest_requests_total", "Binance REST requests by status", ["path", "status"])
rest_latency_seconds = Histogram("binance_rest_latency_seconds", "Binance REST request latency", ["path"])
rest_retries_total = Counter("binance_rest_retries_total", "Retries attempted by REST client", ["path"])
rest_errors_total = Counter("binance_rest_errors_total", "REST client errors by type", ["path", "type"])

ws_reconnects_total = Counter("ws_reconnects_total", "WebSocket reconnects")
ws_disconnects_total = Counter("ws_disconnects_total", "WebSocket disconnects", ["reason"])
ws_messages_total = Counter("ws_messages_total", "WebSocket messages received", ["type", "symbol"])
ws_inter_message_seconds = Histogram("ws_inter_message_seconds", "Time between WS messages per symbol", ["symbol"])
ws_last_tick_ts = Gauge("ws_last_tick_ts", "Unix epoch of last tick per symbol", ["symbol"])

order_ack_latency_seconds = Histogram("order_ack_latency_seconds", "Order send to ack latency", ["symbol", "side", "type"])
order_slippage_bp = Histogram("order_slippage_bp", "Order slippage in basis points", ["symbol", "side", "type"])

risk_drawdown_today = Gauge("risk_drawdown_today", "Current drawdown today (USD)")
risk_pnl_today = Gauge("risk_pnl_today", "Current PnL today (USD)")
cooldown_daily_count = Gauge("cooldown_daily_count", "Autobuy cooldown daily trade count")
cooldown_daily_cap = Gauge("cooldown_daily_cap", "Autobuy cooldown daily trade cap")
kill_switch_state = Gauge("kill_switch_state", "Kill switch state (1=on, 0=off)")
binance_last_1003_epoch = Gauge("binance_last_1003_epoch", "Epoch of last -1003/418 event")
binance_circuit_ttl_seconds = Gauge("binance_circuit_ttl_seconds", "Remaining TTL on circuit breaker")

canary_steps_total = Counter("canary_steps_total", "Canary threshold steps", ["symbol", "direction"])
canary_promotions_total = Counter("canary_promotions_total", "Canary promotions applied", ["symbol"])
canary_reverts_total = Counter("canary_reverts_total", "Canary demotions applied", ["symbol"])

trade_results_total = Counter(
    "trade_results_total",
    "Trade outcomes (win/lose)",
    ["symbol", "mode", "result", "side"],
)
paper_slippage_bp = Histogram("paper_slippage_bp", "Paper trade slippage in bps", ["symbol", "side"])
paper_fill_ratio = Histogram("paper_fill_ratio", "Paper trade fill ratio (0-1)", ["symbol", "side"])

calibration_ece = Gauge("calibration_ece", "ECE per symbol", ["symbol"])
calibration_brier = Gauge("calibration_brier", "Brier score per symbol", ["symbol"])
psi_current = Gauge("psi_current", "Population Stability Index per feature", ["feature"])

price_age_seconds = Gauge("price_age_seconds", "Age of cached price per symbol", ["symbol"])
feature_age_seconds = Gauge("feature_age_seconds", "Age of cached features per symbol", ["symbol"])
shadow_match_ratio = Gauge("shadow_match_ratio", "Ratio of live vs shadow decision matches", ["symbol"])
kill_symbol_state = Gauge("kill_symbol_state", "Per-symbol kill state (1=on, 0=off)", ["symbol"])


class Metrics:
    def __init__(self) -> None:
        try:
            self.registry: CollectorRegistry | None = CollectorRegistry()  # type: ignore[assignment]
            self.feature_ingest = Counter(
                "feature_ingest_total",
                "Number of feature rows ingested",
                registry=self.registry,
            )
            self.trainer_runs = Counter(
                "trainer_runs_total",
                "Number of online trainer runs",
                registry=self.registry,
            )
            self.stream_reconnects = Counter(
                "stream_reconnects_total",
                "Number of user-data stream reconnects",
                registry=self.registry,
            )
            self.shadow_pnl = Gauge(
                "shadow_pnl",
                "Latest shadow PnL value",
                ["symbol"],
                registry=self.registry,
            )
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            self.registry = None
            self.feature_ingest = _Noop()  # type: ignore[name-defined]
            self.trainer_runs = _Noop()  # type: ignore[name-defined]
            self.stream_reconnects = _Noop()  # type: ignore[name-defined]
            self.shadow_pnl = _Noop()  # type: ignore[name-defined]


class _Noop:  # type: ignore[override]
    def labels(self, *_args: Any, **_kwargs: Any) -> _Noop:
        return self

    def inc(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set(self, *_args: Any, **_kwargs: Any) -> None:
        return None


metrics = Metrics()

__all__ = [
    "CONTENT_TYPE_LATEST",
    "CollectorRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "Metrics",
    "binance_circuit_ttl_seconds",
    "binance_last_1003_epoch",
    "calibration_brier",
    "calibration_ece",
    "canary_promotions_total",
    "canary_reverts_total",
    "canary_steps_total",
    "cooldown_daily_cap",
    "cooldown_daily_count",
    "decision_latency_seconds",
    "decision_skipped",
    "decision_total",
    "exit_executed_total",
    "exit_triggers_total",
    "feature_age_seconds",
    "generate_latest",
    "kill_switch_state",
    "kill_switch_toggles_total",
    "kill_symbol_state",
    "limiter_circuit_open",
    "limiter_consume_wait_seconds",
    "limiter_consumes_total",
    "limiter_denied_total",
    "limiter_tokens",
    "metrics",
    "order_ack_latency_seconds",
    "order_slippage_bp",
    "order_type_total",
    "overrides_changes_total",
    "paper_fill_ratio",
    "paper_slippage_bp",
    "price_age_seconds",
    "psi_current",
    "rest_errors_total",
    "rest_latency_seconds",
    "rest_requests_total",
    "rest_retries_total",
    "risk_drawdown_today",
    "risk_pnl_today",
    "shadow_match_ratio",
    "symbol_kill_toggles_total",
    "trade_results_total",
    "ws_disconnects_total",
    "ws_inter_message_seconds",
    "ws_last_tick_ts",
    "ws_messages_total",
    "ws_reconnects_total",
]
