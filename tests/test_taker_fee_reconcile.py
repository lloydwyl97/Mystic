"""Taker-fee reconciliation: warn when config diverges from the exchange rate.

Read-only safety check. It never changes config and never gates a trade; it
only tells an operator the net-edge gate is calibrated to the wrong fee.
"""

from __future__ import annotations

from backend.config import trading_economics as te


def _stub(value):
    """Return a zero-arg-compatible stand-in for fetch_account_taker_fee."""

    def _inner(timeout: float = 10.0):
        return value

    return _inner


def test_reconcile_matches_configured(monkeypatch):
    monkeypatch.setattr(te, "fetch_account_taker_fee", _stub(te.TAKER_FEE))
    rec = te.reconcile_taker_fee()
    assert rec["reconciled"] is True
    assert rec["diverged"] is False
    assert rec["actual_taker_fee"] == te.TAKER_FEE
    assert abs(rec["delta_bps"]) < 1e-6


def test_reconcile_flags_divergence(monkeypatch):
    # Exchange charges 10 bps but config assumes 2 bps -> must flag.
    monkeypatch.setattr(te, "fetch_account_taker_fee", _stub(0.0010))
    rec = te.reconcile_taker_fee()
    assert rec["diverged"] is True
    assert rec["actual_taker_fee"] == 0.0010
    # delta is actual - configured, reported in bps.
    assert rec["delta_bps"] == round((0.0010 - te.TAKER_FEE) * 10000.0, 3)


def test_reconcile_skips_when_unavailable(monkeypatch):
    monkeypatch.setattr(te, "fetch_account_taker_fee", _stub(None))
    rec = te.reconcile_taker_fee()
    assert rec["reconciled"] is False
    assert rec["diverged"] is False
    assert rec["actual_taker_fee"] is None
    assert rec["delta_bps"] is None


def test_reconcile_respects_tolerance(monkeypatch):
    # Within default 0.5 bps tolerance -> not flagged.
    monkeypatch.setattr(te, "fetch_account_taker_fee", _stub(te.TAKER_FEE + 0.00003))
    assert te.reconcile_taker_fee()["diverged"] is False
    # Just beyond tolerance -> flagged.
    monkeypatch.setattr(te, "fetch_account_taker_fee", _stub(te.TAKER_FEE + 0.00006))
    assert te.reconcile_taker_fee()["diverged"] is True


def test_fetch_returns_none_without_credentials(monkeypatch):
    for var in ("BINANCE_US_API_KEY", "BINANCE_API_KEY", "BINANCE_US_SECRET_KEY", "BINANCE_SECRET"):
        monkeypatch.delenv(var, raising=False)
    assert te.fetch_account_taker_fee() is None
