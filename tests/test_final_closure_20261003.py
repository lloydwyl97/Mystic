"""Final closure: same-asset ownership sum, secret redaction, SCALP sleeve sharing."""

from __future__ import annotations

import gzip
import logging
import sqlite3

import pytest

from backend.services.adaptive_learning import SIZE_BOUNDS
from backend.services.balance_sync_ownership import load_asset_ownership, quantity_drift
from backend.services.external_capital_flows import DEPOSIT, WITHDRAWAL, parse_endpoint, sync_flows
from backend.services.live_account_basis import TRAILING_BUY_ANCHOR_EQUITY
from backend.services.two_engine_capital import CapitalSnapshot, EngineBudget, scalp_order_notional
from backend.utils.historical_log_sanitize import sanitize_log_file, sanitize_log_text
from backend.utils.secret_log_filter import install_secret_redacting_filter, redact_secrets

FAKE_TELEGRAM = "123456789:AAFakeTokenValueForTestsOnlyXXXX"
FAKE_NEWS = "newsapifake0123456789abcdef"
FAKE_BINANCE_KEY = "binanceapifake0123456789abcd"
FAKE_SIGNATURE = "fakesignature0123456789abcdef"
FAKE_BEARER = "fakebearer0123456789abcdef"


def _snap(target: float, remaining: float) -> CapitalSnapshot:
    snap = CapitalSnapshot(equity=target * 2, free_cash=target * 2)
    snap.scalp = EngineBudget(engine_id="SCALP_V2", target_capital=target, remaining_budget=remaining)
    return snap


def test_same_asset_components_sum_and_keep_their_labels(tmp_path):
    db = sqlite3.connect(tmp_path / "own.db")
    db.execute(
        """
        CREATE TABLE portfolio_engine_positions (
            engine_id TEXT, symbol TEXT, quantity REAL, status TEXT,
            PRIMARY KEY (engine_id, symbol)
        )
        """
    )
    db.executemany(
        "INSERT INTO portfolio_engine_positions VALUES (?,?,?,?)",
        [
            ("DAY_V2", "XRP/USDT", 151.16976, "ACTIVE"),
            ("SCALP_V2", "XRP/USDT", 0.0933, "DUST_PENDING"),
            ("SCALP_V2", "BTCUSDT", 0.01, "ACTIVE"),
        ],
    )
    db.execute("CREATE TABLE engine_strategy_dust (engine_id TEXT, symbol TEXT, quantity REAL, status TEXT)")
    db.execute("INSERT INTO engine_strategy_dust VALUES ('SCALP_V2', 'ETH/USDT', 0.004, 'HELD')")
    db.execute("INSERT INTO engine_strategy_dust VALUES ('DAY_V2', 'ETH/USDT', 0.001, 'RETIRED')")
    db.execute("CREATE TABLE protected_external_inventory (symbol TEXT, quantity REAL)")
    db.execute("INSERT INTO protected_external_inventory VALUES ('SOL/USDT', 0.02)")
    owned = load_asset_ownership(db)
    assert owned["XRP"].total == pytest.approx(151.26306)
    assert ("DAY_V2", "ACTIVE", 151.16976) in owned["XRP"].components
    assert ("SCALP_V2", "DUST_PENDING", 0.0933) in owned["XRP"].components
    assert owned["BTC"].total == pytest.approx(0.01)
    assert owned["ETH"].total == pytest.approx(0.004)
    assert ("DAY_V2", "HELD_DUST", 0.001) not in owned["ETH"].components
    assert owned["SOL"].total == pytest.approx(0.02)
    assert owned["SOL"].components == [("PROTECTED", "PROTECTED", 0.02)]


def test_aggregated_xrp_matches_exchange_and_a_real_gap_still_warns():
    exchange = 151.16976 + 0.0933
    owned = 151.16976 + 0.0933
    overwritten = 0.0933
    assert quantity_drift(exchange, overwritten)
    assert not quantity_drift(exchange, owned)
    assert quantity_drift(exchange + 0.02, owned)
    assert not quantity_drift(exchange + 0.01, owned)


def test_four_scalp_slots_share_one_sleeve_at_each_multiplier():
    lo, hi = SIZE_BOUNDS["SCALP_V2"]
    assert lo == pytest.approx(0.50)
    assert hi == pytest.approx(1.25)
    slots = 4
    for equity in (800, 1500, 3000, 10000):
        sleeve = equity * 0.5
        for mult in (lo, 0.75, 1.00, hi):
            remaining = sleeve
            taken = []
            for _slot in range(slots):
                order = scalp_order_notional(_snap(sleeve, remaining), slots=slots, size_mult=mult)
                assert 0.0 <= order <= remaining + 1e-9
                assert order == pytest.approx(min(sleeve / slots * mult, remaining))
                taken.append(order)
                remaining -= order
            assert sum(taken) <= sleeve + 1e-6
            if mult > 1.0:
                assert slots * (sleeve / slots) * mult > sleeve


def test_emergency_cap_is_off_unless_configured():
    sleeve = 1500 * 0.5
    assert scalp_order_notional(_snap(sleeve, sleeve), slots=4, emergency_max_notional=0.0) > 50.0
    capped = scalp_order_notional(_snap(sleeve, sleeve), slots=4, emergency_max_notional=50.0)
    assert capped == pytest.approx(50.0)


def test_capital_flow_applied_once_and_conversion_is_not_a_deposit(tmp_path):
    db = str(tmp_path / "cap.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE portfolio_engine_ledger (id INTEGER PRIMARY KEY, principal REAL, realized_pnl REAL)")
        conn.execute("INSERT INTO portfolio_engine_ledger VALUES (1, ?, 0)", (float(TRAILING_BUY_ANCHOR_EQUITY),))
    fiat = {
        "assetLogRecordList": [
            {"orderId": "dep-1", "orderStatus": "Successful", "fiatCurrency": "USD", "amount": "100", "createTime": 1791000000000},
        ]
    }
    withdraw = {
        "assetLogRecordList": [
            {"orderId": "wd-1", "orderStatus": "Successful", "fiatCurrency": "USD", "amount": "40", "transactionFee": "1", "createTime": 1791000001000},
        ]
    }
    conversion_like = [
        {"coin": "USDT", "amount": "100", "status": 0, "txId": "", "insertTime": 1791000002000},
        {"coin": "USDT", "amount": "100", "status": 1, "txId": "", "insertTime": 1791000002000},
    ]
    flows = parse_endpoint("fiat", DEPOSIT, fiat) + parse_endpoint("fiat", WITHDRAWAL, withdraw) + parse_endpoint("crypto", DEPOSIT, conversion_like)
    assert [f.flow_id for f in flows] == ["fiat:DEPOSIT:dep-1", "fiat:WITHDRAWAL:wd-1"]
    _, base = sync_flows(db, flows)
    expected = float(TRAILING_BUY_ANCHOR_EQUITY) + 100 - 41
    assert base == pytest.approx(expected)
    _, again = sync_flows(db, flows)
    assert again == pytest.approx(expected)
    with sqlite3.connect(db) as conn:
        pnl = conn.execute("SELECT realized_pnl FROM portfolio_engine_ledger WHERE id=1").fetchone()[0]
        n = conn.execute("SELECT COUNT(*) FROM external_capital_flows").fetchone()[0]
    assert pnl == 0
    assert n == 2


class _URL:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


@pytest.mark.parametrize(
    "text",
    [
        f"GET https://api.telegram.org/bot{FAKE_TELEGRAM}/getUpdates",
        f"GET https://newsapi.org/v2/everything?q=btc&apiKey={FAKE_NEWS}",
        f"GET https://api.binance.us/api/v3/account?timestamp=1&signature={FAKE_SIGNATURE}",
        f"X-MBX-APIKEY: {FAKE_BINANCE_KEY}",
        f"Authorization: Bearer {FAKE_BEARER}",
        f"Bearer {FAKE_BEARER}",
        f"https://example.test/path?secret={FAKE_BEARER}&other=1",
    ],
)
def test_redaction_covers_synthetic_secrets(text):
    out = redact_secrets(text)
    for secret in (FAKE_TELEGRAM, FAKE_NEWS, FAKE_BINANCE_KEY, FAKE_SIGNATURE, FAKE_BEARER):
        assert secret not in out
    assert "***" in out


def test_redaction_of_request_objects_and_headers(caplog):
    install_secret_redacting_filter()
    lg = logging.getLogger("httpx._client.closure_test")
    url = _URL(f"https://api.binance.us/api/v3/order?signature={FAKE_SIGNATURE}&timestamp=1")
    with caplog.at_level(logging.INFO):
        lg.info("HTTP Request: %s %s %s", "GET", url, f"Authorization: Bearer {FAKE_BEARER}")
    assert FAKE_SIGNATURE not in caplog.text
    assert FAKE_BEARER not in caplog.text
    assert "signature=***" in caplog.text


def test_historical_sanitizer_keeps_context_and_verifies_gzip(tmp_path):
    line = f"2026-10-03 01:41:03,364 INFO HTTP Request: GET https://api.telegram.org/bot{FAKE_TELEGRAM}/getUpdates"
    plain = tmp_path / "mystic.log"
    plain.write_text(line + "\n", encoding="utf-8")
    result = sanitize_log_file(plain)
    assert result["status"] == "sanitized"
    assert result["before"] == 1
    assert result["after"] == 0
    kept = plain.read_text(encoding="utf-8")
    assert kept.startswith("2026-10-03 01:41:03,364 INFO")
    assert FAKE_TELEGRAM not in kept
    assert "[REDACTED_SECRET]" in kept

    gz = tmp_path / "mystic.log.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        fh.write(f"2026-10-03 02:00:00 WARNING GET https://newsapi.org/v2/everything?apiKey={FAKE_NEWS}&signature={FAKE_SIGNATURE}\n")
    gz_result = sanitize_log_file(gz)
    assert gz_result["status"] == "sanitized"
    with gzip.open(gz, "rt", encoding="utf-8") as fh:
        body = fh.read()
    assert body.startswith("2026-10-03 02:00:00 WARNING")
    assert FAKE_NEWS not in body and FAKE_SIGNATURE not in body
    assert sanitize_log_text(line).count("[REDACTED_SECRET]") == 1


def test_sanitizer_refuses_sqlite(tmp_path):
    db = tmp_path / "mystic_trading.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE paper_trades (id INTEGER)")
    conn.commit()
    conn.close()
    out = sanitize_log_file(db)
    assert out["status"] == "skipped_sqlite"
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 0
