"""Final closure: same-asset ownership sum, secret redaction, SCALP sleeve sharing."""

from __future__ import annotations

import gzip
import logging
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from backend.services.adaptive_learning import SIZE_BOUNDS
from backend.services.balance_sync_ownership import load_asset_ownership, material, quantity_drift
from backend.services.external_capital_flows import DEPOSIT, WITHDRAWAL, parse_endpoint, sync_flows
from backend.services.live_account_basis import TRAILING_BUY_ANCHOR_EQUITY
from backend.services.portfolio_engine import SCALP_MAX_OPEN_POSITIONS
from backend.services.two_engine_capital import (
    ENGINE_BUDGET_EXCEEDED,
    CapitalSnapshot,
    EngineBudget,
    check_engine_budget,
    compute_snapshot,
    get_capital_shares,
    scalp_order_notional,
)
from backend.utils.historical_log_sanitize import sanitize_log_file, sanitize_log_text
from backend.utils.secret_log_filter import credential_shape_count, install_secret_redacting_filter, redact_secrets

REPO = Path(__file__).resolve().parents[1]
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


MARKS = {"BTC/USDT": 84696.32, "ETH/USDT": 2685.68, "SOL/USDT": 119.67, "XRP/USDT": 1.4867}
XRP_MATCHED = [
    ("DAY_V2", "XRP/USDT", 151.16976, "ACTIVE"),
    ("SCALP_V2", "XRP/USDT", 0.0933, "DUST_PENDING"),
    ("DAY_V2", "BTC/USDT", 0.00000962, "DUST_PENDING"),
]


def _wallet(**coins: tuple[float, float]) -> list[dict]:
    return [{"asset": a, "free": str(f), "locked": str(lk)} for a, (f, lk) in coins.items()]


class _SyncResponse:
    status_code = 200

    def __init__(self, balances: list[dict]) -> None:
        self._balances = balances

    def json(self) -> dict:
        return {"balances": self._balances}


class _SyncClient:
    def __init__(self, balances: list[dict]) -> None:
        self._balances = balances

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None):
        return _SyncResponse(self._balances)


class _Limiter:
    async def consume(self, *args, **kwargs):
        return True


async def _run_balance_sync_once(monkeypatch, tmp_path, positions, balances, *, held=(), cash=591.67864992):
    import asyncio

    import backend.services.portfolio_engine_integration as pei
    from backend.utils import binance_weight_limiter as bwl

    db = tmp_path / "sync.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE portfolio_engine_positions (engine_id TEXT, symbol TEXT, quantity REAL, status TEXT, PRIMARY KEY (engine_id, symbol))")
    conn.executemany("INSERT INTO portfolio_engine_positions VALUES (?,?,?,?)", positions)
    conn.execute("CREATE TABLE engine_strategy_dust (engine_id TEXT, symbol TEXT, quantity REAL, status TEXT)")
    conn.executemany("INSERT INTO engine_strategy_dust VALUES (?,?,?,'HELD')", held)
    conn.execute("CREATE TABLE portfolio_engine_ledger (id INTEGER PRIMARY KEY, cash_balance REAL)")
    conn.execute("INSERT INTO portfolio_engine_ledger VALUES (1, ?)", (cash,))
    conn.execute("CREATE TABLE exchange_symbol_constraints (symbol TEXT, min_qty REAL, min_notional REAL)")
    conn.executemany(
        "INSERT INTO exchange_symbol_constraints VALUES (?,?,?)",
        [("BTC/USDT", 0.00001, 10.0), ("ETH/USDT", 0.0001, 10.0), ("SOL/USDT", 0.001, 10.0), ("XRP/USDT", 0.1, 10.0)],
    )
    conn.commit()
    conn.close()

    async def _create():
        return _Limiter()

    async def _no_flows(*args, **kwargs):
        return None

    real_sleep = asyncio.sleep
    integ = pei.PortfolioEngineIntegration.__new__(pei.PortfolioEngineIntegration)
    integ.is_running = True
    integ.engine = None
    integ.current_prices = dict(MARKS)
    integ._sync_external_capital_flows = _no_flows

    async def _sleep(seconds, *args, **kwargs):
        if seconds >= 300:
            integ.is_running = False
        await real_sleep(0)

    monkeypatch.setattr(pei, "DATABASE_PATH", str(db))
    monkeypatch.setenv("BINANCE_API_KEY", "synthetic-sync-key")
    monkeypatch.setenv("BINANCE_SECRET", "synthetic-sync-secret")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setattr(bwl.BinanceWeightLimiter, "create", staticmethod(_create))
    monkeypatch.setattr(httpx, "AsyncClient", lambda *_a, **_k: _SyncClient(balances))
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    await integ._binance_sync_loop()


async def test_balance_sync_sums_day_and_scalp_xrp_without_a_drift_warning(monkeypatch, tmp_path, caplog):
    balances = _wallet(USDT=(591.67864992, 0.0), XRP=(151.0, 0.26306), BTC=(0.00000962, 0.0))
    with caplog.at_level(logging.INFO, logger="backend.services.portfolio_engine_integration"):
        await _run_balance_sync_once(monkeypatch, tmp_path, XRP_MATCHED, balances)
    assert "drift" not in caplog.text.lower()
    assert "NOT on Binance" not in caplog.text
    assert "BINANCE_SYNC: OK" in caplog.text
    assert "BINANCE_SYNC_OWNERSHIP XRP Binance=151.26306000 owned=151.26306000" in caplog.text
    assert "DAY_V2 ACTIVE=151.16976000" in caplog.text and "SCALP_V2 DUST_PENDING=0.09330000" in caplog.text


@pytest.mark.parametrize(
    ("positions", "held", "balances", "expected"),
    [
        (XRP_MATCHED, (), _wallet(USDT=(591.67864992, 0.0), XRP=(152.26306, 0.0), BTC=(0.00000962, 0.0)), "XRP drift!"),
        (XRP_MATCHED, (), _wallet(USDT=(591.67864992, 0.0), XRP=(151.16976, 0.0), BTC=(0.00000962, 0.0)), "XRP drift!"),
        ([("SCALP_V2", "BTC/USDT", 0.0012, "ACTIVE")], (), _wallet(USDT=(591.67864992, 0.0), BTC=(0.00000962, 0.0)), "BTC drift!"),
        ([("SCALP_V2", "BTC/USDT", 0.0012, "ACTIVE")], (), _wallet(USDT=(591.67864992, 0.0)), "BTC exists locally"),
        ([("DAY_V2", "ETH/USDT", 0.03, "ACTIVE")], (), _wallet(USDT=(591.67864992, 0.0), ETH=(0.025, 0.0)), "ETH drift!"),
        ([("DAY_V2", "SOL/USDT", 1.0, "ACTIVE")], (("SCALP_V2", "SOL/USDT", 0.004),), _wallet(USDT=(591.67864992, 0.0), SOL=(0.9, 0.0)), "SOL drift!"),
    ],
)
async def test_balance_sync_still_warns_on_a_real_quantity_mismatch(monkeypatch, tmp_path, caplog, positions, held, balances, expected):
    with caplog.at_level(logging.INFO, logger="backend.services.portfolio_engine_integration"):
        await _run_balance_sync_once(monkeypatch, tmp_path, positions, balances, held=held)
    assert expected in caplog.text
    assert "BINANCE_SYNC: Drift detected" in caplog.text


async def test_balance_sync_ignores_unowned_dust(monkeypatch, tmp_path, caplog):
    balances = _wallet(USDT=(591.67864992, 0.0), BTC=(0.00000962, 0.0))
    with caplog.at_level(logging.INFO, logger="backend.services.portfolio_engine_integration"):
        await _run_balance_sync_once(monkeypatch, tmp_path, [], balances)
    assert "drift" not in caplog.text.lower()
    assert "BINANCE_SYNC: OK" in caplog.text


def test_priced_tolerance_catches_btc_gaps_the_unit_tolerance_missed():
    btc = MARKS["BTC/USDT"]
    gap = 0.0012
    assert not quantity_drift(0.00000962, 0.00000962 + gap)
    assert quantity_drift(0.00000962, 0.00000962 + gap, price=btc)
    assert not quantity_drift(151.26306, 151.26306 + 1e-9, price=MARKS["XRP/USDT"])
    assert not material(0.00000962, btc)
    assert material(0.0012, btc)


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


def _lot(engine: str, cost: float, status: str = "ACTIVE") -> SimpleNamespace:
    return SimpleNamespace(engine_id=engine, status=status, quantity=1.0, entry_price=cost, original_position_cost=cost)


def _reservation_db(tmp_path, rows=()) -> str:
    db = str(tmp_path / "capital.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE day_entry_reservations (reservation_id TEXT, sleeve TEXT, notional_usd REAL, status TEXT, expires_at REAL)")
        conn.executemany("INSERT INTO day_entry_reservations VALUES (?,?,?,?,?)", rows)
    return db


@pytest.mark.parametrize("equity", [800.0, 1500.0, 3000.0, 10000.0])
@pytest.mark.parametrize("mult", [0.50, 0.75, 1.00, 1.25])
def test_each_scalp_slot_sees_the_sleeve_left_by_earlier_slots(tmp_path, equity, mult):
    db = _reservation_db(tmp_path)
    scalp_share = get_capital_shares()[1]
    positions: dict[str, SimpleNamespace] = {}
    sleeve = equity * scalp_share
    remaining_seen = []
    for slot in range(1, SCALP_MAX_OPEN_POSITIONS + 1):
        deployed = sum(p.original_position_cost for p in positions.values())
        free_cash = equity - deployed
        snap = compute_snapshot(db, equity, free_cash, positions)
        assert snap.scalp.target_capital == pytest.approx(sleeve)
        assert snap.scalp.remaining_budget == pytest.approx(sleeve - deployed)
        order = scalp_order_notional(snap, slots=SCALP_MAX_OPEN_POSITIONS, size_mult=mult)
        assert order == pytest.approx(min(snap.scalp.target_capital / SCALP_MAX_OPEN_POSITIONS * mult, snap.scalp.remaining_budget))
        ok, reason, _ = check_engine_budget(db, "SCALP_V2", order, equity, free_cash, positions)
        assert ok, (slot, reason)
        remaining_seen.append(snap.scalp.remaining_budget)
        positions[f"SCALP_V2::S{slot}"] = _lot("SCALP_V2", order)
    assert remaining_seen == sorted(remaining_seen, reverse=True)
    assert len(set(remaining_seen)) == SCALP_MAX_OPEN_POSITIONS
    assert sum(p.original_position_cost for p in positions.values()) <= sleeve + 1e-9
    after = compute_snapshot(db, equity, equity - sleeve, positions)
    over = after.scalp.remaining_budget + 0.01
    ok, reason, _ = check_engine_budget(db, "SCALP_V2", over, equity, equity, positions)
    assert not ok and reason == ENGINE_BUDGET_EXCEEDED


def test_full_size_slots_cannot_overrun_the_sleeve():
    lo, hi = SIZE_BOUNDS["SCALP_V2"]
    sleeve = 1500.0 * get_capital_shares()[1]
    snap = _snap(sleeve, sleeve)
    first = scalp_order_notional(snap, slots=SCALP_MAX_OPEN_POSITIONS, size_mult=hi)
    assert first == pytest.approx(sleeve / SCALP_MAX_OPEN_POSITIONS * hi)
    assert SCALP_MAX_OPEN_POSITIONS * first > sleeve
    assert scalp_order_notional(snap, slots=SCALP_MAX_OPEN_POSITIONS, size_mult=lo) == pytest.approx(sleeve / SCALP_MAX_OPEN_POSITIONS * lo)


def test_pending_scalp_reservations_and_other_engines_lots_are_counted_correctly(tmp_path):
    equity = 1500.0
    sleeve = equity * get_capital_shares()[1]
    far = time.time() + 3600
    db = _reservation_db(
        tmp_path,
        [("r-scalp", "SCALP_V2", sleeve / 8, "ACTIVE", far), ("r-day", "DAY_V2", 200.0, "ACTIVE", far), ("r-old", "SCALP_V2", 999.0, "ACTIVE", time.time() - 5)],
    )
    positions = {
        "DAY_V2::XRP/USDT": _lot("DAY_V2", 300.0),
        "SCALP_V2::XRP/USDT": _lot("SCALP_V2", 5.0, status="DUST_PENDING"),
        "SCALP_V2::BTC/USDT": _lot("SCALP_V2", sleeve / 4),
    }
    snap = compute_snapshot(db, equity, equity - 300.0 - sleeve / 4, positions)
    assert snap.scalp.committed_reservations == pytest.approx(sleeve / 8)
    assert snap.scalp.committed_positions == pytest.approx(sleeve / 4)
    assert snap.scalp.remaining_budget == pytest.approx(sleeve - sleeve / 4 - sleeve / 8)
    assert snap.day.committed_reservations == pytest.approx(200.0)


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


FAKE_LONG_BINANCE_KEY = "SyntheticBinanceApiKey" + "0123456789" * 4
FAKE_HEX_SIGNATURE = "ab" * 32


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        (f"HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded with url: /bot{FAKE_TELEGRAM}/sendMessage", FAKE_TELEGRAM),
        (f'https://api.telegram.org:443 "POST /bot{FAKE_TELEGRAM}/sendMessage HTTP/1.1" 200 None', FAKE_TELEGRAM),
        (f"headers={{'X-MBX-APIKEY': '{FAKE_LONG_BINANCE_KEY}'}}", FAKE_LONG_BINANCE_KEY),
        (f'{{"X-Api-Key": "{FAKE_NEWS}"}}', FAKE_NEWS),
        (f"headers={{'Authorization': 'Bearer {FAKE_BEARER}'}}", FAKE_BEARER),
        (f"params={{'timestamp': 1, 'signature': '{FAKE_HEX_SIGNATURE}'}}", FAKE_HEX_SIGNATURE),
        (f"Cfg(api_key='{FAKE_BINANCE_KEY}', secret_key=\"{FAKE_SIGNATURE}\")", FAKE_BINANCE_KEY),
        (f"GET /api/v3/order?symbol=XRPUSDT&timestamp=1&signature={FAKE_HEX_SIGNATURE}", FAKE_HEX_SIGNATURE),
        (f"https://example.test/cb?access_token={FAKE_BEARER}", FAKE_BEARER),
        (f"https://example.test/cb?password={FAKE_BEARER}&ok=1", FAKE_BEARER),
    ],
)
def test_redaction_covers_requests_urllib3_and_mapping_forms(text, secret):
    out = redact_secrets(text)
    assert secret not in out
    assert redact_secrets(out) == out
    assert credential_shape_count(redact_secrets(text, replacement="[REDACTED_SECRET]")) == 0


@pytest.mark.parametrize(
    "line",
    [
        "BINANCE_SYNC: OK - Balances match (USDT=$12.34, 3 assets)",
        "{'symbol': 'XRPUSDT', 'orderId': 123, 'status': 'FILLED'}",
        "token bucket refill 5",
        "2026-10-03 21:13:36,484 INFO SCALP slot 2 notional=102.20",
    ],
)
def test_redaction_leaves_ordinary_lines_alone(line):
    assert redact_secrets(line) == line


def test_httpx_url_request_and_headers_objects_are_redacted(caplog):
    install_secret_redacting_filter()
    url = httpx.URL(f"https://api.telegram.org/bot{FAKE_TELEGRAM}/getUpdates", params={"offset": 1})
    req = httpx.Request("GET", f"https://api.binance.us/api/v3/account?timestamp=1&signature={FAKE_HEX_SIGNATURE}", headers={"X-MBX-APIKEY": FAKE_LONG_BINANCE_KEY})
    with caplog.at_level(logging.INFO):
        logging.getLogger("httpx").info('HTTP Request: %s %s "%s %d %s"', "GET", url, "HTTP/1.1", 200, "OK")
        logging.getLogger("closure.request").info("request %r headers %s", req, req.headers)
    for secret in (FAKE_TELEGRAM, FAKE_HEX_SIGNATURE, FAKE_LONG_BINANCE_KEY):
        assert secret not in caplog.text
    assert "/bot***/getUpdates" in caplog.text


def test_exception_tracebacks_and_stack_text_are_redacted(caplog):
    install_secret_redacting_filter()
    lg = logging.getLogger("closure.traceback")
    with caplog.at_level(logging.INFO):
        try:
            raise httpx.HTTPStatusError(
                f"Client error '401 Unauthorized' for url 'https://api.binance.us/api/v3/order?signature={FAKE_HEX_SIGNATURE}'",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(401),
            )
        except httpx.HTTPStatusError:
            lg.exception("order failed")
        try:
            raise ConnectionError(f"Max retries exceeded with url: /bot{FAKE_TELEGRAM}/getUpdates")
        except ConnectionError as exc:
            lg.error("telegram failed: %s", exc, exc_info=True)
        lg.info("stack", stack_info=True)
    assert "Traceback" in caplog.text
    assert FAKE_HEX_SIGNATURE not in caplog.text
    assert FAKE_TELEGRAM not in caplog.text


def test_untraced_exception_keeps_exc_info_for_handlers(caplog):
    install_secret_redacting_filter()
    with caplog.at_level(logging.ERROR):
        try:
            raise ValueError("ordinary failure")
        except ValueError:
            logging.getLogger("closure.plain").exception("plain")
    assert caplog.records[-1].exc_info is not None
    assert "ordinary failure" in caplog.text


def test_configured_credential_value_is_redacted_in_any_form(monkeypatch, caplog):
    value = "syntheticConfiguredValue0123456789"
    monkeypatch.setenv("MYSTIC_TEST_SYNTHETIC_API_KEY", value)
    install_secret_redacting_filter()
    assert value not in redact_secrets(f"using {value} for this call")
    with caplog.at_level(logging.INFO):
        logging.getLogger("closure.env").info("raw %s", value)
    assert value not in caplog.text


@pytest.mark.parametrize("in_thread", [False, True])
def test_uncaught_exceptions_written_to_stderr_are_redacted(in_thread):
    body = f"raise RuntimeError('Max retries exceeded with url: /bot{FAKE_TELEGRAM}/getUpdates')"
    if in_thread:
        code = f"import threading\nimport backend\ndef w():\n    {body}\nt = threading.Thread(target=w)\nt.start()\nt.join()\n"
    else:
        code = f"import backend\n{body}\n"
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert "RuntimeError" in proc.stderr
    assert FAKE_TELEGRAM not in proc.stderr
    assert "/bot***/getUpdates" in proc.stderr


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
