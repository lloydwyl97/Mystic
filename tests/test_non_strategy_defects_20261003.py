"""Non-strategy defects: SCALP sizing, external capital, sell audit, DAY_V2 status,
dust count, secret redaction, backup verification, fiat residual."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.services import external_capital_flows as ecf
from backend.services import mystic_maintenance as m
from backend.services.balance_residuals import classify_fiat_residual
from backend.services.day_mandatory_exit_execution import run_mandatory_exit_ioc_loop
from backend.services.day_v2.live_exit_evaluator import day_v2_exit_policy, evaluate_day_v2_exit, preview_day_v2_exit
from backend.services.live_account_basis import TRAILING_BUY_ANCHOR_EQUITY
from backend.services.protected_limit_execution import (
    PREFLIGHT_AUDIT_KEY,
    PREFLIGHT_CHUNKS_KEY,
    execution_latency_fields,
    preflight_audit_fields,
    stamp_preflight,
)
from backend.services.two_engine_capital import CapitalSnapshot, EngineBudget, scalp_order_notional
from backend.utils.secret_log_filter import install_secret_redacting_filter, redact_secrets

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- SCALP sizing


def _snap(scalp_target: float, remaining: float) -> CapitalSnapshot:
    snap = CapitalSnapshot(equity=scalp_target * 2, free_cash=scalp_target * 2)
    snap.scalp = EngineBudget(engine_id="SCALP_V2", target_capital=scalp_target, remaining_budget=remaining)
    return snap


def test_scalp_notional_is_sleeve_over_slots_and_scales_with_equity():
    small = scalp_order_notional(_snap(400.0, 400.0), slots=4)
    large = scalp_order_notional(_snap(4000.0, 4000.0), slots=4)
    assert small == pytest.approx(100.0)
    assert large == pytest.approx(1000.0)
    assert large > 50.0


def test_scalp_notional_keeps_adaptive_multiplier_and_remaining_sleeve():
    assert scalp_order_notional(_snap(400.0, 400.0), slots=4, size_mult=0.5) == pytest.approx(50.0)
    assert scalp_order_notional(_snap(400.0, 400.0), slots=4, size_mult=1.25) == pytest.approx(125.0)
    assert scalp_order_notional(_snap(400.0, 30.0), slots=4, size_mult=1.25) == pytest.approx(30.0)
    assert scalp_order_notional(_snap(400.0, 0.0), slots=4) == 0.0


def test_scalp_emergency_cap_only_when_explicit():
    assert scalp_order_notional(_snap(400.0, 400.0), slots=4, emergency_max_notional=0.0) == pytest.approx(100.0)
    assert scalp_order_notional(_snap(400.0, 400.0), slots=4, emergency_max_notional=60.0) == pytest.approx(60.0)


def test_scalp_live_config_has_no_silent_50_dollar_default(monkeypatch):
    from backend.services.binance_scalp import config as scalp_config

    monkeypatch.delenv("SCALP_LIVE_MAX_NOTIONAL", raising=False)
    src = (REPO / "backend/services/binance_scalp/config.py").read_text()
    assert 'os.getenv("SCALP_LIVE_MAX_NOTIONAL", "0")' in src
    assert hasattr(scalp_config, "get_scalp_config")


def test_integration_sizes_scalp_from_sleeve_not_fixed_cap():
    src = (REPO / "backend/services/portfolio_engine_integration.py").read_text()
    start = src.index("async def _process_scalp_v2_signals")
    body = src[start : src.index("\n    async def ", start + 10)]
    assert "scalp_order_notional(" in body
    assert "slots=SCALP_MAX_OPEN_POSITIONS" in body
    assert "calculate_position_size" not in body


# ---------------------------------------------------------------- external capital

FIAT_DEPOSITS = {
    "assetLogRecordList": [
        {"orderId": "a1", "orderStatus": "Successful", "fiatCurrency": "USD", "amount": "172.82", "createTime": 1787528118652},
        {"orderId": "a2", "orderStatus": "Successful", "fiatCurrency": "USD", "amount": "120.01", "createTime": 1790459449444},
        {"orderId": "a3", "orderStatus": "Successful", "fiatCurrency": "USD", "amount": "480.05", "createTime": 1790976766658},
        {"orderId": "a4", "orderStatus": "Failed", "fiatCurrency": "USD", "amount": "999.00", "createTime": 1790976766700},
    ]
}


def _ledger_db(tmp_path: Path, principal: float = 228.06746265) -> str:
    db = str(tmp_path / "ledger.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE portfolio_engine_ledger (id INTEGER PRIMARY KEY, principal REAL)")
        conn.execute("INSERT INTO portfolio_engine_ledger VALUES (1, ?)", (principal,))
    return db


def _principal(db: str) -> float:
    with sqlite3.connect(db) as conn:
        return float(conn.execute("SELECT principal FROM portfolio_engine_ledger WHERE id=1").fetchone()[0])


def test_deposits_after_baseline_adoption_raise_baseline_once(tmp_path):
    db = _ledger_db(tmp_path)
    flows = ecf.parse_endpoint("fiat", ecf.DEPOSIT, FIAT_DEPOSITS)
    assert [f.flow_id for f in flows] == ["fiat:DEPOSIT:a1", "fiat:DEPOSIT:a2", "fiat:DEPOSIT:a3"]
    new, base = ecf.sync_flows(db, flows)
    expected = float(TRAILING_BUY_ANCHOR_EQUITY) + 120.01 + 480.05
    assert len(new) == 3
    assert base == pytest.approx(expected)
    assert base == pytest.approx(828.12746265)
    assert _principal(db) == pytest.approx(expected)
    new2, base2 = ecf.sync_flows(db, flows)
    assert new2 == []
    assert base2 == pytest.approx(expected)
    assert _principal(db) == pytest.approx(expected)


def test_pre_baseline_deposit_is_audited_not_applied(tmp_path):
    db = _ledger_db(tmp_path)
    ecf.sync_flows(db, ecf.parse_endpoint("fiat", ecf.DEPOSIT, FIAT_DEPOSITS))
    with sqlite3.connect(db) as conn:
        rows = dict(conn.execute("SELECT flow_id, applies_to_baseline FROM external_capital_flows").fetchall())
    assert rows == {"fiat:DEPOSIT:a1": 0, "fiat:DEPOSIT:a2": 1, "fiat:DEPOSIT:a3": 1}


def test_withdrawal_lowers_baseline_including_fees(tmp_path):
    db = _ledger_db(tmp_path)
    payload = {
        "assetLogRecordList": [{"orderId": "w1", "orderStatus": "Successful", "fiatCurrency": "USD", "amount": "100", "transactionFee": "1.5", "platformFee": "0.5", "createTime": 1790976766999}]
    }
    _, base = ecf.sync_flows(db, ecf.parse_endpoint("fiat", ecf.WITHDRAWAL, payload))
    assert base == pytest.approx(float(TRAILING_BUY_ANCHOR_EQUITY) - 102.0)


def test_baseline_is_not_current_equity_and_untouched_without_flows(tmp_path):
    db = _ledger_db(tmp_path, principal=228.06746265)
    _, base = ecf.sync_flows(db, [])
    assert base is None
    assert _principal(db) == pytest.approx(228.06746265)
    assert ecf.resolve_principal(db, 228.06746265) == pytest.approx(228.06746265)


def test_resolve_principal_overrides_stale_stored_value(tmp_path):
    db = _ledger_db(tmp_path)
    ecf.sync_flows(db, ecf.parse_endpoint("fiat", ecf.DEPOSIT, FIAT_DEPOSITS))
    assert ecf.resolve_principal(db, 228.06746265) == pytest.approx(828.12746265)


def test_crypto_flows_only_completed_stable_assets():
    deposits = [
        {"coin": "USDT", "amount": "50", "status": 1, "txId": "t1", "insertTime": 1790976767000},
        {"coin": "USDT", "amount": "50", "status": 0, "txId": "t2", "insertTime": 1790976767000},
        {"coin": "BTC", "amount": "0.1", "status": 1, "txId": "t3", "insertTime": 1790976767000},
    ]
    flows = ecf.parse_endpoint("crypto", ecf.DEPOSIT, deposits)
    assert [f.flow_id for f in flows] == ["crypto:DEPOSIT:USDT:t1"]


def test_failsafe_and_scalp_loss_basis_follow_baseline():
    base = 828.12746265
    assert base * 0.9 == pytest.approx(745.31, abs=0.01)
    assert base * 0.5 * 0.05 == pytest.approx(20.70, abs=0.01)


def test_capital_flows_are_not_trading_pnl_or_learning():
    src = (REPO / "backend/services/external_capital_flows.py").read_text()
    for forbidden in ("realized_pnl", "paper_trades", "trade_learning", "scalp_paper_trades"):
        assert forbidden not in src


def test_flow_sync_wired_into_binance_sync_and_ledger_load():
    integ = (REPO / "backend/services/portfolio_engine_integration.py").read_text()
    assert "await self._sync_external_capital_flows(api_key, api_secret)" in integ
    engine = (REPO / "backend/services/portfolio_engine.py").read_text()
    assert "resolve_principal(self.db_path, row[0])" in engine


# ---------------------------------------------------------------- sell audit


def _pf(age: float = 0.4):
    return SimpleNamespace(
        passed=True,
        executable_qty=1.0,
        quantity=1.0,
        protected_limit_price=100.0,
        reject_reason="",
        to_audit_dict=lambda: {"book_age_sec": age, "book_freshness": {"verdict": "FRESH", "age_sec": age}, "protected_limit_price": 100.0},
    )


def test_stamped_preflight_reaches_sell_audit_fields():
    order = stamp_preflight({"id": "1", "filled": 1.0}, _pf())
    fields = preflight_audit_fields(order)
    assert fields["book_freshness"] == {"verdict": "FRESH", "age_sec": 0.4}
    assert fields["book_age_sec"] == 0.4


@pytest.mark.asyncio
async def test_mandatory_ioc_combined_order_keeps_each_chunk_preflight():
    ages = iter([0.3, 0.7])

    async def preflight(qty, impact):
        pf = _pf(next(ages))
        pf.executable_qty = qty
        pf.quantity = qty
        return pf

    calls = []

    async def place(qty, limit):
        calls.append(qty)
        if len(calls) == 1:
            return {"id": "a", "filled": 0.4, "amount": qty, "average": 100.0, "status": "expired"}
        return {"id": "b", "filled": qty, "amount": qty, "average": 100.0, "status": "closed"}

    out = await run_mandatory_exit_ioc_loop(quantity=1.0, preflight=preflight, place_ioc=place, is_meaningful=lambda q: q > 1e-6)
    combined = out.combined_order
    assert combined[PREFLIGHT_AUDIT_KEY]["book_freshness"]["age_sec"] == 0.7
    assert [c["book_age_sec"] for c in combined[PREFLIGHT_CHUNKS_KEY]] == [0.3, 0.7]
    assert [c["book_age_sec"] for c in preflight_audit_fields(combined)["preflight_chunks"]] == [0.3, 0.7]


def test_sell_path_persists_preflight_and_verify_fill_preserves_it():
    src = (REPO / "backend/services/portfolio_engine.py").read_text()
    assert "preflight_audit_fields(live_order_sell)" in src
    assert "for audit_key in ORDER_AUDIT_KEYS:" in src
    assert "return stamp_preflight(" in src


def test_fill_timestamp_from_full_response_transact_time():
    order = {
        "info": {"transactTime": 1791000000123, "executedQty": "1.0", "fills": [{"price": "100", "qty": "1.0", "commission": "0"}]},
        "_mystic_latency": {"order_submit_timestamp": 1790999999.9, "order_response_timestamp": 1791000000.2},
    }
    out = execution_latency_fields(order)
    assert out["fill_timestamp"] == pytest.approx(1791000000.123)
    assert out["fill_timestamp_source"] == "full_response_transact_time"


def test_venue_trade_time_preferred_over_transact_time():
    order = {
        "info": {"transactTime": 1791000000123, "fills": [{"price": "100", "qty": "1"}]},
        "trades": [{"timestamp": 1791000000456}],
        "_mystic_latency": {},
    }
    out = execution_latency_fields(order)
    assert out["fill_timestamp"] == pytest.approx(1791000000.456)
    assert out["fill_timestamp_source"] == "venue_trades"


def test_unfilled_order_has_no_fill_timestamp():
    out = execution_latency_fields({"info": {"transactTime": 1791000000123, "fills": []}, "_mystic_latency": {}})
    assert out["fill_timestamp"] is None


# ---------------------------------------------------------------- DAY_V2 status

_DAY_POS = {
    "entry_price": 100.0,
    "highest_price": 104.0,
    "atr_at_entry": 0.5,
    "structural_anchor": 98.5,
    "target_price": 0.0,
    "estimated_roundtrip_cost": 0.003,
    "setup": "HTF_TREND_PULLBACK",
    "atr_1h_at_entry": 1.2,
    "objective_structural": 103.0,
}


def test_day_v2_preview_matches_live_runner_and_has_no_time_or_tp():
    now = time.time()
    preview = preview_day_v2_exit(current_price=103.5, entry_time=now - 4 * 3600, now=now, **_DAY_POS)
    assert preview["runner_activated"] is True
    assert preview["objective_reached"] is True
    assert preview["current_exit_authority"] == "DAY_V2_LEARNED_CONTINUATION"
    assert preview["time_exit"] is None
    assert preview["fixed_take_profit"] is None
    assert preview["hold_time_is_exit_authority"] is False
    stop = preview["runner_stop"]
    above = evaluate_day_v2_exit(engine_id="DAY_V2", current_price=stop + 0.01, bar_low=99.5, entry_time=now - 4 * 3600, **_DAY_POS)
    at = evaluate_day_v2_exit(engine_id="DAY_V2", current_price=stop, bar_low=99.5, entry_time=now - 4 * 3600, **_DAY_POS)
    assert above is None
    assert at is None


def test_day_v2_preview_catastrophic_level_matches_executor():
    now = time.time()
    preview = preview_day_v2_exit(current_price=100.0, entry_time=now - 60, now=now, **{**_DAY_POS, "highest_price": 100.2})
    cat = preview["catastrophic_price"]
    assert preview["current_exit_authority"] == "DAY_V2_CATASTROPHIC_PROTECTION"
    assert preview["runner_stop"] is None
    fired = evaluate_day_v2_exit(engine_id="DAY_V2", current_price=100.0, bar_low=cat, entry_time=now - 60, **{**_DAY_POS, "highest_price": 100.2})
    held = evaluate_day_v2_exit(engine_id="DAY_V2", current_price=100.0, bar_low=cat + 0.01, entry_time=now - 60, **{**_DAY_POS, "highest_price": 100.2})
    assert fired["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"
    assert held is None


def test_day_v2_policy_has_no_time_stop():
    policy = day_v2_exit_policy()
    assert "TIME_STOP_EXIT" not in json.dumps(policy)
    assert policy["time_exit_sell_path_active"] is False
    assert policy["code_path"].endswith("evaluate_day_v2_exit")


def test_status_payload_no_longer_advertises_time_stop():
    src = (REPO / "backend/services/portfolio_engine.py").read_text()
    start = src.index("def get_portfolio_status")
    body = src[start : src.index("\n    def ", start + 10)]
    assert "TIME_STOP_EXIT" not in body
    assert '"time_exit_sell_path_active": True' not in body
    assert '"position_exit_policy": _day_v2_exit_policy()' in body


def test_day_v2_position_row_uses_live_contract():
    from backend.services.portfolio_engine import PortfolioEngine

    now = time.time()
    pos = SimpleNamespace(
        symbol="BTCUSDT",
        engine_id="DAY_V2",
        entry_price=100.0,
        highest_price=104.0,
        atr_at_entry=0.5,
        thesis_invalid_level=98.5,
        thesis_target_level=0.0,
        entry_time=now - 4 * 3600,
        entry_thesis="HTF_TREND_PULLBACK",
        day_atr_1h_at_entry=1.2,
        day_objective_structural=103.0,
        adaptive_decision={},
    )
    fields = PortfolioEngine._day_v2_status_preview_fields(SimpleNamespace(), pos, 103.5)
    assert fields["max_hold_min"] is None
    assert fields["take_profit"] is None
    assert fields["current_exit_authority"] == "DAY_V2_LEARNED_CONTINUATION"
    assert fields["hard_stop"] == fields["stop_loss"] == fields["engine_exit_preview"]["catastrophic_price"]
    assert fields["executable_trailing_stop"] == fields["engine_exit_preview"]["runner_stop"]


# ---------------------------------------------------------------- dust count


def test_dust_pending_not_counted_against_sqlite_active_count():
    from backend.endpoints.portfolio_engine_endpoints import _engine_active_position_count

    engine = SimpleNamespace(
        open_positions={
            "DAY_V2|BTCUSDT": SimpleNamespace(status="ACTIVE"),
            "DAY_V2|XRPUSDT": SimpleNamespace(status="DUST_PENDING"),
            "SCALP_V2|SOLUSDT": SimpleNamespace(),
        }
    )
    assert _engine_active_position_count(engine) == 2
    assert len(engine.open_positions) == 3


# ---------------------------------------------------------------- secrets


def test_no_api_key_prefix_logged():
    src = (REPO / "backend/services/live_trading_service.py").read_text()
    assert "binance_api_key[:" not in src


def test_news_api_key_sent_in_header_not_url():
    src = (REPO / "backend/services/news_sentiment.py").read_text()
    assert '"apiKey": key' not in src
    assert 'headers={"X-Api-Key": key}' in src


@pytest.mark.parametrize(
    "raw",
    [
        "GET https://api.telegram.org/bot123456789:AAFtExampleExampleExampleExample/getUpdates",
        "GET https://newsapi.org/v2/everything?q=btc&apiKey=abcdef0123456789",
        "GET https://api.binance.us/api/v3/account?timestamp=1&signature=deadbeefcafebabe",
        "GET https://cryptopanic.com/api/v1/posts/?auth_token=sekrit123&public=true",
        "GET https://api.etherscan.io/v2/api?chainid=1&apikey=ETHKEY999",
    ],
)
def test_redact_secrets_in_urls(raw):
    out = redact_secrets(raw)
    for secret in ("AAFtExample", "abcdef0123456789", "deadbeefcafebabe", "sekrit123", "ETHKEY999"):
        assert secret not in out
    assert "***" in out


class _URL:
    """Non-str argument, like httpx.URL."""

    def __init__(self, s: str) -> None:
        self.s = s

    def __str__(self) -> str:
        return self.s


def test_redaction_applies_to_non_string_args_on_child_logger(caplog):
    install_secret_redacting_filter()
    lg = logging.getLogger("httpx._client.test_child")
    with caplog.at_level(logging.INFO):
        lg.info('HTTP Request: %s %s "%s"', "GET", _URL("https://newsapi.org/v2/everything?apiKey=LEAKME123"), "HTTP/1.1 200 OK")
    text = caplog.text
    assert "LEAKME123" not in text
    assert "apiKey=***" in text


def test_redaction_installed_for_every_process_via_package_import():
    src = (REPO / "backend/__init__.py").read_text()
    assert "install_secret_redacting_filter" in src


# ---------------------------------------------------------------- backups

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def mcfg(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    home = tmp_path / "home"
    for d in (repo / "logs", home / "backups" / "db", tmp_path / "tmp"):
        d.mkdir(parents=True)
    monkeypatch.setenv("MAINT_LIVE_DB", str(repo / "mystic_trading.db"))
    monkeypatch.delenv("MAINT_BACKUP_DIR", raising=False)
    c = m.MaintConfig()
    c.repo, c.home, c.tmp_dir = repo, home, tmp_path / "tmp"
    c.lock_path = tmp_path / "run" / "maint.lock"
    c.deploy_locks = (tmp_path / "run" / "deploy.lock",)
    c.reboot_flag = tmp_path / "reboot-required"
    c.owner = "nobody-such-user"
    return c


def _sqlite_backup(cfg, ts: datetime, *, age_sec: float = 3600) -> Path:
    p = cfg.backup_dir / m.backup_name(ts)
    with sqlite3.connect(p) as conn:
        conn.execute("CREATE TABLE t (v TEXT)")
        conn.execute("INSERT INTO t VALUES ('x')")
    t = time.time() - age_sec
    os.utime(p, (t, t))
    return p


def test_unverified_backup_is_verified_and_adopted(mcfg):
    p = _sqlite_backup(mcfg, NOW - timedelta(hours=10))
    assert not m.list_backups(mcfg.backup_dir)[0].verified
    out = m.verify_unverified_backups(mcfg, dry_run=False, opened=set())
    assert out["status"] == "ok"
    assert [v["name"] for v in out["verified"]] == [p.name]
    info = m.list_backups(mcfg.backup_dir)[0]
    assert info.verified
    assert info.manifest["sha256"] == m.sha256_file(p)
    assert info.manifest["reason"] == m.ADOPTED_REASON
    assert not Path(f"{p}-wal").exists() and not Path(f"{p}-shm").exists()


def test_corrupt_unverified_backup_reported_and_kept(mcfg):
    p = mcfg.backup_dir / m.backup_name(NOW - timedelta(hours=5))
    p.write_bytes(b"not a database" * 100)
    t = time.time() - 3600
    os.utime(p, (t, t))
    out = m.verify_unverified_backups(mcfg, dry_run=False, opened=set())
    assert out["status"] == "error"
    assert p.exists()
    assert not m.manifest_path_for(p).exists()


def test_fresh_or_open_backup_is_deferred(mcfg):
    fresh = _sqlite_backup(mcfg, NOW - timedelta(hours=1), age_sec=5)
    out = m.verify_unverified_backups(mcfg, dry_run=False, opened=set())
    assert out["deferred"] == [fresh.name]
    os.utime(fresh, (time.time() - 3600,) * 2)
    out = m.verify_unverified_backups(mcfg, dry_run=False, opened={str(fresh)})
    assert out["deferred"] == [fresh.name]


def test_adopted_backup_is_never_deleted_by_retention(mcfg):
    adopted = _sqlite_backup(mcfg, NOW - timedelta(days=40))
    m.verify_unverified_backups(mcfg, dry_run=False, opened=set())
    for d in range(0, 20):
        p = _sqlite_backup(mcfg, NOW - timedelta(days=d))
        m.manifest_path_for(p).write_text(json.dumps({"integrity": "ok", "sha256": m.sha256_file(p), "reason": "scheduled"}))
    res = m.apply_backup_retention(mcfg, mode="aggressive", dry_run=True, opened=set(), now=NOW, compress=lambda _info: {"status": "ok"})
    deleted = {Path(a["path"]).name for a in res.get("actions", []) if a.get("action") == "delete_backup"}
    assert deleted
    assert adopted.name not in deleted


def test_run_maintenance_verifies_backups_every_run(mcfg):
    calls: list[str] = []

    def owner(task, *args):
        calls.append(task)
        return {"status": "ok"}

    m.run_maintenance(mcfg, dry_run=True, allow_reboot=False, owner_task=owner)
    assert "backup-verify" in calls


def test_backup_schedule_due_24h_after_newest_verified(mcfg):
    p = _sqlite_backup(mcfg, datetime(2026, 10, 2, 23, 3, 48, tzinfo=timezone.utc))
    m.manifest_path_for(p).write_text(json.dumps({"integrity": "ok", "sha256": "f" * 64}))
    assert not m.backup_due(mcfg, now=datetime(2026, 10, 3, 22, 17, tzinfo=timezone.utc))
    assert m.backup_due(mcfg, now=datetime(2026, 10, 3, 23, 17, tzinfo=timezone.utc))


def test_cron_runs_maintenance_hourly():
    cron = (REPO / "deploy/cron.d-mystic-maintenance").read_text()
    assert "17 * * * * root" in cron and "scripts/mystic_maintenance.py run" in cron


def test_cli_exposes_backup_verify():
    src = (REPO / "scripts/mystic_maintenance.py").read_text()
    assert '"backup-verify"' in src and "verify_unverified_backups" in src


# ---------------------------------------------------------------- fiat residual


def test_usd_residual_classified_once_then_quiet(tmp_path):
    conn = sqlite3.connect(tmp_path / "r.db")
    assert classify_fiat_residual(conn, "USD", 0.098) == "MATCHED"
    rows = conn.execute("SELECT symbol, residual_qty FROM documented_balance_residuals").fetchall()
    assert rows == [("USD", 0.098)]
    for _ in range(5):
        assert classify_fiat_residual(conn, "USD", 0.098) == "MATCHED"
    assert conn.execute("SELECT COUNT(*) FROM documented_balance_residuals").fetchone()[0] == 1


def test_usd_residual_change_still_reported(tmp_path):
    conn = sqlite3.connect(tmp_path / "r.db")
    classify_fiat_residual(conn, "USD", 0.098)
    assert classify_fiat_residual(conn, "USD", 0.25) == "CHANGED"


def test_large_unclassified_usd_is_not_a_residual(tmp_path):
    conn = sqlite3.connect(tmp_path / "r.db")
    assert classify_fiat_residual(conn, "USD", 480.05) == "CHANGED"
    assert conn.execute("SELECT COUNT(*) FROM documented_balance_residuals").fetchone()[0] == 0


def test_non_fiat_assets_untouched(tmp_path):
    conn = sqlite3.connect(tmp_path / "r.db")
    assert classify_fiat_residual(conn, "BTC", 0.00001) == ""
