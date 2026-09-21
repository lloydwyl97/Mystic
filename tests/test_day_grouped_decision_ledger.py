import json
import sqlite3

from backend.services.day_grouped_decision_ledger import build_groups, ledger_counts


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE decision_book_tape (
            id INTEGER PRIMARY KEY,
            ts_utc TEXT,
            engine TEXT,
            symbol TEXT,
            selected_action TEXT,
            selection_reason TEXT,
            buy_ev REAL,
            hold_ev REAL,
            model_version TEXT,
            best_bid REAL,
            best_ask REAL,
            mid REAL,
            spread_pct REAL,
            book_source TEXT,
            book_age_sec REAL,
            extras_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE paper_trades (
            symbol TEXT, side TEXT, timestamp TEXT, quantity REAL, price REAL,
            decision_id TEXT, strategy_id TEXT, remaining_position REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE day_decision_records (
            decision_id TEXT, created_at TEXT, symbol TEXT, final_decision TEXT,
            first_hard_block TEXT, detail_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE ai_inference_log (
            strategy_id TEXT, symbol TEXT, ts_utc TEXT, prob_buy REAL, prob_hold REAL,
            prob_sell REAL, feature_version TEXT, feature_dim INTEGER, features_json TEXT
        )
        """
    )
    conn.execute("CREATE TABLE portfolio_engine_rejects (ts TEXT, symbol TEXT, reason TEXT)")
    return conn


def _insert_group(conn, ts, winner, evs, *, book=True):
    extras = {
        "btc_path_ev": evs["BTCUSDT"],
        "eth_path_ev": evs["ETHUSDT"],
        "sol_path_ev": evs["SOLUSDT"],
        "xrp_path_ev": evs["XRPUSDT"],
        "hold_ev": 0.0,
        "path_ev_winner": winner,
        "selected_action": "HOLD" if winner == "HOLD" else f"BUY_{winner}",
        "selected_symbol": "" if winner == "HOLD" else winner,
        "why_selected": "HOLD_WINS" if winner == "HOLD" else "PATH_NET_BEATS_HOLD",
        "path_net_model_id": "day_path_net_v1",
    }
    payload = json.dumps(extras)
    selected = extras["selected_action"]
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD"):
        action = selected if (sym == winner or (sym == "HOLD" and winner == "HOLD")) else "HOLD"
        conn.execute(
            """
            INSERT INTO decision_book_tape (
                ts_utc, engine, symbol, selected_action, selection_reason, buy_ev, hold_ev,
                model_version, best_bid, best_ask, mid, spread_pct, book_source, book_age_sec, extras_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                ts,
                "day",
                sym,
                action,
                extras["why_selected"],
                0.0 if sym == "HOLD" else evs[sym],
                0.0,
                "day_path_net_v1",
                100.0 if book and sym != "HOLD" else None,
                100.1 if book and sym != "HOLD" else None,
                100.05 if book and sym != "HOLD" else None,
                0.001 if book and sym != "HOLD" else None,
                "redis_orderbook" if book and sym != "HOLD" else ("hold" if sym == "HOLD" else "missing"),
                1.0 if book else None,
                payload,
            ),
        )


def test_groups_include_hold_and_do_not_count_reject_as_fill():
    conn = _conn()
    ts = "2026-09-01T12:00:00+00:00"
    _insert_group(conn, ts, "ETHUSDT", {"BTCUSDT": 0.0002, "ETHUSDT": 0.0004, "SOLUSDT": -0.0001, "XRPUSDT": 0.0001})
    conn.execute(
        "INSERT INTO day_decision_records VALUES ('d1',?, 'ETH/USDT','reject','EXECUTION_GATE','{}')",
        (ts,),
    )
    groups = build_groups(conn)
    assert len(groups) == 1
    assert {c.symbol for c in groups[0].candidates} == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "HOLD"}
    eth = next(c for c in groups[0].candidates if c.symbol == "ETHUSDT")
    btc = next(c for c in groups[0].candidates if c.symbol == "BTCUSDT")
    hold = next(c for c in groups[0].candidates if c.symbol == "HOLD")
    assert eth.outcome_class == "blocked_after_ranking"
    assert btc.outcome_class == "ranking_loser"
    assert hold.outcome_class == "hold"
    assert hold.path_ev == 0.0
    counts = ledger_counts(groups)
    assert counts["outcome_classes"]["selected_execute"] == 0
    assert counts["outcome_classes"]["ranking_loser"] == 3
    assert counts["outcome_classes"]["blocked_after_ranking"] == 1
    assert counts["outcome_classes"].get("terminal_fill_failure", 0) == 0


def test_selected_without_reject_or_order_is_no_order_match():
    conn = _conn()
    ts = "2026-09-01T13:00:00+00:00"
    _insert_group(conn, ts, "ETHUSDT", {"BTCUSDT": 0.0, "ETHUSDT": 0.0004, "SOLUSDT": 0.0, "XRPUSDT": 0.0})
    groups = build_groups(conn)
    eth = next(c for c in groups[0].candidates if c.symbol == "ETHUSDT")
    assert eth.outcome_class == "no_order_match"


def test_selected_execute_requires_actual_fill():
    conn = _conn()
    ts = "2026-09-01T12:00:00+00:00"
    _insert_group(conn, ts, "ETHUSDT", {"BTCUSDT": 0.0, "ETHUSDT": 0.0004, "SOLUSDT": 0.0, "XRPUSDT": 0.0})
    conn.execute(
        "INSERT INTO paper_trades VALUES ('ETH/USDT','BUY',?,0.9,2410.0,'day_eth','day',0.9)",
        (ts,),
    )
    groups = build_groups(conn)
    eth = next(c for c in groups[0].candidates if c.symbol == "ETHUSDT")
    assert eth.outcome_class == "selected_execute"


def test_closed_fill_is_execute_not_partial():
    conn = _conn()
    ts = "2026-09-01T12:00:00+00:00"
    _insert_group(conn, ts, "ETHUSDT", {"BTCUSDT": 0.0, "ETHUSDT": 0.0004, "SOLUSDT": 0.0, "XRPUSDT": 0.0})
    conn.execute(
        "INSERT INTO paper_trades VALUES ('ETH/USDT','BUY',?,0.9,2410.0,'day_eth','day',0.0)",
        (ts,),
    )
    groups = build_groups(conn)
    eth = next(c for c in groups[0].candidates if c.symbol == "ETHUSDT")
    assert eth.outcome_class == "selected_execute"


def test_path_ev_comes_from_tape_extras_not_row_only():
    conn = _conn()
    ts = "2026-09-01T12:00:00+00:00"
    evs = {"BTCUSDT": 0.001, "ETHUSDT": -0.002, "SOLUSDT": 0.0, "XRPUSDT": 0.0}
    _insert_group(conn, ts, "BTCUSDT", evs)
    groups = build_groups(conn)
    btc = next(c for c in groups[0].candidates if c.symbol == "BTCUSDT")
    eth = next(c for c in groups[0].candidates if c.symbol == "ETHUSDT")
    assert btc.path_ev == 0.001
    assert eth.path_ev == -0.002
    assert btc.field_authority["path_ev"] == "exact_tape_extras"
