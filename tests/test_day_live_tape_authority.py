from backend.services.day_live_tape_authority import classify_tape, score_live_coin


def _bars(closes: list[float]) -> list[dict]:
    return [{"open": c, "high": c, "low": c, "close": c, "volume": 1.0} for c in closes]


def test_classify_up_down_sideways():
    up = [100.0 + i * 0.2 for i in range(40)]
    down = [108.0 - i * 0.2 for i in range(40)]
    flat = [100.0 + (0.01 if i % 2 == 0 else -0.01) for i in range(40)]
    assert classify_tape(_bars(up))["state"] == "up"
    assert classify_tape(_bars(down))["state"] == "down"
    assert classify_tape(_bars(flat))["state"] == "sideways"


def test_down_tape_holds_even_if_signal_says_buy(monkeypatch):
    down = [108.0 - i * 0.2 for i in range(40)]
    monkeypatch.setattr("backend.services.day_live_tape_authority.load_recent_bars", lambda *_a, **_k: _bars(down))
    row = score_live_coin(
        symbol="BTCUSDT",
        db_path="",
        signal={"side": "buy", "prob_buy": 0.7, "prob_hold": 0.2, "prob_sell": 0.1, "buy_margin": 0.2},
        context={"ctx_market_regime": "trending_up"},
    )
    assert row["tape_state"] == "down"
    assert row["ev"] == 0.0


def test_high_hold_probability_does_not_buy(monkeypatch):
    up = [100.0 + i * 0.2 for i in range(40)]
    monkeypatch.setattr("backend.services.day_live_tape_authority.load_recent_bars", lambda *_a, **_k: _bars(up))
    row = score_live_coin(
        symbol="ETHUSDT",
        db_path="",
        signal={"side": "buy", "prob_buy": 0.08, "prob_hold": 0.92, "prob_sell": 0.0, "buy_margin": -0.8},
        context={"ctx_market_regime": "trending_up"},
    )
    assert row["ev"] == 0.0
    assert "HOLD" in row["why"]
