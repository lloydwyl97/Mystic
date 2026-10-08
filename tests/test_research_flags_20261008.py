"""Research switches default off and are not live entry inputs."""

from __future__ import annotations

import inspect

from backend.config.research_flags import cross_venue_research_enabled, policy_research_enabled
from backend.services import adaptive_learning as al
from backend.services.live_market_data import LiveMarketDataService
from backend.services.portfolio_engine_integration import PortfolioEngineIntegration


def test_research_switches_default_off(monkeypatch):
    monkeypatch.delenv("POLICY_RESEARCH_ENABLED", raising=False)
    monkeypatch.delenv("CROSS_VENUE_RESEARCH_ENABLED", raising=False)
    assert policy_research_enabled() is False
    assert cross_venue_research_enabled() is False


def test_research_switches_are_explicit(monkeypatch):
    monkeypatch.setenv("POLICY_RESEARCH_ENABLED", "true")
    monkeypatch.setenv("CROSS_VENUE_RESEARCH_ENABLED", "on")
    assert policy_research_enabled() is True
    assert cross_venue_research_enabled() is True


def test_live_entry_does_not_read_the_research_switches():
    entry = inspect.getsource(al.day_net_expectancy)
    fund = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    assert "POLICY_RESEARCH_ENABLED" not in entry
    assert "CROSS_VENUE_RESEARCH_ENABLED" not in entry
    assert "policy_research_enabled" not in fund
    assert "external_discovery" not in fund
    start = inspect.getsource(LiveMarketDataService.start)
    assert "cross_venue_research_enabled()" in start
