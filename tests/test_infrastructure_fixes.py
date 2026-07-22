import asyncio
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy.sql.elements import TextClause


def test_db_default_path_and_engine_singleton(monkeypatch):
    from backend import database_schema
    from backend.services import db

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MYSTIC_DB_PATH", raising=False)
    assert db._database_url() == f"sqlite:///{Path(database_schema.DATABASE_PATH).resolve()}"

    db._engine = None
    first = db.get_engine()
    second = db.get_engine()
    try:
        assert first is second
    finally:
        first.dispose()
        db._engine = None


def test_db_explicit_urls_remain_supported(monkeypatch):
    from backend.services import db

    monkeypatch.setenv("MYSTIC_DB_PATH", "relative/custom.db")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db._database_url() == "sqlite:///relative/custom.db"

    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@example/db")
    assert db._database_url() == "postgresql://user:pass@example/db"


def test_live_execution_flag_reads_named_environment_variable(monkeypatch):
    from backend.config import live_test_mode

    monkeypatch.setenv("LIVE_EXECUTION", "true")
    assert live_test_mode._live_execution_flag_value() is True
    monkeypatch.setenv("LIVE_EXECUTION", "false")
    assert live_test_mode._live_execution_flag_value() is False


def test_performance_monitor_uses_async_shared_redis_without_closing(monkeypatch):
    from backend import performance_monitor as module

    class FakeRedis:
        closed = False

        async def get(self, key):
            if key == "paper_trading:stats":
                return '{"total_trades": 3, "win_rate": 0.5}'
            return None

        async def scan_iter(self, **_kwargs):
            yield "paper:position:BTCUSDT"

        async def hgetall(self, _key):
            return {"quantity": "2", "current_price": "10"}

        def close(self):
            self.closed = True
            raise AssertionError("shared Redis client must not be closed")

    redis = FakeRedis()
    monkeypatch.setattr(module, "REDIS_AVAILABLE", True)
    monkeypatch.setattr(module, "get_redis_service", lambda: redis)

    stats = asyncio.run(module.PerformanceMonitor()._get_redis_trading_stats())
    assert stats["total_trades"] == 3
    assert stats["active_positions"] == 1
    assert stats["portfolio_value"] == 20
    assert redis.closed is False


def test_performance_monitor_database_probe_uses_sqlalchemy_text(monkeypatch):
    from backend import performance_monitor as module

    executed = []

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, statement):
            executed.append(statement)

    monkeypatch.setattr(module, "SessionLocal", FakeSession)
    assert asyncio.run(module.PerformanceMonitor()._check_database_connectivity()) is True
    assert isinstance(executed[0], TextClause)


def test_user_stream_starts_with_lazy_trading_client_when_live(monkeypatch):
    from backend.services import binance_user_stream as module

    async def fake_get_http_client():
        return SimpleNamespace()

    async def fake_ensure_listen_key():
        return None

    async def fake_create_task(coro, name):
        coro.close()
        return SimpleNamespace(name=name)

    monkeypatch.setenv("BINANCE_US_API_KEY", "valid-api-key")
    monkeypatch.setenv("BINANCE_US_SECRET_KEY", "valid-secret-key")
    monkeypatch.setattr(module, "is_live_execution_allowed_sync", lambda: True)
    monkeypatch.setattr(module, "get_http_client", fake_get_http_client)
    monkeypatch.setattr(module.task_manager, "create_task", fake_create_task)
    monkeypatch.setattr(module.trading_service, "binance", None)

    worker = module.BinanceUserStreamWorker()
    monkeypatch.setattr(worker, "_ensure_listen_key", fake_ensure_listen_key)
    asyncio.run(worker.start())

    assert worker._running is True
    assert worker._task is not None
    assert worker._keepalive_task is not None


def test_user_stream_remains_disabled_in_paper_mode(monkeypatch):
    from backend.services import binance_user_stream as module

    monkeypatch.setenv("BINANCE_US_API_KEY", "valid-api-key")
    monkeypatch.setenv("BINANCE_US_SECRET_KEY", "valid-secret-key")
    monkeypatch.setattr(module, "is_live_execution_allowed_sync", lambda: False)

    worker = module.BinanceUserStreamWorker()
    asyncio.run(worker.start())

    assert worker._running is False
    assert worker._client is None
