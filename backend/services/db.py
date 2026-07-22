"""
Database Engine & Session Utilities

- Backward-compatible helpers:
    - get_engine()
    - get_sessionmaker(engine)

- Enhancements added:
    - Robust DATABASE_URL resolution with sqlite fallback and postgres scheme fix.
    - Configurable pooling & echo via env vars.
    - Safe, idempotent SQLite pragmas (WAL, synchronous, foreign_keys, busy_timeout).
    - Context manager `session_scope()` for commit/rollback safety.
    - Lightweight `db_health()` probe for diagnostics.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

log = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _database_url() -> str:
    """
    Prefer DATABASE_URL; fallback to local SQLite (MYSTIC_DB_PATH or mystic_trading.db).
    Also normalize postgres:// to postgresql:// for SQLAlchemy.
    """
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url:
        # Normalize legacy postgres scheme for SQLAlchemy
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        return url
    configured_path = (os.getenv("MYSTIC_DB_PATH") or "").strip()
    if configured_path:
        db_file = configured_path
    else:
        # Match backend.database_schema.DATABASE_PATH without depending on CWD.
        db_file = str(Path(__file__).resolve().parents[2] / "mystic_trading.db")
    return f"sqlite:///{db_file}"


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        return default


def _engine_kwargs(url: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "echo": _bool_env("ECHO_SQL", False),
        "future": True,
        "pool_pre_ping": True,
    }

    # Pool configuration (ignored by some dialects like SQLite memory/URI)
    kwargs["pool_recycle"] = _int_env("DB_POOL_RECYCLE", 1800)  # seconds
    kwargs["pool_timeout"] = float(os.getenv("DB_POOL_TIMEOUT", "30"))
    # Only set pool_size/max_overflow if provided (avoid warnings on some drivers)
    pool_size = os.getenv("DB_POOL_SIZE")
    max_overflow = os.getenv("DB_MAX_OVERFLOW")
    if pool_size:
        with suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            kwargs["pool_size"] = int(pool_size)
    if max_overflow:
        with suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            kwargs["max_overflow"] = int(max_overflow)

    # SQLite-specific connect args
    if url.startswith("sqlite:///"):
        kwargs["connect_args"] = {"check_same_thread": False}

    return kwargs


def _apply_sqlite_pragmas(engine: Engine) -> None:
    """
    Apply safe SQLite pragmas on each new connection.
    WAL improves concurrency; foreign_keys enforces FKs.
    """
    if not str(engine.url).startswith("sqlite:///"):
        return

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, _):  # type: ignore[no-redef]
        try:
            cur = dbapi_conn.cursor()
            # Journal/WAL & sync policy
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            # Enforce FK constraints
            cur.execute("PRAGMA foreign_keys=ON;")
            # Busy timeout to reduce 'database is locked' errors (ms)
            cur.execute(f"PRAGMA busy_timeout={_int_env('SQLITE_BUSY_TIMEOUT_MS', 5000)};")
            cur.close()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Don't hard-fail on pragmas; just log
            with suppress(ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                log.debug("SQLite pragma setup failed: %s", e)


def get_engine() -> Engine:
    """
    Create (once) and return a SQLAlchemy Engine based on env configuration.
    Backward compatible signature.
    """
    global _engine
    if _engine is not None:
        return _engine

    url = _database_url()
    kwargs = _engine_kwargs(url)
    _engine = create_engine(url, **kwargs)
    _apply_sqlite_pragmas(_engine)
    return _engine


# Session maker state - using dict to avoid global keyword
_session_local_state: dict[str, sessionmaker[Session] | None] = {"instance": None}


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker[Session]:
    """
    Return a configured sessionmaker (singleton per process).
    Backward compatible signature.
    """
    if _session_local_state["instance"] is not None:
        return _session_local_state["instance"]
    eng = engine or get_engine()
    _session_local_state["instance"] = sessionmaker(bind=eng, autoflush=False, autocommit=False)
    return _session_local_state["instance"]


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """
    Context-managed DB session with commit/rollback semantics.

    Example:
        with session_scope() as s:
            s.add(obj)
    """
    SessionLocal = get_sessionmaker(engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        session.rollback()
        raise
    finally:
        session.close()


def db_health() -> dict[str, Any]:
    """
    Lightweight connectivity probe and metadata.
    Does not expose credentials.
    """
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        url_str = str(eng.url)
        # Hide password if present in URL
        if "@" in url_str and "://" in url_str:
            scheme, rest = url_str.split("://", 1)
            if "@" in rest and ":" in rest.split("@", 1)[0]:
                creds, tail = rest.split("@", 1)
                user = creds.split(":", 1)[0]
                url_str = f"{scheme}://{user}:***@{tail}"
        return {
            "ok": True,
            "dialect": eng.dialect.name,
            "driver": getattr(eng.dialect, "driver", ""),
            "url": url_str,
        }
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        return {"ok": False, "error": str(e)}
