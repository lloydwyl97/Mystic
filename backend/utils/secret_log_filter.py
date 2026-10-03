"""Redact secrets (Telegram bot tokens, API keys, request signatures) from log records."""

from __future__ import annotations

import logging
import re

_TELEGRAM_BOT_RE = re.compile(r"(https?://api\.telegram\.org/(?:file/)?bot)([^/\s]+)(/[^\s]*)?")
_BOT_TOKENISH_RE = re.compile(r"\b(\d{6,}:[A-Za-z0-9_-]{20,})\b")
_SECRET_QUERY_RE = re.compile(
    r"([?&](?:api_?key|access_token|auth_token|token|key|signature|secret|client_secret|password|listenKey)=)[^&\s\"'#]+",
    re.IGNORECASE,
)


def redact_secrets(text: str) -> str:
    if not text:
        return text
    out = _TELEGRAM_BOT_RE.sub(r"\1***\3", text)
    out = _BOT_TOKENISH_RE.sub("***", out)
    out = _SECRET_QUERY_RE.sub(r"\1***", out)
    return out


def _redact_record(record: logging.LogRecord) -> None:
    """Render the message once and redact it.

    Arguments are frequently non-str objects (httpx passes ``httpx.URL``), so
    redacting only str arguments misses request URLs.
    """
    try:
        message = record.getMessage()
    except Exception:
        return
    redacted = redact_secrets(message)
    if redacted != message:
        record.msg = redacted
        record.args = None


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        _redact_record(record)
        return True


_factory_installed = False


def install_secret_redacting_filter() -> None:
    """Redact every record at creation, whichever logger or handler emits it."""
    global _factory_installed
    if not _factory_installed:
        base_factory = logging.getLogRecordFactory()

        def _factory(*args, **kwargs) -> logging.LogRecord:
            record = base_factory(*args, **kwargs)
            _redact_record(record)
            return record

        logging.setLogRecordFactory(_factory)
        _factory_installed = True
    filt = SecretRedactingFilter()
    for name in ("", "httpx", "httpcore"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, SecretRedactingFilter) for f in lg.filters):
            lg.addFilter(filt)


__all__ = ["SecretRedactingFilter", "install_secret_redacting_filter", "redact_secrets"]
