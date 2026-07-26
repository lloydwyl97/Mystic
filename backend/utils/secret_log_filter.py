"""Redact secrets (Telegram bot tokens, etc.) from log records."""

from __future__ import annotations

import logging
import re

_TELEGRAM_BOT_RE = re.compile(r"(https?://api\.telegram\.org/bot)([^/\s]+)(/[^\s]*)?")
_BOT_TOKENISH_RE = re.compile(r"\b(\d{6,}:[A-Za-z0-9_-]{20,})\b")


def redact_secrets(text: str) -> str:
    if not text:
        return text
    out = _TELEGRAM_BOT_RE.sub(r"\1***\3", text)
    out = _BOT_TOKENISH_RE.sub("***", out)
    return out


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact_secrets(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: redact_secrets(v) if isinstance(v, str) else v for k, v in record.args.items()}
                elif isinstance(record.args, tuple):
                    record.args = tuple(redact_secrets(a) if isinstance(a, str) else a for a in record.args)
        except Exception:
            pass
        return True


def install_secret_redacting_filter() -> None:
    filt = SecretRedactingFilter()
    root = logging.getLogger()
    if not any(isinstance(f, SecretRedactingFilter) for f in root.filters):
        root.addFilter(filt)
    # httpx logs request URLs at INFO on its own logger
    httpx_logger = logging.getLogger("httpx")
    if not any(isinstance(f, SecretRedactingFilter) for f in httpx_logger.filters):
        httpx_logger.addFilter(filt)
    httpcore_logger = logging.getLogger("httpcore")
    if not any(isinstance(f, SecretRedactingFilter) for f in httpcore_logger.filters):
        httpcore_logger.addFilter(filt)


__all__ = ["SecretRedactingFilter", "install_secret_redacting_filter", "redact_secrets"]
