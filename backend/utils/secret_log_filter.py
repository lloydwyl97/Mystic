"""Redact secrets (Telegram bot tokens, API keys, request signatures) from log records."""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import traceback

_NOT_REDACTED = r"(?!\*\*\*|\[REDACTED_SECRET\])"
_ALREADY_REDACTED = frozenset({"***", "[REDACTED_SECRET]", "[secure]"})

_TELEGRAM_BOT_RE = re.compile(rf"(https?://api\.telegram\.org/(?:file/)?bot){_NOT_REDACTED}([^/\s]+)(/[^\s]*)?")
# requests and urllib3 print the host-less ``/bot<token>/method`` path, where no word boundary precedes the digits.
_BOT_TOKENISH_RE = re.compile(r"(?:(?<=bot)|\b)(\d{6,}:[A-Za-z0-9_-]{20,})")
_SECRET_QUERY_RE = re.compile(
    r"([?&](?:api_?key|access_token|auth_token|token|key|signature|secret|client_secret|password|listenKey)=)"
    rf"{_NOT_REDACTED}[^&\s\"'#]+",
    re.IGNORECASE,
)
_HEADER_SECRET_RE = re.compile(r"(?i)\b((?:authorization|x-mbx-apikey|x-api-key|api-key)\s*[:=]\s*)(?!(?:bearer\s+)?(?:\*\*\*|\[REDACTED_SECRET\]))(?:bearer\s+)?\S+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]{8,}={0,2}")
_SECRET_FIELDS = (
    r"authorization|proxy-authorization|x-mbx-apikey|x-api-key|api-key|apikey|api_key|api_secret|secret|secret_key|"
    r"client_secret|signature|access_token|auth_token|refresh_token|token|bot_token|password|passwd|listenkey|private_key"
)
# dict / JSON / httpx.Headers reprs: {'X-MBX-APIKEY': '...'}, {"signature": "..."}
_QUOTED_FIELD_RE = re.compile(rf"(?i)(?P<key>(?P<kq>['\"])(?:{_SECRET_FIELDS})(?P=kq)\s*:\s*b?)(?P<val>'[^']*'|\"[^\"]*\")")
# keyword reprs: Config(api_key='...'), signature="..."
_KWARG_FIELD_RE = re.compile(rf"(?i)(?P<key>\b(?:{_SECRET_FIELDS})\s*=\s*b?)(?P<val>'[^']*'|\"[^\"]*\")")

_ENV_SECRET_NAME_RE = re.compile(r"TOKEN|SECRET|PASSWORD|PASSPHRASE|API_?KEY|PRIVATE_KEY", re.IGNORECASE)
_ENV_SECRET_MIN_LEN = 12
_ENV_REFRESH_SEC = 300.0
_env_cache: dict[str, object] = {"size": -1, "at": 0.0, "values": ()}


def _configured_secret_values() -> tuple[str, ...]:
    """Credential values present in the process environment, longest first.

    Re-read when the environment grows (``.env`` loaded after import) or every few minutes.
    """
    size = len(os.environ)
    now = time.monotonic()
    if size != _env_cache["size"] or now - float(_env_cache["at"]) > _ENV_REFRESH_SEC:
        try:
            items = list(os.environ.items())
        except RuntimeError:
            return _env_cache["values"]  # type: ignore[return-value]
        values = {v.strip() for k, v in items if _ENV_SECRET_NAME_RE.search(k) and len(v.strip()) >= _ENV_SECRET_MIN_LEN}
        _env_cache.update(size=size, at=now, values=tuple(sorted(values, key=len, reverse=True)))
    return _env_cache["values"]  # type: ignore[return-value]


def _redact(text: str, replacement: str) -> tuple[str, int]:
    count = 0

    def field(match: re.Match[str]) -> str:
        nonlocal count
        val = match.group("val")
        inner = val[1:-1].split()
        if not inner or inner[-1] in _ALREADY_REDACTED:
            return match.group(0)
        count += 1
        return f"{match.group('key')}{val[0]}{replacement}{val[0]}"

    out = text
    for value in _configured_secret_values():
        if value in out:
            count += out.count(value)
            out = out.replace(value, replacement)
    out, n = _TELEGRAM_BOT_RE.subn(lambda m: f"{m.group(1)}{replacement}{m.group(3) or ''}", out)
    count += n
    out, n = _BOT_TOKENISH_RE.subn(replacement, out)
    count += n
    out, n = _SECRET_QUERY_RE.subn(lambda m: f"{m.group(1)}{replacement}", out)
    count += n
    out, n = _HEADER_SECRET_RE.subn(lambda m: f"{m.group(1)}{replacement}", out)
    count += n
    out = _QUOTED_FIELD_RE.sub(field, out)
    out = _KWARG_FIELD_RE.sub(field, out)
    out, n = _BEARER_RE.subn(f"Bearer {replacement}", out)
    count += n
    return out, count


def credential_shape_count(text: str) -> int:
    """How many credential-shaped values a line still contains. Not the values."""
    if not text:
        return 0
    return _redact(text, "[REDACTED_SECRET]")[1]


def redact_secrets(text: str, *, replacement: str = "***") -> str:
    if not text:
        return text
    return _redact(text, replacement)[0]


_TRACEBACK_FORMATTER = logging.Formatter()


def _redact_record(record: logging.LogRecord) -> None:
    """Render the message and any traceback once and redact them.

    Arguments are frequently non-str objects (httpx passes ``httpx.URL``), so
    redacting only str arguments misses request URLs. Exception text is built
    by the handler's formatter after filters run, so it is rendered here.
    """
    try:
        message = record.getMessage()
    except Exception:
        message = None
    if message is not None:
        redacted = redact_secrets(message)
        if redacted != message:
            record.msg = redacted
            record.args = None
    if record.exc_info and not record.exc_text:
        try:
            record.exc_text = _TRACEBACK_FORMATTER.formatException(record.exc_info)
        except Exception:
            pass
    if record.exc_text:
        clean = redact_secrets(record.exc_text)
        if clean != record.exc_text:
            record.exc_text = clean
            # Formatters that re-render exc_info would print the raw traceback again.
            record.exc_info = None
    if record.stack_info:
        record.stack_info = redact_secrets(record.stack_info)


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        _redact_record(record)
        return True


def _redacting_excepthook(exc_type, exc, tb) -> None:
    sys.stderr.write(redact_secrets("".join(traceback.format_exception(exc_type, exc, tb))))


def _redacting_thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else threading.get_ident()
    text = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    stream = sys.stderr
    if stream is not None:
        stream.write(f"Exception in thread {name}:\n{redact_secrets(text)}")
        stream.flush()


_factory_installed = False


def install_secret_redacting_filter() -> None:
    """Redact every record at creation, whichever logger or handler emits it.

    Uncaught exceptions bypass logging and go straight to stderr, which the
    launchers redirect into the log files, so the default hooks are wrapped too.
    """
    global _factory_installed
    if not _factory_installed:
        base_factory = logging.getLogRecordFactory()

        def _factory(*args, **kwargs) -> logging.LogRecord:
            record = base_factory(*args, **kwargs)
            _redact_record(record)
            return record

        logging.setLogRecordFactory(_factory)
        _factory_installed = True
    if sys.excepthook is sys.__excepthook__:
        sys.excepthook = _redacting_excepthook
    if threading.excepthook is threading.__excepthook__:
        threading.excepthook = _redacting_thread_excepthook
    filt = SecretRedactingFilter()
    for name in ("", "httpx", "httpcore"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, SecretRedactingFilter) for f in lg.filters):
            lg.addFilter(filt)


__all__ = ["SecretRedactingFilter", "credential_shape_count", "install_secret_redacting_filter", "redact_secrets"]
