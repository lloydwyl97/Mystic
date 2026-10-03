"""Replace credential-shaped values already written into log files.

The replacement is the fixed marker ``[REDACTED_SECRET]``. SQLite files are
never rewritten. Compressed logs are replaced only after the new archive
opens and the marker file no longer contains a credential-shaped value.
"""

from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path

from backend.utils.secret_log_filter import credential_shape_count, redact_secrets

MARKER = "[REDACTED_SECRET]"
_SQLITE_MAGIC = b"SQLite format 3\x00"


def credential_shapes(text: str) -> int:
    return credential_shape_count(text)


def sanitize_log_text(text: str) -> str:
    return redact_secrets(text, replacement=MARKER)


def sanitize_log_file(path: Path) -> dict[str, int | str]:
    """Rewrite one log. Returns counts; never returns the secret text."""
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(_SQLITE_MAGIC):
        return {"path": str(path), "status": "skipped_sqlite", "before": 0, "after": 0}
    compressed = path.name.endswith(".gz")
    text = gzip.decompress(raw).decode("utf-8", errors="replace") if compressed else raw.decode("utf-8", errors="replace")
    before = credential_shapes(text)
    cleaned = sanitize_log_text(text)
    after = credential_shapes(cleaned)
    if cleaned == text:
        return {"path": str(path), "status": "unchanged", "before": before, "after": after}
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if compressed:
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                fh.write(cleaned)
            with gzip.open(tmp, "rt", encoding="utf-8") as fh:
                verified = fh.read()
        else:
            tmp.write_text(cleaned, encoding="utf-8")
            verified = tmp.read_text(encoding="utf-8")
        if credential_shapes(verified):
            raise RuntimeError(f"sanitized log still contains a credential-shaped value: {path.name}")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"path": str(path), "status": "sanitized", "before": before, "after": credential_shapes(verified)}
