import hashlib
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def register_entity(entity_name: str, public_key: str, *, preview_len: int = 16) -> str:
    name = " ".join((entity_name or "").split()).strip()
    key = (public_key or "").strip()

    if not name or not key:
        msg = "entity_name and public_key are required"
        raise ValueError(msg)

    if not isinstance(preview_len, int) or preview_len < 1:
        msg = "preview_len must be a positive integer"
        raise ValueError(msg)

    payload = f"{name}|{key}".encode()
    signature = hashlib.sha256(payload).hexdigest()

    ts = datetime.now(timezone.utc).isoformat()
    logger.info(f"[REGISTRY] {ts} Registered: {name} -> ID: {signature[:preview_len]}")
    return signature
