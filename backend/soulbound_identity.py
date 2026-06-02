import hashlib
import uuid


def generate_soul_signature(seed=None):
    unique_str = str(seed) if seed is not None else str(uuid.uuid4())
    hash_id = hashlib.sha256(unique_str.encode("utf-8")).hexdigest()
    return f"SOULBOUND-{hash_id[:16]}"
