# CRITICAL: Force IPv4 BEFORE any other imports (Binance US requirement).
# Shared, idempotent patch — see backend/utils/network_ipv4.py for why this is
# a single bootstrap location instead of a per-module monkey-patch.
from backend.utils.network_ipv4 import ensure_ipv4_only

ensure_ipv4_only()

from backend.app_factory import create_app

# Create the app first
app = create_app()

# Orchestrator will be started manually after app startup

# ServiceManager initializes inside backend.app_factory lifespan to ensure async wiring happens there.
