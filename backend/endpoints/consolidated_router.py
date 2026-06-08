"""Consolidated router with production-only live endpoints."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from fastapi import APIRouter

logger = logging.getLogger(__name__)

# Create main consolidated router
consolidated_router = APIRouter()

# Track (module_path, prefix) to avoid accidental double-includes
_included: set[tuple[str, str | None]] = set()


def _include_if_available(
    import_path: str,
    attr: str = "router",
    prefixes: Iterable[str | None] = (None,),
    display_name: str = "",
    critical: bool = False,
) -> None:
    """Best-effort import and include of a router with optional prefix aliases.

    - critical=True logs as error if missing; otherwise warning
    - prefixes can include None for root-mount and strings like "/api"
    - Creates separate router instances for multiple prefixes to avoid FastAPI deduplication
    """
    name = display_name or import_path
    try:
        module = __import__(import_path, fromlist=[attr])
        source_router = getattr(module, attr, None)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError, ImportError, ModuleNotFoundError) as e:
        msg = f"Error importing router from {import_path}: {e}"
        if critical:
            logger.exception(f"[ERROR] {msg}")
            raise
        else:
            # Check if this is a canonical_cache availability issue (non-critical, router will work at runtime)
            if "canonical_cache not available" in str(e):
                logger.info(f"[SKIP] Router {import_path} skipped due to canonical_cache not initialized yet (will work at runtime)")
            else:
                logger.warning(f"[SKIP] {msg}")
            return

    # Validate router exists outside try to avoid TRY301
    if source_router is None:
        msg = f"Module {import_path} missing attribute '{attr}'"
        raise AttributeError(msg)

    try:
        prefix_list = list(prefixes)
        for _idx, p in enumerate(prefix_list):
            key = (import_path, p or "")
            if key in _included:
                continue

            # For multiple prefixes, create separate router instances to avoid FastAPI deduplication
            if len(prefix_list) > 1:
                new_router = APIRouter(tags=getattr(source_router, "tags", None))
                # Copy all routes from source router
                for route in source_router.routes:
                    new_router.routes.append(route)
                consolidated_router.include_router(new_router, prefix=p or "")
            else:
                # Single prefix - use original router directly
                consolidated_router.include_router(source_router, prefix=p or "")

            _included.add(key)
        logger.info(f"[OK] Loaded {name} ({', '.join([p or 'root' for p in prefix_list])})")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        msg = f"Failed to load {name}: {e}"
        if critical:
            logger.exception(f"[ERROR] {msg}")
        else:
            logger.warning(f"[WARN] {msg}")


def load_all_endpoints() -> None:
    """Load strict production endpoint surface only."""
    # Mystic's live production router surface. ``backend.routes.signals`` and the
    # legacy market/exchange routes were retired: they depended on deleted
    # services (signal_service, ai_signal_service) and returned HTTP 500 in
    # paper/live mode. Do not re-add modules that depend on retired services.
    essentials: list[tuple[str, tuple[str | None, ...], str, bool]] = [
        ("backend.endpoints.portfolio_engine_endpoints", (None,), "portfolio engine endpoints", True),
        ("backend.endpoints.live_trading_endpoints", (None,), "live trading endpoints", False),
        ("backend.endpoints.performance_endpoints", (None,), "performance endpoints", False),
        ("backend.endpoints.ai_diagnostics_endpoints", (None,), "ai diagnostics endpoints", False),
        ("backend.endpoints.paper_trading_endpoints", (None,), "paper trading endpoints", False),
        ("backend.endpoints.scalp_status_endpoints", (None,), "scalp status endpoints", False),
        ("backend.routes.orders", (None,), "orders routes", False),
        ("backend.routes.system_health", (None,), "system health routes", True),
    ]
    for module_path, prefixes, name, critical in essentials:
        _include_if_available(module_path, prefixes=prefixes, display_name=name, critical=critical)

    logger.info("[TARGET] CORE ENDPOINTS LOADED: %d routes", len(consolidated_router.routes))


# load_all_endpoints() is now called from app_factory after canonical_cache initialization

# Export the router
router = consolidated_router
