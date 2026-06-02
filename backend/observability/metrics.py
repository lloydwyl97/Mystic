from __future__ import annotations

from fastapi import APIRouter, Response

# Direct imports for production
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

metrics_router = APIRouter()


@metrics_router.get("/metrics")
def metrics():
    """Prometheus metrics endpoint using default REGISTRY."""
    if REGISTRY is None:
        return Response(
            content=b"# Prometheus metrics not available\n",
            media_type=CONTENT_TYPE_LATEST,
        )

    # Use default REGISTRY (which your counters/histograms were added to)
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
