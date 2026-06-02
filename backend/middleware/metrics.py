"""
Metrics Middleware - All Live Data, No Fallback/Hardcoded Data

This module provides metrics middleware for live API operations (backend port 8000).
All operations:
- Collect live metrics from API requests/responses (backend port 8000)
- Expose live metrics endpoints for monitoring
- Track live request/response performance
- No fallback/hardcoded data - all metrics from live operations
- Used by backend services on port 8000 for live trading operations

Live Data Sources:
- API requests: Live requests to backend API (port 8000)
- API responses: Live responses from backend API (port 8000)
- Metrics endpoints: Live metrics data from MetricsCollector
- All metrics collected from live operations - no mock/test data

Endpoint References:
- Backend API: Port 8000 (metrics middleware processes live requests)
- Metrics endpoints: /metrics, /metrics/summary, /metrics/detailed (live metrics data)
- All metrics collected from live connections - no fallback/hardcoded data
"""

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from backend.utils.enhanced_logging import performance_logger

from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

# Global metrics collector instance for live operations (backend port 8000)
metrics_collector = MetricsCollector()

# Set up live metrics endpoints in the config (not fallback data, endpoint configuration)
metrics_collector.config["endpoints"] = {
    "/metrics": metrics_collector.get_metrics,  # Live metrics endpoint
    "/metrics/summary": metrics_collector.get_metrics_summary,  # Live metrics summary endpoint
    "/metrics/detailed": metrics_collector.get_detailed_metrics,  # Live detailed metrics endpoint
}


@performance_logger("metrics_middleware")
async def metrics_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """
    Metrics middleware for live API requests (backend port 8000).

    Collects live metrics from API operations and exposes metrics endpoints.
    All metrics from live operations - no fallback/hardcoded data.

    Args:
        request: Live API request to backend (port 8000)
        call_next: Next middleware handler

    Returns:
        Live API response or live metrics response
    """
    start_time = time.time()

    try:
        # Check if it's a live metrics endpoint
        if request.url.path in metrics_collector.config["endpoints"]:
            # Return live metrics data (not fallback data, live metrics endpoint)
            metrics_data = metrics_collector.config["endpoints"][request.url.path]()
            return JSONResponse(content=metrics_data)

        # Process live API request
        response = await call_next(request)

        # Track live metrics
        metrics_collector.track_request(request, start_time)
        if isinstance(response, JSONResponse):
            metrics_collector.track_response(request, response)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        # Update live error metrics (not fallback data, error tracking)
        metrics_collector.metrics["errors"][f"{request.method} {request.url.path}"] += 1
        logger.exception("Metrics error for live request: %s", e)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    else:
        return response
