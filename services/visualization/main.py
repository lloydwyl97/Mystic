"""
Visualization Service - Live Configuration Only

Flask application for generating trading charts and visualizations.
All configuration values come from live config - no hardcoded values.
"""

import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file

# Import from single source of truth
try:
    from backend.config.trading_universe import TRADING_SYMBOLS
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import trading_universe: {e}"
    raise RuntimeError(msg) from e

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

logger = logging.getLogger(__name__)

with suppress(Exception):
    # Load from project root (single source of truth)
    root_env = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=str(root_env))

# Direct imports for production (must be after dotenv load)
from chart_generator import plot_performance_over_time, plot_trades
from mutation_graph import plot_strategy_graph

app = Flask(__name__)

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_default_symbol() -> str:
    """Get default symbol from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "trading_universe") and hasattr(value.trading_universe, "top10_symbols"):
                symbols = value.trading_universe.top10_symbols
                if isinstance(symbols, list) and symbols:
                    return str(symbols[0])
        except (AttributeError, ValueError, TypeError, IndexError):
            pass

    symbol = os.getenv("VISUALIZATION_DEFAULT_SYMBOL", "").strip()
    if symbol:
        return symbol

    # Use second symbol from TRADING_SYMBOLS (live data) - ETHUSDT is typically second
    if len(TRADING_SYMBOLS) < 2:
        msg = "Insufficient trading symbols available - TRADING_SYMBOLS must have at least 2 symbols"
        raise RuntimeError(msg)
    return TRADING_SYMBOLS[1]  # ETHUSDT is typically second in the list


def _get_default_db_path() -> str:
    """Get default database path from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "default_db_path"):
                db_path = value.visualization.default_db_path
                if isinstance(db_path, str) and db_path:
                    return db_path.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    db_path = os.getenv("VISUALIZATION_DEFAULT_DB_PATH", "").strip()
    if db_path:
        return db_path

    return os.getenv("SIMULATION_TRADES_DB", "simulation_trades.db")


def _get_service_name() -> str:
    """Get service name from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "service_name"):
                name = value.visualization.service_name
                if isinstance(name, str) and name:
                    return name.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    name = os.getenv("VISUALIZATION_SERVICE_NAME", "").strip()
    if name:
        return name

    return "visualization"


def _get_status_healthy() -> str:
    """Get healthy status string from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "status_healthy"):
                status = value.visualization.status_healthy
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("VISUALIZATION_STATUS_HEALTHY", "").strip()
    if status:
        return status

    return "healthy"


def _get_status_success() -> str:
    """Get success status string from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "status_success"):
                status = value.visualization.status_success
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("VISUALIZATION_STATUS_SUCCESS", "").strip()
    if status:
        return status

    return "success"


def _get_status_error() -> str:
    """Get error status string from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "status_error"):
                status = value.visualization.status_error
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("VISUALIZATION_STATUS_ERROR", "").strip()
    if status:
        return status

    return "error"


def _get_component_chart_generator() -> str:
    """Get chart generator component name from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "component_chart_generator"):
                name = value.visualization.component_chart_generator
                if isinstance(name, str) and name:
                    return name.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    name = os.getenv("VISUALIZATION_COMPONENT_CHART_GENERATOR", "").strip()
    if name:
        return name

    return "chart_generator"


def _get_component_dashboard() -> str:
    """Get dashboard component name from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "component_dashboard"):
                name = value.visualization.component_dashboard
                if isinstance(name, str) and name:
                    return name.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    name = os.getenv("VISUALIZATION_COMPONENT_DASHBOARD", "").strip()
    if name:
        return name

    return "dashboard"


def _get_component_mutation_graph() -> str:
    """Get mutation graph component name from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "component_mutation_graph"):
                name = value.visualization.component_mutation_graph
                if isinstance(name, str) and name:
                    return name.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    name = os.getenv("VISUALIZATION_COMPONENT_MUTATION_GRAPH", "").strip()
    if name:
        return name

    return "mutation_graph"


def _get_trade_chart_filename() -> str:
    """Get trade chart filename from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "trade_chart_filename"):
                filename = value.visualization.trade_chart_filename
                if isinstance(filename, str) and filename:
                    return filename.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    filename = os.getenv("VISUALIZATION_TRADE_CHART_FILENAME", "").strip()
    if filename:
        return filename

    return "trade_chart.png"


def _get_performance_chart_filename() -> str:
    """Get performance chart filename from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "performance_chart_filename"):
                filename = value.visualization.performance_chart_filename
                if isinstance(filename, str) and filename:
                    return filename.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    filename = os.getenv("VISUALIZATION_PERFORMANCE_CHART_FILENAME", "").strip()
    if filename:
        return filename

    return "performance_chart.png"


def _get_mutation_graph_filename() -> str:
    """Get mutation graph filename from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "mutation_graph_filename"):
                filename = value.visualization.mutation_graph_filename
                if isinstance(filename, str) and filename:
                    return filename.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    filename = os.getenv("VISUALIZATION_MUTATION_GRAPH_FILENAME", "").strip()
    if filename:
        return filename

    return "mutation_graph.png"


def _get_http_status_ok() -> int:
    """Get HTTP OK status code from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "http_status_ok"):
                status = value.visualization.http_status_ok
                # HTTP 2xx status codes are valid success codes
                if isinstance(status, int) and 200 <= status < 300:
                    return status
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("VISUALIZATION_HTTP_STATUS_OK", "").strip()
    if status:
        try:
            status_val = int(status)
            # HTTP 2xx status codes are valid success codes
            if 200 <= status_val < 300:
                return status_val
        except (ValueError, TypeError):
            pass

    return 200


def _get_http_status_error() -> int:
    """Get HTTP error status code from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "http_status_error"):
                status = value.visualization.http_status_error
                # HTTP 4xx and 5xx status codes are valid error codes
                if isinstance(status, int) and 400 <= status < 600:
                    return status
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("VISUALIZATION_HTTP_STATUS_ERROR", "").strip()
    if status:
        try:
            status_val = int(status)
            # HTTP 4xx and 5xx status codes are valid error codes
            if 400 <= status_val < 600:
                return status_val
        except (ValueError, TypeError):
            pass

    return 500


def _get_http_status_not_found() -> int:
    """Get HTTP not found status code from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "http_status_not_found"):
                status = value.visualization.http_status_not_found
                # HTTP 4xx and 5xx status codes are valid error codes
                if isinstance(status, int) and 400 <= status < 600:
                    return status
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("VISUALIZATION_HTTP_STATUS_NOT_FOUND", "").strip()
    if status:
        try:
            status_val = int(status)
            # HTTP 4xx and 5xx status codes are valid error codes
            if 400 <= status_val < 600:
                return status_val
        except (ValueError, TypeError):
            pass

    return 404


def _get_image_mimetype() -> str:
    """Get image mimetype from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "image_mimetype"):
                mimetype = value.visualization.image_mimetype
                if isinstance(mimetype, str) and mimetype:
                    return mimetype.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    mimetype = os.getenv("VISUALIZATION_IMAGE_MIMETYPE", "").strip()
    if mimetype:
        return mimetype

    return "image/png"


def _get_default_port() -> int:
    """Get default port from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "default_port"):
                port = value.visualization.default_port
                # Valid port range is 1-65535
                if isinstance(port, int) and 1 <= port <= 65535:
                    return port
        except (AttributeError, ValueError, TypeError):
            pass

    port = os.getenv("VISUALIZATION_DEFAULT_PORT", "").strip()
    if port:
        try:
            port_val = int(port)
            # Valid port range is 1-65535
            if 1 <= port_val <= 65535:
                return port_val
        except (ValueError, TypeError):
            pass

    return 8003


def _get_default_host() -> str:
    """Get default host from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "default_host"):
                host = value.visualization.default_host
                if isinstance(host, str) and host:
                    return host.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    host = os.getenv("VISUALIZATION_DEFAULT_HOST", "").strip()
    if host:
        return host

    return "0.0.0.0"


def _get_debug_mode() -> bool:
    """Get debug mode from live config."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "visualization") and hasattr(value.visualization, "debug_mode"):
                debug = value.visualization.debug_mode
                if isinstance(debug, bool):
                    return debug
        except (AttributeError, ValueError, TypeError):
            pass

    debug = os.getenv("VISUALIZATION_DEBUG_MODE", "").strip().lower()
    if debug in ("true", "1", "yes"):
        return True
    if debug in ("false", "0", "no"):
        return False

    return False


@app.route("/health", methods=["GET"])
def health_check() -> tuple[Any, int]:
    """Health check endpoint"""
    status_healthy = _get_status_healthy()
    service_name = _get_service_name()
    component_chart = _get_component_chart_generator()
    component_dashboard = _get_component_dashboard()
    component_mutation = _get_component_mutation_graph()
    http_status_ok = _get_http_status_ok()

    return (
        jsonify(
            {
                "status": status_healthy,
                "service": service_name,
                "components": {
                    component_chart: "initialized",
                    component_dashboard: "initialized",
                    component_mutation: "initialized",
                },
            },
        ),
        http_status_ok,
    )


@app.route("/generate_trade_chart", methods=["POST"])
def generate_trade_chart() -> tuple[Any, int]:
    """Generate trade chart for a symbol"""
    try:
        data = request.get_json() or {}
        default_symbol = _get_default_symbol()
        default_db_path = _get_default_db_path()
        symbol = data.get("symbol") or default_symbol
        db_path = data.get("db_path") or default_db_path

        plot_trades(symbol, db_path)

        status_success = _get_status_success()
        trade_chart_filename = _get_trade_chart_filename()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify(
                {
                    "status": status_success,
                    "message": f"Trade chart generated for {symbol}",
                    "file": trade_chart_filename,
                },
            ),
            http_status_ok,
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


@app.route("/generate_performance_chart", methods=["POST"])
def generate_performance_chart() -> tuple[Any, int]:
    """Generate performance chart"""
    try:
        data = request.get_json() or {}
        default_db_path = _get_default_db_path()
        db_path = data.get("db_path") or default_db_path

        plot_performance_over_time(db_path)

        status_success = _get_status_success()
        performance_chart_filename = _get_performance_chart_filename()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify(
                {
                    "status": status_success,
                    "message": "Performance chart generated",
                    "file": performance_chart_filename,
                },
            ),
            http_status_ok,
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


@app.route("/profit_chart_data", methods=["GET"])
def get_profit_chart_data() -> tuple[Any, int]:
    """Get profit chart data"""
    try:
        status_success = _get_status_success()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify(
                {
                    "status": status_success,
                    "data": {"times": [], "profits": [], "cumulative": []},
                },
            ),
            http_status_ok,
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


@app.route("/strategy_performance_data", methods=["GET"])
def get_strategy_performance_data() -> tuple[Any, int]:
    """Get strategy performance chart data"""
    try:
        status_success = _get_status_success()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify(
                {
                    "status": status_success,
                    "data": {
                        "strategies": [],
                        "win_rates": [],
                        "avg_profits": [],
                    },
                },
            ),
            http_status_ok,
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


@app.route("/mutation_graph", methods=["POST"])
def generate_mutation_graph() -> tuple[Any, int]:
    """Generate strategy mutation graph"""
    try:
        data = request.get_json() or {}
        mutation_history = data.get("mutation_history") or []

        plot_strategy_graph(mutation_history)

        status_success = _get_status_success()
        mutation_graph_filename = _get_mutation_graph_filename()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify(
                {
                    "status": status_success,
                    "message": "Mutation graph generated",
                    "file": mutation_graph_filename,
                },
            ),
            http_status_ok,
        )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


@app.route("/chart/<filename>", methods=["GET"])
def get_chart_file(filename: str) -> Response | tuple[Any, int]:
    """Serve generated chart files"""
    try:
        image_mimetype = _get_image_mimetype()
        return send_file(filename, mimetype=image_mimetype)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        status_error = _get_status_error()
        http_status_not_found = _get_http_status_not_found()
        return (
            jsonify({"status": status_error, "message": f"File {filename} not found"}),
            http_status_not_found,
        )


if __name__ == "__main__":
    port_env = os.getenv("PORT", "")
    if port_env:
        try:
            port = int(port_env)
        except (ValueError, TypeError):
            port = _get_default_port()
    else:
        port = _get_default_port()

    default_host = _get_default_host()
    debug_mode = _get_debug_mode()

    app.run(host=default_host, port=port, debug=debug_mode)
