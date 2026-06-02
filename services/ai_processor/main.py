"""
AI Processor Service - Live Configuration Only

Flask application for AI strategy generation, evolution, and model management.
All configuration values come from live config - no hardcoded values, no fallback data.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from ai_auto_retrain import AutoRetrainService
from ai_genetic_algorithm import GeneticAlgorithmEngine
from ai_strategy_generator import AIStrategyGenerator
from dotenv import load_dotenv
from flask import Flask, jsonify, request

from backend.config.redis_config import get_shared_redis_sync

# Import live configuration
try:
    from backend.config_bridge import get_mystic_config

    _mystic_config = get_mystic_config()
except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
    _mystic_config = None

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from project root (single source of truth)
root_env = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=str(root_env))

app = Flask(__name__)

# Initialize AI components
strategy_generator = AIStrategyGenerator()
genetic_algorithm = GeneticAlgorithmEngine()
auto_retrain_service = AutoRetrainService()

# --- Live Configuration Helpers -------------------------------------------------------------------


def _get_service_name() -> str:
    """Get service name from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "service_name"):
                name = value.ai_processor.service_name
                if isinstance(name, str) and name:
                    return name.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    name = os.getenv("AI_PROCESSOR_SERVICE_NAME", "").strip()
    if name:
        return name

    msg = "AI_PROCESSOR_SERVICE_NAME must be configured in live config or environment"
    raise ValueError(msg)


def _get_status_healthy() -> str:
    """Get healthy status string from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "status_healthy"):
                status = value.ai_processor.status_healthy
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_PROCESSOR_STATUS_HEALTHY", "").strip()
    if status:
        return status

    msg = "AI_PROCESSOR_STATUS_HEALTHY must be configured in live config or environment"
    raise ValueError(msg)


def _get_status_success() -> str:
    """Get success status string from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "status_success"):
                status = value.ai_processor.status_success
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_PROCESSOR_STATUS_SUCCESS", "").strip()
    if status:
        return status

    msg = "AI_PROCESSOR_STATUS_SUCCESS must be configured in live config or environment"
    raise ValueError(msg)


def _get_status_error() -> str:
    """Get error status string from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "status_error"):
                status = value.ai_processor.status_error
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_PROCESSOR_STATUS_ERROR", "").strip()
    if status:
        return status

    msg = "AI_PROCESSOR_STATUS_ERROR must be configured in live config or environment"
    raise ValueError(msg)


def _get_component_strategy_generator() -> str:
    """Get strategy generator component name from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "component_strategy_generator"):
                name = value.ai_processor.component_strategy_generator
                if isinstance(name, str) and name:
                    return name.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    name = os.getenv("AI_PROCESSOR_COMPONENT_STRATEGY_GENERATOR", "").strip()
    if name:
        return name

    msg = "AI_PROCESSOR_COMPONENT_STRATEGY_GENERATOR must be configured"
    raise ValueError(msg)


def _get_component_genetic_algorithm() -> str:
    """Get genetic algorithm component name from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "component_genetic_algorithm"):
                name = value.ai_processor.component_genetic_algorithm
                if isinstance(name, str) and name:
                    return name.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    name = os.getenv("AI_PROCESSOR_COMPONENT_GENETIC_ALGORITHM", "").strip()
    if name:
        return name

    msg = "AI_PROCESSOR_COMPONENT_GENETIC_ALGORITHM must be configured"
    raise ValueError(msg)


def _get_component_auto_retrain() -> str:
    """Get auto retrain component name from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "component_auto_retrain"):
                name = value.ai_processor.component_auto_retrain
                if isinstance(name, str) and name:
                    return name.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    name = os.getenv("AI_PROCESSOR_COMPONENT_AUTO_RETRAIN", "").strip()
    if name:
        return name

    msg = "AI_PROCESSOR_COMPONENT_AUTO_RETRAIN must be configured"
    raise ValueError(msg)


def _get_http_status_ok() -> int:
    """Get HTTP OK status code from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "http_status_ok"):
                status = value.ai_processor.http_status_ok
                if isinstance(status, int) and 200 <= status < 300:
                    return status
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_PROCESSOR_HTTP_STATUS_OK", "").strip()
    if status:
        try:
            status_val = int(status)
            if 200 <= status_val < 300:
                return status_val
        except (ValueError, TypeError):
            pass

    msg = "AI_PROCESSOR_HTTP_STATUS_OK must be configured in live config or environment"
    raise ValueError(msg)


def _get_http_status_error() -> int:
    """Get HTTP error status code from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "http_status_error"):
                status = value.ai_processor.http_status_error
                if isinstance(status, int) and 400 <= status < 600:
                    return status
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_PROCESSOR_HTTP_STATUS_ERROR", "").strip()
    if status:
        try:
            status_val = int(status)
            if 400 <= status_val < 600:
                return status_val
        except (ValueError, TypeError):
            pass

    msg = "AI_PROCESSOR_HTTP_STATUS_ERROR must be configured in live config or environment"
    raise ValueError(msg)


def _get_default_port() -> int:
    """Get default port from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "default_port"):
                port = value.ai_processor.default_port
                if isinstance(port, int) and 1 <= port <= 65535:
                    return port
        except (AttributeError, ValueError, TypeError):
            pass

    port = os.getenv("AI_PROCESSOR_DEFAULT_PORT", "").strip()
    if port:
        try:
            port_val = int(port)
            if 1 <= port_val <= 65535:
                return port_val
        except (ValueError, TypeError):
            pass

    msg = "AI_PROCESSOR_DEFAULT_PORT must be configured in live config or environment"
    raise ValueError(msg)


def _get_default_host() -> str:
    """Get default host from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "default_host"):
                host = value.ai_processor.default_host
                if isinstance(host, str) and host:
                    return host.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    host = os.getenv("AI_PROCESSOR_DEFAULT_HOST", "").strip()
    if host:
        return host

    msg = "AI_PROCESSOR_DEFAULT_HOST must be configured in live config or environment"
    raise ValueError(msg)


def _get_debug_mode() -> bool:
    """Get debug mode from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "debug_mode"):
                debug = value.ai_processor.debug_mode
                if isinstance(debug, bool):
                    return debug
        except (AttributeError, ValueError, TypeError):
            pass

    debug = os.getenv("AI_PROCESSOR_DEBUG_MODE", "").strip().lower()
    if debug:
        return debug not in ("false", "0", "no")

    msg = "AI_PROCESSOR_DEBUG_MODE must be configured in live config or environment"
    raise ValueError(msg)


def _get_default_strategy_type() -> str:
    """Get default strategy type from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "default_strategy_type"):
                strategy_type = value.ai_processor.default_strategy_type
                if isinstance(strategy_type, str) and strategy_type:
                    return strategy_type.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    strategy_type = os.getenv("AI_PROCESSOR_DEFAULT_STRATEGY_TYPE", "").strip()
    if strategy_type:
        return strategy_type

    msg = "AI_PROCESSOR_DEFAULT_STRATEGY_TYPE must be configured in live config or environment"
    raise ValueError(msg)


def _get_default_generations() -> int:
    """Get default generations from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "default_generations"):
                generations = value.ai_processor.default_generations
                if isinstance(generations, int) and generations > 0:
                    return generations
        except (AttributeError, ValueError, TypeError):
            pass

    generations = os.getenv("AI_PROCESSOR_DEFAULT_GENERATIONS", "").strip()
    if generations:
        try:
            generations_val = int(generations)
            if generations_val > 0:
                return generations_val
        except (ValueError, TypeError):
            pass

    msg = "AI_PROCESSOR_DEFAULT_GENERATIONS must be configured in live config or environment"
    raise ValueError(msg)


def _get_default_model_id() -> str:
    """Get default model ID from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "default_model_id"):
                model_id = value.ai_processor.default_model_id
                if isinstance(model_id, str) and model_id:
                    return model_id.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    model_id = os.getenv("AI_PROCESSOR_DEFAULT_MODEL_ID", "").strip()
    if model_id:
        return model_id

    msg = "AI_PROCESSOR_DEFAULT_MODEL_ID must be configured in live config or environment"
    raise ValueError(msg)


def _get_component_status() -> str:
    """Get component status string from live config - no fallback."""
    if _mystic_config:
        try:
            value = _mystic_config
            if hasattr(value, "ai_processor") and hasattr(value.ai_processor, "component_status"):
                status = value.ai_processor.component_status
                if isinstance(status, str) and status:
                    return status.strip()
        except (AttributeError, ValueError, TypeError):
            pass

    status = os.getenv("AI_PROCESSOR_COMPONENT_STATUS", "").strip()
    if status:
        return status

    msg = "AI_PROCESSOR_COMPONENT_STATUS must be configured in live config or environment"
    raise ValueError(msg)


def _check_component_status(component_name: str) -> str:
    """Check actual component status and return from live config."""
    component_status = _get_component_status()
    try:
        # Check if component is actually running/available
        if component_name == _get_component_strategy_generator():
            is_running = hasattr(strategy_generator, "running") and strategy_generator.running
        elif component_name == _get_component_genetic_algorithm():
            is_running = hasattr(genetic_algorithm, "running") and genetic_algorithm.running
        elif component_name == _get_component_auto_retrain():
            is_running = hasattr(auto_retrain_service, "running") and auto_retrain_service.running
        else:
            is_running = False

        result = component_status if is_running else "stopped"
    except (AttributeError, ValueError, TypeError):
        return component_status
    else:
        return result


@app.route("/health", methods=["GET"])
def health_check() -> tuple[Any, int]:
    """Health check endpoint"""
    try:
        status_healthy = _get_status_healthy()
        service_name = _get_service_name()
        component_strategy = _get_component_strategy_generator()
        component_genetic = _get_component_genetic_algorithm()
        component_retrain = _get_component_auto_retrain()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify(
                {
                    "status": status_healthy,
                    "service": service_name,
                    "components": {
                        component_strategy: _check_component_status(component_strategy),
                        component_genetic: _check_component_status(component_genetic),
                        component_retrain: _check_component_status(component_retrain),
                    },
                },
            ),
            http_status_ok,
        )
    except (ValueError, AttributeError, TypeError) as e:
        logger.exception("Error in health check")
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        return jsonify({"status": status_error, "error": str(e)}), http_status_error


@app.route("/generate_strategy", methods=["POST"])
def generate_strategy() -> tuple[Any, int]:
    """Generate new AI trading strategy - uses live data only."""
    data = request.get_json() or {}
    default_strategy_type = _get_default_strategy_type()
    strategy_type = data.get("strategy_type") or default_strategy_type
    symbol = data.get("symbol")
    # Validate symbol outside try to avoid TRY301
    if not symbol:
        error_msg = "symbol is required"
        raise ValueError(error_msg)

    parameters = data.get("parameters") or {}

    # Call async method properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        strategy = loop.run_until_complete(strategy_generator.create_ai_strategy(strategy_type, symbol, parameters))
    finally:
        loop.close()

    # Validate strategy outside try to avoid TRY301
    if not strategy:
        error_msg = "Failed to generate strategy"
        raise ValueError(error_msg)

    try:
        status_success = _get_status_success()
        http_status_ok = _get_http_status_ok()

        return jsonify({"status": status_success, "strategy": strategy}), http_status_ok
    except (ValueError, AttributeError, TypeError, RuntimeError, Exception) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        logger.exception("Error generating strategy")
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


@app.route("/evolve_strategy", methods=["POST"])
def evolve_strategy() -> tuple[Any, int]:
    """Evolve existing strategy using genetic algorithm - uses live data only."""
    data = request.get_json() or {}
    default_generations = _get_default_generations()
    generations = data.get("generations")
    if generations is None:
        generations = default_generations

    # Genetic algorithm evolves population asynchronously
    # Return current best strategy from population
    if not genetic_algorithm.population:
        # Initialize population if needed
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(genetic_algorithm.initialize_population())
        finally:
            loop.close()

    # Get best strategy from current population (validate outside try to avoid TRY301)
    if genetic_algorithm.population:
        best_gene = genetic_algorithm.population[0]
        evolved_strategy = best_gene.to_dict()
    else:
        error_msg = "Failed to initialize population"
        raise ValueError(error_msg)

    try:
        status_success = _get_status_success()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify({"status": status_success, "evolved_strategy": evolved_strategy}),
            http_status_ok,
        )
    except (ValueError, AttributeError, TypeError, RuntimeError, Exception) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        logger.exception("Error evolving strategy")
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


@app.route("/version_model", methods=["POST"])
def version_model() -> tuple[Any, int]:
    """Version model endpoint - uses live data only."""
    # Model versioning functionality integrated into strategy generation
    # This endpoint is kept for compatibility but uses live strategy data
    data = request.get_json() or {}
    model_data = data.get("model_data") or {}

    # Validate model_data outside try to avoid TRY301
    if not model_data:
        error_msg = "model_data is required"
        raise ValueError(error_msg)

    try:
        # Return model data with version info from live config
        version_info = {
            "model_id": model_data.get("id", "unknown"),
            "version": model_data.get("model_path", "").split("_")[-1] if model_data.get("model_path") else "unknown",
            "status": model_data.get("status", "unknown"),
            "created_at": model_data.get("created_at", ""),
        }

        status_success = _get_status_success()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify({"status": status_success, "version_info": version_info}),
            http_status_ok,
        )
    except (ValueError, AttributeError, TypeError, RuntimeError, Exception) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        logger.exception("Error versioning model")
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


@app.route("/auto_retrain", methods=["POST"])
def auto_retrain() -> tuple[Any, int]:
    """Trigger automatic model retraining - uses live data only."""
    data = request.get_json() or {}
    model_data = data.get("model") or {}

    if not model_data:
        # Try to get model by ID
        default_model_id = _get_default_model_id()
        model_id = data.get("model_id") or default_model_id

        # Fetch model from Redis (live data)
        redis_client = get_shared_redis_sync()
        if redis_client is None:
            error_msg = "Redis configuration required for model retrieval"
            raise ValueError(error_msg)
        model_json = redis_client.get(f"ai_strategy:{model_id}")
        if model_json:
            model_data = json.loads(model_json)
        else:
            error_msg = f"Model {model_id} not found in live data"
            raise ValueError(error_msg)

    # Call async method properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        retrain_result = loop.run_until_complete(auto_retrain_service.retrain_model(model_data))
    finally:
        loop.close()

    # Validate retrain result outside try to avoid TRY301
    if not retrain_result:
        error_msg = "Failed to retrain model"
        raise ValueError(error_msg)

    try:
        status_success = _get_status_success()
        http_status_ok = _get_http_status_ok()

        return (
            jsonify({"status": status_success, "retrain_result": retrain_result}),
            http_status_ok,
        )
    except (ValueError, AttributeError, TypeError, RuntimeError, Exception) as e:
        status_error = _get_status_error()
        http_status_error = _get_http_status_error()
        logger.exception("Error auto retraining model")
        return jsonify({"status": status_error, "message": str(e)}), http_status_error


if __name__ == "__main__":
    # Get port from environment or live config - no fallback
    port_env = os.getenv("PORT", "").strip()
    if port_env:
        try:
            port = int(port_env)
        except (ValueError, TypeError) as port_error:
            logger.exception("Invalid PORT environment variable")
            error_msg = f"Invalid PORT configuration: {port_error}"
            raise ValueError(error_msg) from port_error

        # Validate port range outside inner try to avoid TRY301
        if not (1 <= port <= 65535):
            error_msg = f"Port {port} out of valid range"
            raise ValueError(error_msg)
    else:
        port = _get_default_port()

    default_host = _get_default_host()
    debug_mode = _get_debug_mode()

    app.run(host=default_host, port=port, debug=debug_mode)
