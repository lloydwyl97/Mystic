from __future__ import annotations

import logging
import threading
import time

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# Lazy imports for optional quantum computing libraries (may not be available in all deployments)
try:
    import cirq
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    cirq = None  # type: ignore[assignment, misc]

try:
    import pennylane as qml
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    qml = None  # type: ignore[assignment, misc]

try:
    from qiskit import Aer, BasicAer, QuantumCircuit, transpile
    from qiskit import execute as q_execute
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    Aer = None  # type: ignore[assignment, misc]
    BasicAer = None  # type: ignore[assignment, misc]
    QuantumCircuit = None  # type: ignore[assignment, misc]
    q_execute = None  # type: ignore[assignment, misc]
    transpile = None  # type: ignore[assignment, misc]

try:
    from qiskit_aer import Aer as AerModern
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError):
    AerModern = None  # type: ignore[assignment, misc]

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e


def _to_ccxt_symbol(base: str, quote: str) -> str:
    """Return CCXT-style symbol BASE/QUOTE."""
    return f"{base}/{quote}"


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quantum", tags=["Quantum Computing"])

# ---------------------------
# Metrics (Prometheus-style)
# ---------------------------

_metrics_lock = threading.Lock()
_metrics: dict[str, float] = {
    "requests_total": 0.0,
    "qiskit_execute_total": 0.0,
    "cirq_optimize_total": 0.0,
    "pennylane_ml_total": 0.0,
    "qiskit_execute_seconds_total": 0.0,
    "cirq_optimize_seconds_total": 0.0,
    "pennylane_ml_seconds_total": 0.0,
}


def _inc_metric(name: str, value: float = 1.0) -> None:
    with _metrics_lock:
        _metrics[name] = _metrics.get(name, 0.0) + value


def _format_prometheus() -> str:
    # Minimal Prometheus exposition format
    lines: list[str] = []
    with _metrics_lock:
        for k, v in _metrics.items():
            lines.append(f"# TYPE {k} counter")
            # Prometheus counters should be non-decreasing
            total = 0.0 if v < 0 else v
            lines.append(f"{k} {total:.6f}")
    return "\n".join(lines) + "\n"


# ---------------------------
# Schemas
# ---------------------------


class QuantumExecuteRequest(BaseModel):
    num_qubits: int = 2
    shots: int = 1024


class QuantumExecuteResponse(BaseModel):
    result: dict[str, int]
    num_qubits: int
    shots: int


class QuantumOptimizeRequest(BaseModel):
    num_qubits: int = 2
    depth: int = 4


class QuantumOptimizeResponse(BaseModel):
    circuit: str
    num_qubits: int
    depth: int


class QuantumMLRequest(BaseModel):
    num_qubits: int = 2
    layers: int = 2


class QuantumMLResponse(BaseModel):
    result: float
    num_qubits: int
    layers: int


# ---------------------------
# Qiskit
# ---------------------------


@router.get(
    "/qiskit/health",
    summary="Qiskit Service Health",
    description="Check the health status of the Qiskit quantum service",
    response_model=dict[str, str],
)
async def qiskit_health():
    _inc_metric("requests_total")
    return {"status": "ok", "service": "qiskit", "version": "1.0.0"}


@router.get(
    "/qiskit/metrics",
    summary="Qiskit Service Metrics",
    description="Get Prometheus metrics from the Qiskit quantum service",
)
async def qiskit_metrics():
    _inc_metric("requests_total")
    data = _format_prometheus()
    return PlainTextResponse(content=data, media_type="text/plain")


@router.post(
    "/qiskit/execute",
    summary="Execute Quantum Circuit",
    description="Execute a quantum circuit using Qiskit",
    response_model=QuantumExecuteResponse,
)
async def qiskit_execute(request: QuantumExecuteRequest):
    _inc_metric("requests_total")
    start = time.perf_counter()

    # Validate inputs
    num_qubits = max(1, min(int(request.num_qubits), 8))
    shots = max(1, min(int(request.shots), 4096))

    # Use optional quantum libraries if available
    try:
        if QuantumCircuit is None or transpile is None:
            raise ImportError("Qiskit not available")

        backend = None
        simulator_name = None
        if AerModern is not None:
            simulator_name = "aer_simulator"
            backend = AerModern.get_backend(simulator_name)  # type: ignore[misc]
        elif Aer is not None:
            simulator_name = "qasm_simulator"
            backend = Aer.get_backend(simulator_name)  # type: ignore[misc]
        elif BasicAer is not None:
            simulator_name = "qasm_simulator"
            backend = BasicAer.get_backend(simulator_name)  # type: ignore[misc]
        else:
            raise ImportError("No Qiskit backend available")
        # Build a simple circuit: put H on all qubits then measure
        qc = QuantumCircuit(num_qubits, num_qubits)
        for q in range(num_qubits):
            qc.h(q)
        qc.measure(range(num_qubits), range(num_qubits))

        # Execute
        try:
            # Newer Aer prefers transpile
            tqc = transpile(qc, backend)  # type: ignore[misc]
            job = backend.run(tqc, shots=shots)  # type: ignore[misc]
            res = job.result()
            counts = res.get_counts()
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            # Very old interface fallback
            if q_execute is None:
                raise ImportError("Qiskit execute not available") from e
            job = q_execute(qc, backend=backend, shots=shots)  # type: ignore[misc]
            res = job.result()
            # Some older APIs require passing qc to get_counts
            try:
                counts = res.get_counts(qc)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
                counts = res.get_counts()

        # Normalize counts to a plain dict mapping strings to ints
        try:
            result: dict[str, int] = {str(k): int(v) for k, v in dict(counts).items()}
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
            msg = "Qiskit returned invalid counts"
            raise RuntimeError(msg) from e
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Qiskit execution error: {e!s}")
        raise HTTPException(status_code=503, detail="Qiskit not available or execution failed") from e
    finally:
        _inc_metric("qiskit_execute_total")
        _inc_metric("qiskit_execute_seconds_total", time.perf_counter() - start)

    return QuantumExecuteResponse(result=result, num_qubits=num_qubits, shots=shots)


# ---------------------------
# Cirq
# ---------------------------


@router.get(
    "/cirq/health",
    summary="Cirq Service Health",
    description="Check the health status of the Cirq quantum service",
    response_model=dict[str, str],
)
async def cirq_health():
    _inc_metric("requests_total")
    return {"status": "ok", "service": "cirq", "version": "1.0.0"}


@router.get(
    "/cirq/metrics",
    summary="Cirq Service Metrics",
    description="Get Prometheus metrics from the Cirq quantum service",
)
async def cirq_metrics():
    _inc_metric("requests_total")
    data = _format_prometheus()
    return PlainTextResponse(content=data, media_type="text/plain")


@router.post(
    "/cirq/optimize",
    summary="Quantum Optimization",
    description="Perform quantum optimization using Cirq",
    response_model=QuantumOptimizeResponse,
)
async def cirq_optimize(request: QuantumOptimizeRequest):
    _inc_metric("requests_total")
    start = time.perf_counter()

    num_qubits = max(1, min(int(request.num_qubits), 12))
    depth = max(1, min(int(request.depth), 64))

    try:
        if cirq is None:
            raise ImportError("Cirq not available")
        qubits = cirq.LineQubit.range(num_qubits)  # type: ignore[misc]
        circuit = cirq.Circuit()

        # Build a parameter-free layered circuit
        for _d in range(depth):
            for q in qubits:
                circuit.append(cirq.rx(0.5).on(q))
                circuit.append(cirq.rz(0.3).on(q))
            # Entangle in a simple ring
            for i in range(num_qubits):
                circuit.append(cirq.CX(qubits[i], qubits[(i + 1) % num_qubits]))

        # Apply simple optimizations when available; if not, skip gracefully
        try:
            # Some cirq versions expose optimizers at top-level, others as functions
            if hasattr(cirq, "merge_single_qubit_gates_into_phased_x_z"):
                circuit = cirq.merge_single_qubit_gates_into_phased_x_z(circuit)
            if hasattr(cirq, "eject_z"):
                circuit = cirq.eject_z(circuit)
            if hasattr(cirq, "drop_negligible_operations"):
                circuit = cirq.drop_negligible_operations(circuit)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as opt_e:
            logger.debug("Cirq optimizations skipped due to: %s", opt_e)

        circuit_text = str(circuit)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Cirq optimization error: {e!s}")
        raise HTTPException(status_code=503, detail="Cirq not available or optimization failed") from e
    finally:
        _inc_metric("cirq_optimize_total")
        _inc_metric("cirq_optimize_seconds_total", time.perf_counter() - start)

    return QuantumOptimizeResponse(circuit=circuit_text, num_qubits=num_qubits, depth=depth)


# ---------------------------
# PennyLane
# ---------------------------


@router.get(
    "/pennylane/health",
    summary="PennyLane Service Health",
    description="Check the health status of the PennyLane quantum service",
    response_model=dict[str, str],
)
async def pennylane_health():
    _inc_metric("requests_total")
    return {"status": "ok", "service": "pennylane", "version": "1.0.0"}


@router.get(
    "/pennylane/metrics",
    summary="PennyLane Service Metrics",
    description="Get Prometheus metrics from the PennyLane quantum service",
)
async def pennylane_metrics():
    _inc_metric("requests_total")
    data = _format_prometheus()
    return PlainTextResponse(content=data, media_type="text/plain")


@router.post(
    "/pennylane/ml",
    summary="Quantum Machine Learning",
    description="Perform quantum machine learning using PennyLane",
    response_model=QuantumMLResponse,
)
async def pennylane_ml(request: QuantumMLRequest):
    _inc_metric("requests_total")
    start = time.perf_counter()

    num_qubits = max(1, min(int(request.num_qubits), 8))
    layers = max(1, min(int(request.layers), 16))

    try:
        if qml is None:
            raise ImportError("PennyLane not available")
        dev = qml.device("default.qubit", wires=num_qubits, shots=None)  # type: ignore[misc]

        def _angles(n_qubits: int, n_layers: int) -> np.ndarray:
            # Deterministic parameter schedule without randomness
            arr = np.empty((n_layers, n_qubits, 3), dtype=float)
            for layer in range(n_layers):
                for w in range(n_qubits):
                    base = 0.1 * (layer + 1) * (w + 1)
                    arr[layer, w, 0] = base
                    arr[layer, w, 1] = base * 1.3
                    arr[layer, w, 2] = base * 0.7
            return arr

        @qml.qnode(dev)
        def model(params: np.ndarray) -> float:
            for layer in range(layers):
                for w in range(num_qubits):
                    a, b, c = params[layer, w]
                    qml.Rot(a, b, c, wires=w)
                for w in range(num_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])
            return qml.expval(qml.PauliZ(0))

        params = _angles(num_qubits, layers)
        val = float(model(params))
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"PennyLane ML error: {e!s}")
        raise HTTPException(status_code=503, detail="PennyLane not available or ML evaluation failed") from e
    finally:
        _inc_metric("pennylane_ml_total")
        _inc_metric("pennylane_ml_seconds_total", time.perf_counter() - start)

    return QuantumMLResponse(result=val, num_qubits=num_qubits, layers=layers)
