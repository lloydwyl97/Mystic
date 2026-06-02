import os

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from qiskit import Aer, QuantumCircuit, execute

app = FastAPI(title="Mystic Qiskit Quantum Service", version="1.0.0")

REQUEST_COUNT = Counter("qiskit_requests_total", "Total Qiskit API Requests", ["endpoint"])

# Health endpoint DELETED


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/quantum/execute")
def quantum_execute(
    num_qubits: int = Query(2, ge=1, le=16),
    shots: int = Query(1024, ge=1, le=200000),
):
    REQUEST_COUNT.labels(endpoint="/quantum/execute").inc()
    try:
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
        qc.measure_all()
        backend = Aer.get_backend("qasm_simulator")
        job = execute(qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(qc)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        return {"result": counts, "num_qubits": num_qubits, "shots": shots}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8087"))
    uvicorn.run(app, host="0.0.0.0", port=port)
