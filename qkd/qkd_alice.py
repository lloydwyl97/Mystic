#!/usr/bin/env python3
"""
Mystic QKD Alice
Quantum Key Distribution - Alice node, with Prometheus metrics.
"""

import asyncio
import os
import secrets
import time
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

app = FastAPI(
    title="Mystic QKD Alice",
    description="Quantum Key Distribution - Alice node.",
    version="1.0.0",
)

# ---------------- Prometheus metrics ----------------
REQUEST_COUNT = Counter("qkd_alice_requests_total", "Total QKD Alice API Requests", ["endpoint"])
KEYS_GENERATED = Counter("qkd_alice_keys_generated", "Total quantum keys generated")
KEY_RATE = Gauge("qkd_alice_key_rate_kbps", "Key generation rate in kbps")
QUANTUM_STATE = Gauge("qkd_alice_quantum_state_quality", "Quantum state quality (0-100)")
GENERATION_TIME = Histogram("qkd_alice_generation_duration_seconds", "Key generation time (seconds)")
ERROR_COUNT = Counter("qkd_alice_errors_total", "Total QKD Alice errors", ["endpoint"])

# ---------------- Endpoints ----------------
# Health endpoint DELETED


@app.get("/metrics")
def metrics():
    REQUEST_COUNT.labels(endpoint="/metrics").inc()
    # Prometheus expects raw bytes with the correct content type
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/qkd/alice/generate", tags=["QKD"])
async def generate_key(
    key_length_bits: int = Query(256, ge=8, le=1_000_000, description="Requested key length in bits"),
    protocol: Literal["BB84", "E91", "B92"] = "BB84",
):
    """
    Simulate QKD key generation.

    - key_length_bits: 8..1,000,000 (upper bound to avoid huge payloads)
    - protocol: one of BB84 | E91 | B92

    Returns a bitstring (simulation only), generation latency, and derived metrics.
    """
    endpoint = "/qkd/alice/generate"
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    t0 = time.monotonic()

    try:
        # Simulated generation time scales with requested length and protocol
        proto_factor = {"BB84": 1.0, "E91": 1.25, "B92": 0.9}[protocol]
        generation_delay = (key_length_bits / 256.0) * 2.0 * proto_factor  # baseline model
        await asyncio.sleep(generation_delay)

        # Cryptographically secure random bits (simulation stand-in for QKD output)
        # Efficient generation: produce bytes then trim to requested bit length
        num_bytes = (key_length_bits + 7) // 8
        raw = secrets.token_bytes(num_bytes)
        # Convert to bit string and trim excess bits from the end
        bitstr = "".join(f"{byte:08b}" for byte in raw)[:key_length_bits]

        # Metrics
        KEYS_GENERATED.inc()
        elapsed = time.monotonic() - t0
        GENERATION_TIME.observe(elapsed)

        # Avoid division by zero; elapsed should always be > 0 given sleep above, but guard anyway
        kbps = (key_length_bits / max(elapsed, 1e-9)) / 1000.0
        KEY_RATE.set(kbps)
        QUANTUM_STATE.set(95.5)  # demo/placeholder quality
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        return {
            "generated": True,
            "protocol": protocol,
            "key_length_bits": key_length_bits,
            "quantum_key": bitstr,
            "generation_time_seconds": elapsed,
            "key_rate_kbps": kbps,
            "quantum_state_quality": 95.5,
            "quantum_bit_error_rate_percent": 0.5,  # simulated QBER
        }


@app.get("/qkd/alice/status", tags=["QKD"])
def alice_status():
    REQUEST_COUNT.labels(endpoint="/qkd/alice/status").inc()
    return {
        "alice_status": "operational",
        "quantum_state": "entangled",
        "keys_generated": 1250,
        "current_key_rate_kbps": 45.5,
        "quantum_state_quality": 95.5,
        "error_rate_percent": 0.5,
        "bob_connection": "established",
        "eve_detection": "active",
    }


@app.post("/qkd/alice/transmit", tags=["QKD"])
async def transmit_quantum_state(
    state_type: Literal["photon", "ion", "electron"] = "photon",
    polarization: Literal["random", "rectilinear", "diagonal"] = "random",
):
    """
    Simulate quantum state transmission.
    """
    endpoint = "/qkd/alice/transmit"
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    t0 = time.monotonic()

    try:
        await asyncio.sleep(0.1)  # simulation delay
        elapsed = time.monotonic() - t0
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        return {
            "transmitted": True,
            "state_type": state_type,
            "polarization": polarization,
            "transmission_time_seconds": elapsed,
            "success_rate_percent": 98.5,
            "quantum_efficiency": 0.95,
        }


@app.get("/qkd/alice/analytics", tags=["QKD"])
def alice_analytics():
    REQUEST_COUNT.labels(endpoint="/qkd/alice/analytics").inc()
    return {
        "key_generation_performance": {
            "total_keys_generated": 1250,
            "average_key_rate_kbps": 45.5,
            "peak_key_rate_kbps": 52.3,
            "key_generation_success_rate": 99.5,
        },
        "quantum_performance": {
            "quantum_state_quality_avg": 95.5,
            "entanglement_fidelity": 0.98,
            "quantum_efficiency": 0.95,
            "photon_detection_rate": 0.92,
        },
        "security_metrics": {
            "eve_detection_rate": 99.8,
            "quantum_bit_error_rate": 0.5,
            "privacy_amplification_efficiency": 0.85,
            "final_key_rate_kbps": 38.7,
        },
        "system_health": {
            "laser_stability_percent": 99.9,
            "detector_efficiency": 0.95,
            "temperature_stability_celsius": 0.1,
            "optical_alignment": "optimal",
        },
    }


# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8106"))
    uvicorn.run(app, host="0.0.0.0", port=port)
