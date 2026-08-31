"""
Regression: the IPv4-only DNS patch must be a single shared, idempotent
bootstrap (backend/utils/network_ipv4.py), not duplicated per-module. Calling
ensure_ipv4_only() repeatedly (as multiple importing modules do) must patch
socket.getaddrinfo exactly once.
"""

from __future__ import annotations

import socket

from backend.utils.network_ipv4 import _PATCH_MARKER, ensure_ipv4_only


def test_ensure_ipv4_only_is_idempotent(monkeypatch):
    calls = {"count": 0}
    original = socket.getaddrinfo

    def _fake_original(host, port, family=0, type=0, proto=0, flags=0):
        calls["count"] += 1
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))]

    monkeypatch.setattr(socket, "getaddrinfo", _fake_original)

    ensure_ipv4_only()
    patched_once = socket.getaddrinfo
    assert getattr(patched_once, _PATCH_MARKER, False) is True

    ensure_ipv4_only()
    ensure_ipv4_only()
    assert socket.getaddrinfo is patched_once, "second/third call must be a no-op, not re-wrap"

    monkeypatch.setattr(socket, "getaddrinfo", original)


def test_ensure_ipv4_only_filters_to_af_inet(monkeypatch):
    def _mixed_family_lookup(host, port, family=0, type=0, proto=0, flags=0):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (host, port))]

    monkeypatch.setattr(socket, "getaddrinfo", _mixed_family_lookup)
    ensure_ipv4_only()
    results = socket.getaddrinfo("example.com", 443)
    assert all(r[0] == socket.AF_INET for r in results)
