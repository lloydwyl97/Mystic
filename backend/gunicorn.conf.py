"""
Gunicorn configuration for Mystic Trading Platform
Production deployment configuration
"""

import multiprocessing
import os
from pathlib import Path

# Server socket
bind = os.getenv("GUNICORN_BIND", "0.0.0.0:8000")
backlog = int(os.getenv("GUNICORN_BACKLOG", "2048"))

# Worker processes
try:
    _cpu = multiprocessing.cpu_count()
    if _cpu is None:
        _cpu = 1
except (NotImplementedError, AttributeError):
    _cpu = 1

_default_workers = max(2, min(8, _cpu * 2 + 1))  # cap to avoid fork storms
workers = int(os.getenv("GUNICORN_WORKERS", str(_default_workers)))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = int(os.getenv("GUNICORN_WORKER_CONNECTIONS", "1000"))
threads = int(os.getenv("GUNICORN_THREADS", "1"))

# Request recycling to mitigate leaks
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.getenv("GUNICORN_MAX_REQUESTS_JITTER", "50"))

# App loading
preload_app = os.getenv("GUNICORN_PRELOAD", "true").lower() == "true"

# Timeouts
timeout = int(os.getenv("GUNICORN_TIMEOUT", "30"))
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "30"))
keepalive = int(os.getenv("GUNICORN_KEEPALIVE", "2"))

# Logging
accesslog = os.getenv("GUNICORN_ACCESSLOG", "-")
errorlog = os.getenv("GUNICORN_ERRORLOG", "-")
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = os.getenv("GUNICORN_PROC_NAME", "mystic-trading-api")

# Security limits
limit_request_line = int(os.getenv("GUNICORN_LIMIT_REQUEST_LINE", "4094"))
limit_request_fields = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELDS", "100"))
limit_request_field_size = int(os.getenv("GUNICORN_LIMIT_REQUEST_FIELD_SIZE", "8190"))

# Forwarded headers (when behind a proxy)
forwarded_allow_ips = os.getenv("GUNICORN_FORWARDED_ALLOW_IPS", "*")
proxy_protocol = os.getenv("GUNICORN_PROXY_PROTOCOL", "false").lower() == "true"

# Tmp dir optimization (Linux only)
worker_tmp_dir = None
if Path("/dev/shm").is_dir():
    worker_tmp_dir = "/dev/shm"

# SSL (optional)
# keyfile = os.getenv("GUNICORN_SSL_KEYFILE", "")
# certfile = os.getenv("GUNICORN_SSL_CERTFILE", "")

# Environment variables passed to workers
raw_env = [
    "ENVIRONMENT=" + os.getenv("ENVIRONMENT", "production"),
    "LOG_LEVEL=" + os.getenv("LOG_LEVEL", "INFO"),
]
