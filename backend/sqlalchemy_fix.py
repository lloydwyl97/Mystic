"""
Fix for SQLAlchemy time.clock issue in Python 3.8+
"""

import time

# Add time.clock if it doesn't exist (removed in Python 3.8+)
if not hasattr(time, "clock"):
    time.clock = time.perf_counter
