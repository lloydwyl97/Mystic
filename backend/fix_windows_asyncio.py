"""
Fix Windows asyncio file descriptor limit
Use Proactor event loop instead of Selector on Windows
"""

import asyncio
import sys


def configure_windows_event_loop():
    """Configure Windows to use Proactor event loop (no file descriptor limit)"""
    if sys.platform == "win32":
        # Use Proactor event loop on Windows (no select() limit)
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        print("[INFO] Windows: Using Proactor event loop (unlimited file descriptors)")
    else:
        print("[INFO] Non-Windows: Using default event loop policy")
