import os

try:
    if os.environ.get("NO_IPV6", "1") == "1":
        try:
            from urllib3.util import connection as conn
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            conn = None

        if conn is not None:
            conn.HAS_IPV6 = False
except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
    pass
