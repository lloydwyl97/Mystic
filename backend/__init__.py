"""Mystic backend package.

Every process imports this package, so secret redaction is installed here
before any logger emits a request URL.
"""

from backend.utils.secret_log_filter import install_secret_redacting_filter as _install_secret_redaction

_install_secret_redaction()
