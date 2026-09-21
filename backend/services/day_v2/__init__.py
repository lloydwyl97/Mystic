"""DAY V2 shadow-only package.

No code in this package may place, cancel, or mutate live orders, positions,
cash balances, or accounting. All engines default to SHADOW or DISABLED.
Only LEGACY_DAY_LIVE retains existing production authority and it is never
instantiated from this package.
"""
