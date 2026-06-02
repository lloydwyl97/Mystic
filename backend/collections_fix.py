"""
Fix for collections classes in Python 3.12+
Import this at the start of your application
"""

import collections
import collections.abc

# Add classes moved to collections.abc in Python 3.12+ back to collections for compatibility
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping

if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence
