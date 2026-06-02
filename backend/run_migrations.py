"""
Script to run Alembic migrations with proper Python path
"""

import sys
from pathlib import Path

from alembic.config import CommandLine

# Add project root to Python path
try:
    project_root = Path(__file__).parent.parent.resolve()
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive mode)
    project_root = Path.cwd().resolve()

root_str = str(project_root)

# Normalize existing sys.path entries and check for presence
_present = False
for p in sys.path:
    if not p:
        # empty entry refers to CWD; compare against that
        try:
            if str(Path.cwd().resolve()) == root_str:
                _present = True
                break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            continue
    else:
        try:
            if str(Path(p).resolve()) == root_str:
                _present = True
                break
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
            if p == root_str:
                _present = True
                break

if not _present:
    sys.path.insert(0, root_str)

CommandLine().main(argv=sys.argv[1:])
