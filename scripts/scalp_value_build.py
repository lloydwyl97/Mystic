"""Train and install the SCALP value model from every elapsed candidate label.

Run with the app stopped: the build replaces the model and its fold clock,
which the running integration advances every minute.

    venv/bin/python3 scripts/scalp_value_build.py --db mystic_trading.db
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.scalp_v2 import continuation_value as cv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--as-of", help="ISO time; default now")
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")).timestamp() if args.as_of else None
    summary = cv.build(args.db, as_of=as_of)
    model = cv.load_model(args.db, fresh=True)
    if model is not None:
        summary["intercept_bps"] = {str(h): round(float(beta[0]) * 1e4, 3) for h in cv.HORIZONS_SEC if (beta := model.beta(h)) is not None}
        summary["effective_n"] = {str(h): round(float(model.stats[h]["n"]), 1) for h in cv.HORIZONS_SEC}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
