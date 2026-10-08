"""Research only: walk-forward relative-opportunity ranking for DAY. Never imported by live code.

Each candidate's outcome is compared with the mean outcome of every candidate in
its decision bar. A key learns the shrunk mean of that relative outcome, and a
relative label is used only once it is due, so a bar is ranked from labels of
earlier bars only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from backend.services.adaptive_learning import PRIOR_STRENGTH


@dataclass
class RankedBar:
    t: float
    scores: dict[int, float]
    outcomes: dict[int, float]
    chosen: int


@dataclass
class RelativeRanker:
    """Nested symbol -> symbol+setup -> key shrunk mean of relative outcome. Prior 0."""

    stats: dict[tuple, list[float]] = field(default_factory=dict)

    @staticmethod
    def _levels(c: dict[str, Any]) -> tuple[tuple, ...]:
        return (("sym", c["symbol"]), ("sym_setup", c["symbol"], c["setup"]), ("key", c["symbol"], c["setup"], c["regime"]))

    def learn(self, c: dict[str, Any], relative: float) -> None:
        for level in self._levels(c):
            s = self.stats.setdefault(level, [0.0, 0.0])
            s[0] += 1.0
            s[1] += float(relative)

    def score(self, c: dict[str, Any]) -> float:
        mu = 0.0
        for level in self._levels(c):
            w, s = self.stats.get(level, [0.0, 0.0])
            mu = (PRIOR_STRENGTH * mu + s) / (PRIOR_STRENGTH + w)
        return mu


def walk_forward(
    candidates: Iterable[dict[str, Any]],
    *,
    outcome: Callable[[dict[str, Any]], float | None],
    label_delay_sec: float,
    ranker: RelativeRanker | None = None,
) -> list[RankedBar]:
    """Rank each multi-coin bar with labels due strictly before or at its time.

    ``candidates`` carry ``id``, ``t``, ``symbol``, ``setup`` and ``regime``.
    A bar's relative labels become usable at ``t + label_delay_sec``."""
    rk = ranker or RelativeRanker()
    bars: dict[float, list[tuple[dict[str, Any], float]]] = {}
    for c in candidates:
        y = outcome(c)
        if y is None:
            continue
        bars.setdefault(float(c["t"]), []).append((c, float(y)))
    times = sorted(t for t, g in bars.items() if len({c["symbol"] for c, _ in g}) >= 2)
    due: list[tuple[float, dict[str, Any], float]] = []
    for t in times:
        g = bars[t]
        mean = sum(y for _, y in g) / len(g)
        due.extend((t + float(label_delay_sec), c, y - mean) for c, y in g)
    due.sort(key=lambda item: item[0])
    out: list[RankedBar] = []
    i = 0
    for t in times:
        while i < len(due) and due[i][0] <= t:
            rk.learn(due[i][1], due[i][2])
            i += 1
        g = bars[t]
        scores = {int(c["id"]): rk.score(c) for c, _ in g}
        chosen = max(scores, key=lambda k: scores[k])
        out.append(RankedBar(t=t, scores=scores, outcomes={int(c["id"]): y for c, y in g}, chosen=chosen))
    return out
