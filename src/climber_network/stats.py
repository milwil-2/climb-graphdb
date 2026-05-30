"""climber_network.stats — small, dependency-free statistics helpers.

Pure stdlib (no numpy / scipy) so the package stays lightweight and the result
is deterministic. Shared by the outcome-variable correlation reports
(``sync.validate_elo`` / ``sync.montecarlo``) and the Phase-2 calibration /
weight-fitting analyses (``elo.calibration`` / ``elo.weightfit``).
"""

from __future__ import annotations

import math


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient of paired samples, or ``None`` if undefined.

    Returns ``None`` when fewer than two pairs are supplied or when either series
    has zero variance (the coefficient is then mathematically undefined).
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx = x - mean_x
        dy = y - mean_y
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)
