"""Probability-integral-transform calibration checks for the MC placement model.

The Monte-Carlo placement model (:mod:`climber_network.elo.montecarlo`) stamps
two scalars onto each representative ``Performance`` node:

* ``result_percentile`` = ``P(finish <= actual_rank)`` under the placement PMF,
  i.e. the CDF evaluated *at* the realised rank (inclusive); and
* ``surprisal`` = ``-log P(finish == actual_rank)``, the information content of
  the realised rank.

This module turns those two stored scalars into the **probability integral
transform (PIT)** of the realised results and scores how *calibrated* the model
is. Under a well-calibrated model the (randomized) PIT values are distributed
``Uniform[0, 1]``; systematic departures from uniformity diagnose
over/under-dispersion or bias in the placement distributions.

Randomized PIT from the two stored scalars
-------------------------------------------
The placement distribution is discrete (a PMF over integer ranks), so the plain
PIT is not continuous and cannot be exactly uniform. The standard fix is the
**randomized** PIT (Dawid; Czado et al. 2009): at the realised value ``x`` draw
``u ~ Uniform[0, 1]`` and report ``P(X < x) + u * P(X == x)``. This *is* exactly
``Uniform[0, 1]`` when the model is correct.

We can recover both pieces from the two stored scalars alone — no PMF needed:

* the point mass at the realised rank is ``P(X == actual) = exp(-surprisal)``;
* the strict-below mass is
  ``P(X < actual) = result_percentile - exp(-surprisal)``

(because ``result_percentile`` is the *inclusive* CDF ``P(X <= actual)``). Hence

    randomized_pit = (result_percentile - point_mass) + u * point_mass

with ``u = 0`` giving ``P(X < actual)`` and ``u = 1`` giving
``result_percentile``.

Determinism
-----------
Any randomness (the ``u`` draw) flows through a caller-supplied
``random.Random`` so the analysis is reproducible. Pure stdlib only — no numpy /
scipy — to match the rest of the package.
"""

from __future__ import annotations

import math
import random

__all__ = [
    "calibration_report",
    "expected_calibration_error",
    "ks_uniform_statistic",
    "point_mass_from_surprisal",
    "randomized_pit",
    "randomized_pit_rng",
    "reliability_bins",
]


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the closed unit interval ``[0, 1]``."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def point_mass_from_surprisal(surprisal: float) -> float:
    """Point mass ``P(X == actual_rank) = exp(-surprisal)``, clamped to ``[0, 1]``.

    ``surprisal`` is the stored ``-log P(finish == actual_rank)`` in nats, so the
    point mass is its exponential. The result is clamped into ``[0, 1]`` to stay a
    valid probability even if floating-point or flooring effects nudge it out of
    range. ``surprisal == 0`` -> ``1.0``; a large ``surprisal`` -> ``~0``.
    """
    return _clamp01(math.exp(-surprisal))


def randomized_pit(result_percentile: float, point_mass: float, u: float) -> float:
    """Randomized probability integral transform from the two stored scalars.

    Computes ``(result_percentile - point_mass) + u * point_mass`` and clamps the
    result into ``[0, 1]``. Here ``result_percentile`` is the *inclusive* CDF
    ``P(X <= actual)`` and ``point_mass`` is ``P(X == actual)`` (see
    :func:`point_mass_from_surprisal`), so ``result_percentile - point_mass`` is
    the strict-below mass ``P(X < actual)``.

    ``u`` is a ``Uniform[0, 1]`` draw spreading the value across the point mass:
    ``u == 0`` -> ``P(X < actual)`` and ``u == 1`` -> ``result_percentile``. Under
    a correct model the returned values are ``Uniform[0, 1]``.
    """
    return _clamp01((result_percentile - point_mass) + u * point_mass)


def randomized_pit_rng(result_percentile: float, point_mass: float, *, rng: random.Random) -> float:
    """As :func:`randomized_pit`, drawing ``u = rng.random()`` for reproducibility.

    All randomness flows through the supplied ``random.Random`` so the analysis is
    deterministic for a fixed seed.
    """
    return randomized_pit(result_percentile, point_mass, rng.random())


def ks_uniform_statistic(values: list[float]) -> float:
    """Two-sided Kolmogorov-Smirnov distance to the ``Uniform[0, 1]`` CDF.

    Sorts the ``n`` values ``v_0 <= ... <= v_{n-1}`` and returns the standard
    two-sided statistic

        D = max_i max( (i + 1) / n - v_i , v_i - i / n )    for i in 0..n-1,

    i.e. the largest gap between the empirical CDF and the identity CDF of
    ``Uniform[0, 1]``. Smaller is more uniform (better calibrated).

    Returns ``0.0`` for empty input (no evidence of mis-calibration).
    """
    n = len(values)
    if n == 0:
        return 0.0
    ordered = sorted(values)
    d = 0.0
    for i, v in enumerate(ordered):
        above = (i + 1) / n - v
        below = v - i / n
        d = max(d, above, below)
    return d


def expected_calibration_error(values: list[float], *, n_bins: int = 10) -> float:
    """L1 deviation of binned PIT frequencies from the uniform expectation.

    Bins ``[0, 1]`` into ``n_bins`` equal-width bins and returns

        sum_b | count_b / N - 1 / n_bins |,

    the total-variation-style departure of the observed bin frequencies from the
    flat ``Uniform[0, 1]`` expectation. ``0`` means perfectly uniform binning;
    larger means worse calibration.

    Values exactly equal to ``1.0`` are placed in the last bin (so every value in
    ``[0, 1]`` is counted exactly once). Returns ``0.0`` for empty input.

    Raises ``ValueError`` if ``n_bins`` is not strictly positive.
    """
    if n_bins <= 0:
        msg = f"n_bins must be strictly positive, got {n_bins!r}"
        raise ValueError(msg)
    n = len(values)
    if n == 0:
        return 0.0
    counts = _bin_counts(values, n_bins)
    expected = 1.0 / n_bins
    return sum(abs(c / n - expected) for c in counts)


def _bin_counts(values: list[float], n_bins: int) -> list[int]:
    """Tally ``values`` into ``n_bins`` equal-width bins over ``[0, 1]``.

    Each value is clamped into ``[0, 1]`` first; a value of exactly ``1.0`` (or
    above) lands in the last bin rather than overflowing.
    """
    counts = [0] * n_bins
    for raw in values:
        v = _clamp01(raw)
        idx = int(v * n_bins)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1
    return counts


def reliability_bins(values: list[float], *, n_bins: int = 10) -> list[dict[str, float]]:
    """Per-bin reliability table for the PIT values.

    Returns one dict per equal-width bin of ``[0, 1]`` with keys:

    * ``lo`` / ``hi`` — the bin's half-open edges (``[lo, hi)``; the last bin
      includes its upper edge ``1.0``);
    * ``count`` — number of values in the bin;
    * ``freq`` — ``count / N`` (the empirical bin frequency);
    * ``expected_freq`` — ``1 / n_bins`` (the uniform target).

    For empty input every ``count`` and ``freq`` is ``0.0``. Raises ``ValueError``
    if ``n_bins`` is not strictly positive.
    """
    if n_bins <= 0:
        msg = f"n_bins must be strictly positive, got {n_bins!r}"
        raise ValueError(msg)
    n = len(values)
    counts = _bin_counts(values, n_bins) if n else [0] * n_bins
    width = 1.0 / n_bins
    expected = 1.0 / n_bins
    bins: list[dict[str, float]] = []
    for b in range(n_bins):
        count = counts[b]
        bins.append(
            {
                "lo": b * width,
                "hi": (b + 1) * width,
                "count": float(count),
                "freq": (count / n) if n else 0.0,
                "expected_freq": expected,
            }
        )
    return bins


def calibration_report(pit_values: list[float], *, n_bins: int = 10) -> dict:
    """Bundle the calibration diagnostics for a set of PIT values.

    Returns a dict with keys:

    * ``n`` — number of PIT values;
    * ``mean`` — mean of the PIT values (a calibrated model has ``mean ~= 0.5``);
    * ``ks`` — :func:`ks_uniform_statistic`;
    * ``ece`` — :func:`expected_calibration_error`;
    * ``bins`` — :func:`reliability_bins`.

    For empty input (``n == 0``) ``mean``, ``ks`` and ``ece`` are ``None`` (the
    statistics are undefined with no data), while ``bins`` is the all-zero table.
    """
    n = len(pit_values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "ks": None,
            "ece": None,
            "bins": reliability_bins(pit_values, n_bins=n_bins),
        }
    return {
        "n": n,
        "mean": sum(pit_values) / n,
        "ks": ks_uniform_statistic(pit_values),
        "ece": expected_calibration_error(pit_values, n_bins=n_bins),
        "bins": reliability_bins(pit_values, n_bins=n_bins),
    }
