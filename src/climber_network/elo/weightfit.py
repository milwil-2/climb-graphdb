"""climber_network.elo.weightfit — Empirical L3 travel-weight fitting.

Re-fits the ``w1`` / ``w2`` weights in the composite ``rested_index`` formula::

    rested_index = clamp(1 - w1 * jetlag_residual - w2 * travel_fatigue, 0, 1)

against observed performance outcomes (e.g. ``elo_residual`` or
``result_percentile``) stored in ``RestednessState`` nodes.  For each candidate
pair ``(w1, w2 = 1 - w1)`` the module recomputes a synthetic ``rested_index``
from the raw stored components and correlates it with the outcome.

**Objective:** the *most-negative* Pearson correlation is the target — lower
restedness should predict a worse-than-expected performance, so if the outcome
variable measures **underperformance** (higher = worse): ``elo_residual``
(``actual_rank - expected_rank``; positive = finished worse than expected) and
``result_percentile`` (higher = a worse finish under the placement PMF) both have
that orientation. The hypothesised relationship is therefore:

    rested_index ↑  ⟹  outcome ↓   (NEGATIVE correlation, signal correct)

So we fit the ``w1`` whose ``pearson(rested_index, outcome)`` is the **most
negative** — the weights that best express the jet-lag-hurts hypothesis. The full
grid ``curve`` is returned alongside, so the sign is always visible if the data
does not support the hypothesis. :func:`fit_weights` documents the exact rule.

**Self-contained:** stdlib only.  No numpy / scipy.  No ``climbing_elo`` /
``knowledge_graph`` imports.  The clamp mirrors :func:`travel.formulas._clamp`
exactly (same expression: ``max(lo, min(hi, value))``).

Usage::

    from climber_network.elo.weightfit import WeightSample, fit_weights

    samples = [WeightSample(jr=0.8, tf=0.6, outcome=-0.3), ...]
    result = fit_weights(samples)
    print(result["best"])   # {"w1": ..., "w2": ..., "pearson": ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from climber_network.stats import pearson

__all__ = [
    "WeightSample",
    "fit_weights",
    "recompute_rested_index",
]

#: Literature-prior weights (from ``config.TravelParams``).
_PRIOR_W1: float = 0.7
_PRIOR_W2: float = 0.3


def recompute_rested_index(
    jetlag_residual: float,
    travel_fatigue: float,
    w1: float,
    w2: float,
) -> float:
    """Recompute ``rested_index`` from stored components and candidate weights.

    Mirrors the formula in :func:`travel.formulas.rested_index` exactly::

        clamp(1 - w1 * jetlag_residual - w2 * travel_fatigue, 0, 1)

    The clamp uses ``max(0.0, min(1.0, value))``, identical to the private
    ``_clamp`` helper in ``travel/formulas.py``.

    Args:
        jetlag_residual: Fraction of jet lag still unresolved, in ``[0, 1]``.
        travel_fatigue:  Raw travel-fatigue fraction, in ``[0, 1]``.
        w1:              Candidate weight for the jet-lag component.
        w2:              Candidate weight for the travel-fatigue component.

    Returns:
        Restedness score in ``[0, 1]`` (1 = fully rested).
    """
    penalty = w1 * jetlag_residual + w2 * travel_fatigue
    # clamp — mirrors travel/formulas.py _clamp(value, lo=0.0, hi=1.0)
    return max(0.0, min(1.0, 1.0 - penalty))


@dataclass(frozen=True)
class WeightSample:
    """One observation used for weight fitting.

    Attributes:
        jetlag_residual:  Stored ``jetlag_residual`` from a ``RestednessState``
                          node, in ``[0, 1]``.
        travel_fatigue:   Stored ``travel_fatigue`` from a ``RestednessState``
                          node, in ``[0, 1]``.
        outcome:          The performance signal to correlate against (e.g.
                          ``elo_residual`` or ``result_percentile``).  Its sign
                          convention is caller-defined; see :func:`fit_weights`.
        discipline:       Optional climbing discipline tag (``"L"``, ``"B"``,
                          ``"S"``).  Unused by the grid search but preserved for
                          downstream stratified analyses.
        travel_direction: Optional travel direction tag (``"E"``, ``"W"``,
                          ``"none"``).  Same: unused by the grid, available for
                          stratification.
    """

    jetlag_residual: float
    travel_fatigue: float
    outcome: float
    discipline: str | None = field(default=None)
    travel_direction: str | None = field(default=None)


def fit_weights(
    samples: list[WeightSample],
    *,
    grid_steps: int = 101,
) -> dict[str, Any]:
    """Grid-search ``w1`` to find the weights that best express the travel signal.

    For each candidate ``w1`` in ``grid_steps`` evenly-spaced points over
    ``[0, 1]`` (with ``w2 = 1 - w1``):

    1. Recompute ``rested_index`` per sample using :func:`recompute_rested_index`.
    2. Compute ``pearson(rested_indices, outcomes)``.

    **Selection objective:** pick the ``w1`` whose Pearson correlation is the
    **most negative** — the weight pair that best expresses the hypothesised
    signal (lower restedness → worse-than-expected outcome). The full ``curve``
    is returned too, so a wrong-sign / no-signal result is always visible.

    ``None`` Pearson values (undefined — zero variance in either series) are
    skipped.  If no valid correlation exists (fewer than 2 samples, or all Pearson
    values are ``None``) the best weights fall back to the literature prior
    (``w1 = 0.7``) and both ``best`` and ``current`` Pearson values are ``None``.

    Args:
        samples:    Observations to fit against.  Empty list is accepted.
        grid_steps: Number of evenly-spaced points in ``[0, 1]`` to evaluate.
                    Must be >= 2.  Default 101 gives a step of 0.01.

    Returns:
        A dict with four keys:

        ``"n"``
            Number of samples provided.
        ``"best"``
            ``{"w1": float, "w2": float, "pearson": float | None}`` — the
            weight pair with the most-negative Pearson correlation, or the
            literature prior when no valid correlation exists.
        ``"current"``
            Same shape, evaluated at the literature prior ``w1 = 0.7``.
        ``"curve"``
            List of ``{"w1", "w2", "pearson"}`` dicts for all ``grid_steps``
            candidates in ``w1`` order — useful for plotting.
    """
    n = len(samples)

    # Build the outcome vector once — it's constant across all w1 candidates.
    outcomes: list[float] = [s.outcome for s in samples]

    # Evaluate every grid point.
    curve: list[dict[str, Any]] = []
    for i in range(grid_steps):
        w1 = i / (grid_steps - 1) if grid_steps > 1 else 0.0
        w2 = 1.0 - w1
        if n >= 2:
            rested = [
                recompute_rested_index(s.jetlag_residual, s.travel_fatigue, w1, w2) for s in samples
            ]
            r = pearson(rested, outcomes)
        else:
            r = None
        curve.append({"w1": w1, "w2": w2, "pearson": r})

    # --- best: MOST NEGATIVE correlation -------------------------------------
    # The project-wide success signal is a *negative* rested_index ↔ outcome
    # correlation (lower rested → worse-than-expected). We fit the weights that
    # best express that hypothesis, i.e. the w1 whose Pearson is the smallest
    # (most negative). The full ``curve`` is returned so the sign is always
    # visible if the data does not support the hypothesis.
    best_entry: dict[str, Any] | None = None
    for entry in curve:
        r = entry["pearson"]
        if r is None:
            continue
        if best_entry is None or r < best_entry["pearson"]:
            best_entry = entry

    if best_entry is None:
        # Fall back to literature prior when no valid correlation was found.
        best: dict[str, Any] = {"w1": _PRIOR_W1, "w2": _PRIOR_W2, "pearson": None}
    else:
        best = {"w1": best_entry["w1"], "w2": best_entry["w2"], "pearson": best_entry["pearson"]}

    # --- current: evaluate the literature prior weights ----------------------
    if n >= 2:
        rested_prior = [
            recompute_rested_index(s.jetlag_residual, s.travel_fatigue, _PRIOR_W1, _PRIOR_W2)
            for s in samples
        ]
        current_pearson: float | None = pearson(rested_prior, outcomes)
    else:
        current_pearson = None
    current: dict[str, Any] = {"w1": _PRIOR_W1, "w2": _PRIOR_W2, "pearson": current_pearson}

    return {
        "n": n,
        "best": best,
        "current": current,
        "curve": curve,
    }
