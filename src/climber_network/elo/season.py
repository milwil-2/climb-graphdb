"""climber_network.elo.season — Phase 4: season-level aggregates (pure, stdlib).

Phase 4 of the outcome-variable work (#48) rolls the per-(athlete, event)
outcomes already stamped on ``Performance`` nodes up to one **season summary** per
(athlete, season, discipline). Where Phases 1-3 produce a point estimate for each
appearance — ``elo_residual`` / ``result_percentile`` / ``surprisal`` / ``p_win`` /
``rank_std`` — this module collapses a whole season's worth of those into a single
``SeasonAggregate`` so a season can be characterised at a glance: how far the
athlete over/under-performed on aggregate, how skilled and how consistent they
were, and how rested.

The headline season signals are:

* ``season_skill`` — mean ``p_win`` across the season (higher = more dominant).
* ``season_consistency`` — mean ``rank_std`` (lower = more consistent placements).
* ``over_under`` — the cumulative **signed** under-performance: the sum of the
  available ``elo_residual`` values. Positive means the athlete finished worse
  than the model expected across the season (under-performed); negative means
  they over-performed. This scales with the number of events, so it is a
  legitimate "total damage" measure but volume-biased.
* ``mean_over_under`` — the per-event-normalized companion (``over_under /
  n_events``). This removes the volume bias so seasons are comparable
  regardless of how many events the athlete entered, and is the right quantity
  to rank by and to correlate against the per-event ``mean_rested_index``.
* ``n_upsets`` — how many appearances were genuine upsets (``surprisal`` above a
  threshold).

A driver report then correlates each athlete-season's ``mean_over_under``
against its ``mean_rested_index`` to test the project's standing hypothesis: a
less-rested season (more jet-lag / travel load) should coincide with more
under-performance, i.e. a *negative* correlation. The per-event-normalized
``mean_over_under`` (not the volume-biased ``over_under`` sum) is used so the
correlation isn't distorted by how many events each athlete entered.

This module is pure stdlib: it never imports ``climbing_elo`` /
``knowledge_graph`` and performs no I/O. The integration layer that reads the
graph and MERGEs ``SeasonSummary`` nodes lives in :mod:`sync.season`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from climber_network.stats import pearson

__all__ = [
    "PerformanceRecord",
    "SeasonAggregate",
    "aggregate_seasons",
    "season_drivers_report",
]


# ---------------------------------------------------------------------------
# Inputs / outputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceRecord:
    """One representative (athlete, event) appearance feeding a season rollup.

    ``athlete_id`` is the **full** athlete node id (e.g. ``ath:5``). The outcome
    fields mirror the props stamped on the representative ``Performance`` node by
    the earlier phases; each is optional because a given build may not have
    populated every outcome variable yet. ``None`` values are simply skipped when
    a mean is taken.
    """

    athlete_id: str
    season: int
    discipline: str
    elo_residual: float | None = None
    result_percentile: float | None = None
    surprisal: float | None = None
    p_win: float | None = None
    rank_std: float | None = None
    rested_index: float | None = None


@dataclass(frozen=True)
class SeasonAggregate:
    """A single rolled-up summary for one (athlete, season, discipline).

    Each ``mean_*`` / ``season_*`` rollup is taken over the **non-None** values of
    the corresponding field across the season's records; a rollup with no usable
    data is ``None``. ``over_under`` is the cumulative signed under-performance
    (sum of the available ``elo_residual`` values — positive = the athlete
    underperformed across the season); ``mean_over_under`` is its
    per-event-normalized companion (``over_under / n_events``, or ``0.0`` when
    the season has no events) used for volume-fair ranking and correlation.
    ``n_upsets`` counts appearances whose ``surprisal`` exceeds the upset
    threshold.
    """

    athlete_id: str
    season: int
    discipline: str
    n_events: int
    mean_elo_residual: float | None
    mean_result_percentile: float | None
    mean_surprisal: float | None
    season_skill: float | None
    season_consistency: float | None
    mean_rested_index: float | None
    n_upsets: int
    over_under: float
    mean_over_under: float


# ---------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean of *values*, or ``None`` when the list is empty."""
    if not values:
        return None
    return sum(values) / len(values)


def _collect(records: list[PerformanceRecord], attr: str) -> list[float]:
    """Return the non-None values of *attr* across *records* as floats."""
    out: list[float] = []
    for rec in records:
        val = getattr(rec, attr)
        if val is not None:
            out.append(float(val))
    return out


def aggregate_seasons(
    records: list[PerformanceRecord],
    *,
    upset_threshold: float = 2.0,
) -> list[SeasonAggregate]:
    """Group *records* by (athlete_id, season, discipline) and roll each up.

    Every ``mean_*`` / ``season_*`` field is the mean of the non-None values of
    its source field within the group (``None`` when none are present).
    ``over_under`` is the sum of the available ``elo_residual`` values, and
    ``mean_over_under`` is that sum divided by ``n_events`` (``0.0`` when the
    group is empty). ``n_upsets`` is the count of records whose ``surprisal`` is
    not ``None`` and
    strictly greater than *upset_threshold*.

    The result is sorted by (athlete_id, season, discipline) for determinism.
    """
    groups: dict[tuple[str, int, str], list[PerformanceRecord]] = defaultdict(list)
    for rec in records:
        groups[(rec.athlete_id, rec.season, rec.discipline)].append(rec)

    aggregates: list[SeasonAggregate] = []
    for (athlete_id, season, discipline), members in groups.items():
        residuals = _collect(members, "elo_residual")
        n_upsets = sum(
            1 for rec in members if rec.surprisal is not None and rec.surprisal > upset_threshold
        )
        n_events = len(members)
        over_under = sum(residuals)
        aggregates.append(
            SeasonAggregate(
                athlete_id=athlete_id,
                season=season,
                discipline=discipline,
                n_events=n_events,
                mean_elo_residual=_mean(residuals),
                mean_result_percentile=_mean(_collect(members, "result_percentile")),
                mean_surprisal=_mean(_collect(members, "surprisal")),
                season_skill=_mean(_collect(members, "p_win")),
                season_consistency=_mean(_collect(members, "rank_std")),
                mean_rested_index=_mean(_collect(members, "rested_index")),
                n_upsets=n_upsets,
                over_under=over_under,
                mean_over_under=(over_under / n_events) if n_events > 0 else 0.0,
            )
        )

    aggregates.sort(key=lambda a: (a.athlete_id, a.season, a.discipline))
    return aggregates


# ---------------------------------------------------------------------------
# Driver report — over_under vs mean_rested_index.
# ---------------------------------------------------------------------------


def _drivers_block(aggregates: list[SeasonAggregate]) -> dict[str, Any]:
    """Pearson(mean_over_under, mean_rested_index) over aggregates that have both.

    The per-event-normalized ``mean_over_under`` (not the volume-biased
    ``over_under`` sum) is correlated against the per-event ``mean_rested_index``
    so the two series are on the same per-event footing.
    """
    xs: list[float] = []
    ys: list[float] = []
    for agg in aggregates:
        if agg.mean_rested_index is None:
            continue
        xs.append(agg.mean_rested_index)
        ys.append(agg.mean_over_under)
    return {"pearson_r": pearson(xs, ys), "n": len(xs)}


def season_drivers_report(aggregates: list[SeasonAggregate]) -> dict[str, Any]:
    """Correlate season ``mean_over_under`` against ``mean_rested_index``.

    Returns the structured shape::

        {
          "overall": {"pearson_r": float|None, "n": int},
          "by_discipline": {code: {"pearson_r": ..., "n": ...}, ...},
          "success_signal": "negative correlation expected ...",
        }

    Only athlete-seasons that carry a ``mean_rested_index`` contribute (the y
    value, ``mean_over_under``, is always present). The per-event-normalized
    ``mean_over_under`` is used (not the volume-biased ``over_under`` sum) so the
    correlation isn't distorted by event count. A *negative* correlation is the
    success signal: a less-rested season should coincide with more
    under-performance.
    """
    by_discipline: dict[str, list[SeasonAggregate]] = defaultdict(list)
    for agg in aggregates:
        if agg.discipline:
            by_discipline[agg.discipline].append(agg)

    return {
        "overall": _drivers_block(aggregates),
        "by_discipline": {code: _drivers_block(group) for code, group in by_discipline.items()},
        "success_signal": (
            "negative correlation expected (less rested season -> more underperformance)"
        ),
    }
