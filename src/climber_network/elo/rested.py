"""climber_network.elo.rested — shared RestednessState correlation helper.

Both outcome-variable syncs (:mod:`sync.validate_elo`, which correlates the
closed-form ``elo_residual``, and :mod:`sync.montecarlo`, which correlates the
Monte-Carlo ``result_percentile``) join the same
``(athlete_id, event_id) -> (rested_index, travel_direction)`` map from
:data:`REST_QUERY` to their representative reps and group the resulting pairs
into ``overall`` / ``by_discipline`` / ``by_travel_direction`` Pearson blocks.

The join, the key handling, and the grouping are identical between the two; only
the outcome field and a human-readable ``success_signal`` string differ. This
module factors out the one shared implementation —
:func:`correlate_against_rested` — that each sync calls with its own outcome
(and discipline / travel-direction) accessors. The Pearson coefficient itself
lives in :mod:`climber_network.stats`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

import climber_network.stats as stats

#: Read query for RestednessState nodes already built by the travel sync (P3*).
#: Keyed ``rest:{ath_id}:{evt_id}``; we read the rested_index plus the
#: discipline / travel-direction breakdown dimensions for the report.
REST_QUERY = (
    "MATCH (r:RestednessState) "
    "RETURN r.athlete_id AS athlete_id, r.event_id AS event_id, "
    "r.rested_index AS rested_index, r.discipline AS discipline, "
    "r.travel_direction AS travel_direction"
)


T = TypeVar("T")


class ReadClientLike(Protocol):
    """Subset of GraphClient the correlation join needs (structural typing)."""

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]: ...


def _read_rested_map(
    client: ReadClientLike,
) -> dict[tuple[int, int], tuple[float, str | None]]:
    """Map ``(athlete_id, event_id) -> (rested_index, travel_direction)`` from the graph.

    Rows missing an athlete id, event id, or rested index are dropped. Returns an
    empty map (driving the graceful ``n = 0`` path) when no RestednessState nodes
    exist yet.
    """
    rested: dict[tuple[int, int], tuple[float, str | None]] = {}
    for row in client.run_read(REST_QUERY):
        athlete_id = row.get("athlete_id")
        event_id = row.get("event_id")
        rested_index = row.get("rested_index")
        if athlete_id is None or event_id is None or rested_index is None:
            continue
        rested[(int(athlete_id), int(event_id))] = (
            float(rested_index),
            row.get("travel_direction"),
        )
    return rested


def correlate_against_rested(
    client: ReadClientLike,
    items: list[T],
    *,
    athlete_id: Callable[[T], int],
    event_id: Callable[[T], int],
    outcome: Callable[[T], float],
    discipline: Callable[[T], str | None],
    success_signal: str,
) -> dict[str, Any]:
    """Correlate ``RestednessState.rested_index`` against a per-item outcome.

    Reads the rested map from :data:`REST_QUERY`, joins each item by
    ``(athlete_id, event_id)``, and groups the joined samples into Pearson blocks.
    Items with no matching RestednessState are dropped. Returns::

        {
          "overall": {"pearson_r": float|None, "n": int},
          "by_discipline": {code: {"pearson_r": ..., "n": ...}, ...},
          "by_travel_direction": {dir: {"pearson_r": ..., "n": ...}, ...},
          "success_signal": <the caller-supplied string>,
        }

    Each block's ``pearson_r`` is :func:`climber_network.stats.pearson` over the
    rested index (x) vs the outcome (y); it is ``None`` for ``n < 2`` or zero
    variance. ``by_discipline`` / ``by_travel_direction`` only include keys whose
    value is truthy (an empty / ``None`` discipline or direction is skipped).
    """
    rested = _read_rested_map(client)

    # Joined samples: (rested_index, outcome, discipline, travel_direction).
    pairs: list[tuple[float, float, str | None, str | None]] = []
    for item in items:
        match = rested.get((athlete_id(item), event_id(item)))
        if match is None:
            continue
        rested_index, travel_direction = match
        pairs.append((rested_index, outcome(item), discipline(item) or None, travel_direction))

    by_discipline: dict[str, list[tuple[float, float]]] = defaultdict(list)
    by_direction: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rested_index, value, disc, direction in pairs:
        if disc:
            by_discipline[disc].append((rested_index, value))
        if direction:
            by_direction[direction].append((rested_index, value))

    overall = [(rested_index, value) for rested_index, value, _, _ in pairs]

    return {
        "overall": _block(overall),
        "by_discipline": {k: _block(v) for k, v in by_discipline.items()},
        "by_travel_direction": {k: _block(v) for k, v in by_direction.items()},
        "success_signal": success_signal,
    }


def _block(samples: list[tuple[float, float]]) -> dict[str, Any]:
    """Build a ``{pearson_r, n}`` block from a list of ``(x, y)`` samples."""
    xs = [x for x, _ in samples]
    ys = [y for _, y in samples]
    return {"pearson_r": stats.pearson(xs, ys), "n": len(samples)}
