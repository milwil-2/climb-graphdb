"""climber_network.elo.reps — Shared representative-round + rating-lookup helpers.

This module extracts the logic for selecting the *representative round* per
(athlete, event) and for building point-in-time rating lookups from the source
``rating_history`` table, so that both ``sync.validate_elo`` and the forthcoming
``sync.montecarlo`` can share the same semantics without duplication.

Public API
----------
* :data:`ROUND_DEPTH` — mapping from round-type string → ordinal depth.
* :class:`RepRound` — frozen dataclass holding one chosen representative round.
* :func:`select_representative_rounds` — pick the deepest eligible round per
  (athlete, event), tallying skip reasons into a caller-supplied counter dict.
* :func:`mu_before_lookup` — build a ``(athlete_id, round_id) → mu_before`` map.
* :func:`sigma_before_lookup` — build a ``(athlete_id, round_id) → sigma_before`` map.

Isolation
---------
This module is self-contained with respect to the isolation constraint in
CLAUDE.md: it may import ``climber_network.source.pg`` (read-only mirror models)
but must NEVER import ``climbing_elo`` or ``knowledge_graph``.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping
from dataclasses import dataclass

from climber_network.source import pg

__all__ = [
    "ROUND_DEPTH",
    "RepRound",
    "select_representative_rounds",
    "mu_before_lookup",
    "sigma_before_lookup",
]

# ---------------------------------------------------------------------------
# Round-type depth ordering.
# ---------------------------------------------------------------------------

#: Round-type depth ordering. A larger number is a deeper (more selective) round,
#: so the representative round per (athlete, event) is the one with the max depth.
ROUND_DEPTH: dict[str, int] = {
    "qualification": 0,
    "qual": 0,
    "semi": 1,
    "semifinal": 1,
    "final": 2,
}


def _round_depth(round_type: str) -> int:
    """Return the selection depth of *round_type* (unknown types sort lowest)."""
    return ROUND_DEPTH.get(round_type.lower(), -1)


# ---------------------------------------------------------------------------
# RepRound dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepRound:
    """A chosen representative round for one (athlete, event).

    Fields
    ------
    athlete_id
        Integer primary key of the athlete in the source store.
    event_id
        Integer primary key of the event in the source store.
    round_id
        Integer primary key of the chosen round (the deepest round reached).
    round_type
        String round type (e.g. ``"final"``, ``"semi"``, ``"qualification"``).
    discipline
        Event discipline code (e.g. ``"L"``, ``"B"``, ``"S"``), or ``""`` if
        the event row was not found.
    actual_rank
        The athlete's actual finishing rank in this round.
    expected_rank
        The model's expected finishing rank (filled in by the caller; initialised
        to :data:`math.nan` when first constructed by
        :func:`select_representative_rounds`).
    elo_residual
        ``actual_rank - expected_rank`` (filled in by the caller; initialised to
        :data:`math.nan` when first constructed).
    """

    athlete_id: int
    event_id: int
    round_id: int
    round_type: str
    discipline: str
    actual_rank: int
    expected_rank: float
    elo_residual: float


# ---------------------------------------------------------------------------
# Representative-round selection.
# ---------------------------------------------------------------------------


def select_representative_rounds(
    session: pg.Session,
    skipped: MutableMapping[str, int],
    *,
    src_rounds_out: list[int] | None = None,
    src_results_out: list[int] | None = None,
) -> list[RepRound]:
    """Pick the representative round per (athlete, event) from the source data.

    The deepest round each athlete reached (final > semi > qualification); ties
    on depth are broken by the larger ``round_id`` (deterministic). Rounds the
    athlete did not start or has no usable rank for are not eligible.

    Parameters
    ----------
    session:
        A read-only SQLAlchemy session bound to the source (climbing-elo) database.
    skipped:
        A mutable counter mapping skip-reason strings to counts. Updated in
        place. Keys emitted:

        * ``"result_round_missing"`` — the result's ``round_id`` has no
          matching row in the rounds table.
        * ``"result_no_rank_or_dns"`` — the result is a DNS or has no rank.
    src_rounds_out:
        Optional one-element list; if supplied, its first element is replaced
        with the total number of round rows read (for reporting).
    src_results_out:
        Optional one-element list; if supplied, its first element is replaced
        with the total number of result rows read (for reporting).

    Returns
    -------
    list[RepRound]
        One :class:`RepRound` per (athlete, event). ``expected_rank`` and
        ``elo_residual`` are both :data:`math.nan` — the caller is responsible
        for filling these in.
    """
    rounds = list(pg.iter_rows(session, pg.Round))
    results = list(pg.iter_rows(session, pg.Result))
    events = list(pg.iter_rows(session, pg.Event))

    if src_rounds_out is not None:
        src_rounds_out[0] = len(rounds)
    if src_results_out is not None:
        src_results_out[0] = len(results)

    rounds_by_id: dict[int, pg.Round] = {}
    for r in rounds:
        assert isinstance(r, pg.Round)
        rounds_by_id[r.id] = r
    event_discipline: dict[int, str] = {}
    for e in events:
        assert isinstance(e, pg.Event)
        event_discipline[e.id] = e.discipline

    # Best (deepest) eligible round per (athlete, event).
    # value tuple: (depth, round_id, actual_rank, round_id_for_tiebreak).
    best: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for res in results:
        assert isinstance(res, pg.Result)
        rnd_row = rounds_by_id.get(res.round_id)
        if rnd_row is None:
            skipped["result_round_missing"] += 1
            continue
        if res.dns or res.rank is None:
            # Did not start, or no placement → not an eligible finish.
            skipped["result_no_rank_or_dns"] += 1
            continue
        event_id = rnd_row.event_id
        depth = _round_depth(rnd_row.round_type)
        key = (res.athlete_id, event_id)
        candidate = (depth, res.round_id, res.rank, res.round_id)
        current = best.get(key)
        if current is None or (depth, res.round_id) > (current[0], current[3]):
            best[key] = candidate

    reps: list[RepRound] = []
    for (athlete_id, event_id), (_depth, round_id, actual_rank, _tb) in best.items():
        rnd_row = rounds_by_id[round_id]
        reps.append(
            RepRound(
                athlete_id=athlete_id,
                event_id=event_id,
                round_id=round_id,
                round_type=rnd_row.round_type,
                discipline=event_discipline.get(event_id, ""),
                actual_rank=actual_rank,
                expected_rank=math.nan,
                elo_residual=math.nan,
            )
        )
    return reps


# ---------------------------------------------------------------------------
# Rating-history lookups.
# ---------------------------------------------------------------------------


def mu_before_lookup(session: pg.Session) -> dict[tuple[int, int], float]:
    """Map ``(athlete_id, round_id)`` → pre-event ``mu_before`` from rating_history.

    Point-in-time μ as of that round, read READ-ONLY from the source store. If a
    duplicate (athlete, round) row appears, the last row in primary-key order wins
    (stable, because :func:`~climber_network.source.pg.iter_rows` yields in pk
    order).

    Parameters
    ----------
    session:
        A read-only SQLAlchemy session bound to the source (climbing-elo) database.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping from ``(athlete_id, round_id)`` to the ``mu_before`` value.
    """
    out: dict[tuple[int, int], float] = {}
    for h in pg.iter_rows(session, pg.RatingHistory):
        assert isinstance(h, pg.RatingHistory)
        out[(h.athlete_id, h.round_id)] = h.mu_before
    return out


def sigma_before_lookup(session: pg.Session) -> dict[tuple[int, int], float]:
    """Map ``(athlete_id, round_id)`` → pre-event ``sigma_before`` from rating_history.

    Point-in-time σ as of that round, read READ-ONLY from the source store. If a
    duplicate (athlete, round) row appears, the last row in primary-key order wins
    (stable, because :func:`~climber_network.source.pg.iter_rows` yields in pk
    order).

    Parameters
    ----------
    session:
        A read-only SQLAlchemy session bound to the source (climbing-elo) database.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping from ``(athlete_id, round_id)`` to the ``sigma_before`` value.
    """
    out: dict[tuple[int, int], float] = {}
    for h in pg.iter_rows(session, pg.RatingHistory):
        assert isinstance(h, pg.RatingHistory)
        out[(h.athlete_id, h.round_id)] = h.sigma_before
    return out
