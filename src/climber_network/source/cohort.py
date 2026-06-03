"""climber_network.source.cohort — MVP cohort scoping for the source mirror.

Computes a small, internally-coherent slice of the upstream climbing-elo data so
the whole pipeline can be exercised cheaply (the full multi-season graph is slow
and costly to sync, simulate, and store). The slice is:

Cohort
    The ``n_per_gender`` most-active men + women who competed **both** in the
    *current* season **and** in at least one *earlier* season (i.e. active this
    season with a track record). Ranked, per gender, by distinct-event count
    within the recent-season window; deterministic tie-break on athlete id.

Full-field, season-bounded scope
    For every event a cohort athlete entered **within the recent-season window**,
    the *entire* field is kept — all rounds, all results, all athletes (cohort +
    opponents), and their ratings / rating history. Keeping the full per-round
    field is required for correctness: ``expected_rank``, the Monte-Carlo PMFs and
    the advancement projection all simulate against each round's full roster
    (:func:`climber_network.elo.reps.round_rosters`, #63/#65); thinning a round to
    just the cohort would silently corrupt every residual.

The resulting :class:`CohortScope` (athlete / event / round id sets) is applied at
the single source-read chokepoint (:func:`climber_network.source.pg.iter_rows`)
via :func:`filter_query`, so all four source-reading syncs see one consistent
subset and the graph-reading layers (geo / travel / season) shrink automatically.

Isolation
    Self-contained with respect to the CLAUDE.md isolation rule: imports only the
    read-only mirror models in :mod:`climber_network.source.pg` and the
    :class:`~climber_network.config.CohortParams` constants — never
    ``climbing_elo`` / ``knowledge_graph``. Issues no writes or DDL.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Query

from climber_network.config import CohortParams
from climber_network.source import pg


@dataclass(frozen=True)
class CohortScope:
    """The id sets defining a cohort slice of the source data.

    Every source read is constrained to these ids by :func:`filter_query`:

    * ``athlete_ids`` — the full field (cohort + opponents) appearing in scope.
    * ``event_ids``   — in-window events a cohort athlete entered.
    * ``round_ids``   — every round of those events.
    """

    athlete_ids: frozenset[int]
    event_ids: frozenset[int]
    round_ids: frozenset[int]


_EMPTY = CohortScope(frozenset(), frozenset(), frozenset())


def _distinct_athletes_in_seasons(
    session: pg.Session, *, lo: int | None, hi: int | None, exact: int | None = None
) -> set[int]:
    """Return athlete ids with a non-DNS result in events of the given season range.

    Pass ``exact`` for a single season; otherwise ``lo``/``hi`` bound it (either
    may be ``None`` for an open end). DNS rows are excluded — they did not compete.
    """
    q = (
        session.query(pg.Result.athlete_id)
        .join(pg.Round, pg.Round.id == pg.Result.round_id)
        .join(pg.Event, pg.Event.id == pg.Round.event_id)
        .filter(pg.Result.dns.is_(False))
    )
    if exact is not None:
        q = q.filter(pg.Event.season == exact)
    if lo is not None:
        q = q.filter(pg.Event.season >= lo)
    if hi is not None:
        q = q.filter(pg.Event.season <= hi)
    return {aid for (aid,) in q.distinct()}


def _select_cohort(
    session: pg.Session, eligible: set[int], *, lo: int, hi: int, n_per_gender: int
) -> set[int]:
    """Pick the top ``n_per_gender`` eligible athletes per gender by in-window events.

    Ranking key per athlete: distinct count of in-window events they competed in
    (non-DNS), descending; ties broken by ascending athlete id (deterministic).
    """
    rows = (
        session.query(
            pg.Athlete.id,
            pg.Athlete.gender,
            func.count(func.distinct(pg.Event.id)).label("n"),
        )
        .join(pg.Result, pg.Result.athlete_id == pg.Athlete.id)
        .join(pg.Round, pg.Round.id == pg.Result.round_id)
        .join(pg.Event, pg.Event.id == pg.Round.event_id)
        .filter(
            pg.Athlete.id.in_(sorted(eligible)),
            pg.Event.season >= lo,
            pg.Event.season <= hi,
            pg.Result.dns.is_(False),
        )
        .group_by(pg.Athlete.id, pg.Athlete.gender)
        .all()
    )
    by_gender: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ath_id, gender, n in rows:
        by_gender[str(gender)].append((int(ath_id), int(n)))

    cohort: set[int] = set()
    for members in by_gender.values():
        members.sort(key=lambda t: (-t[1], t[0]))
        cohort.update(ath_id for ath_id, _ in members[:n_per_gender])
    return cohort


def resolve_scope(session: pg.Session, params: CohortParams) -> CohortScope:
    """Compute the :class:`CohortScope` for *params* against the source *session*.

    Read-only: runs a handful of small aggregate / DISTINCT queries and returns id
    sets. Returns an **empty** scope (selecting nothing) when the source has no
    eligible athletes — callers should treat that as "no data" rather than "all
    data". See the module docstring for the selection semantics.
    """
    current_season = params.current_season
    if current_season is None:
        current_season = session.query(func.max(pg.Event.season)).scalar()
    if current_season is None:
        return _EMPTY  # empty source store
    current_season = int(current_season)
    window_lo = current_season - (params.season_window - 1)

    # Eligible = active this season AND has a prior-season track record.
    active_now = _distinct_athletes_in_seasons(session, lo=None, hi=None, exact=current_season)
    has_history = _distinct_athletes_in_seasons(session, lo=None, hi=current_season - 1)
    eligible = active_now & has_history
    if not eligible:
        return _EMPTY

    cohort_ids = _select_cohort(
        session, eligible, lo=window_lo, hi=current_season, n_per_gender=params.n_per_gender
    )
    if not cohort_ids:
        return _EMPTY

    # In-window events any cohort athlete entered (any result row — even a DNS
    # entry counts as "entered" for scoping the event in).
    event_ids = {
        eid
        for (eid,) in session.query(pg.Event.id)
        .join(pg.Round, pg.Round.event_id == pg.Event.id)
        .join(pg.Result, pg.Result.round_id == pg.Round.id)
        .filter(
            pg.Event.season >= window_lo,
            pg.Event.season <= current_season,
            pg.Result.athlete_id.in_(sorted(cohort_ids)),
        )
        .distinct()
    }
    if not event_ids:
        return _EMPTY

    # Every round of those events.
    round_ids = {
        rid
        for (rid,) in session.query(pg.Round.id).filter(pg.Round.event_id.in_(sorted(event_ids)))
    }
    # Full field: every athlete with a result in those rounds (cohort + opponents).
    athlete_ids = {
        aid
        for (aid,) in session.query(pg.Result.athlete_id)
        .filter(pg.Result.round_id.in_(sorted(round_ids)))
        .distinct()
    }

    return CohortScope(
        athlete_ids=frozenset(athlete_ids),
        event_ids=frozenset(event_ids),
        round_ids=frozenset(round_ids),
    )


def filter_query(query: Query[pg.Base], model: type[pg.Base], scope: CohortScope) -> Query[pg.Base]:
    """Constrain a per-*model* source ``Query`` to *scope* via its natural key.

    Each read model joins the scope through one column: athletes/ratings by
    athlete id, events by event id, rounds/results/rating-history by round id.
    Models with no scope key (none currently) are returned unfiltered.
    """
    if model is pg.Athlete:
        return query.filter(pg.Athlete.id.in_(sorted(scope.athlete_ids)))
    if model is pg.Event:
        return query.filter(pg.Event.id.in_(sorted(scope.event_ids)))
    if model is pg.Round:
        return query.filter(pg.Round.id.in_(sorted(scope.round_ids)))
    if model is pg.Result:
        return query.filter(pg.Result.round_id.in_(sorted(scope.round_ids)))
    if model is pg.Rating:
        return query.filter(pg.Rating.athlete_id.in_(sorted(scope.athlete_ids)))
    if model is pg.RatingHistory:
        return query.filter(pg.RatingHistory.round_id.in_(sorted(scope.round_ids)))
    return query


def scope_from_config(session: pg.Session, *, override: bool | None = None) -> CohortScope | None:
    """Resolve a :class:`CohortScope` from config/CLI, or ``None`` when disabled.

    *override* is the tri-state ``--cohort/--no-cohort`` CLI value: ``None`` defers
    to :func:`climber_network.config.COHORT_ENABLED`. When the slice is disabled,
    returns ``None`` (the syncs then read the full unfiltered source).
    """
    from climber_network import config

    enabled = override if override is not None else config.COHORT_ENABLED()
    if not enabled:
        return None
    return resolve_scope(session, config.cohort_params_from_env())
