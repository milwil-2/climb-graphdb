"""Tests for the MVP cohort slice (`climber_network.source.cohort`).

Exercises the cohort selection, full-field season-bounded scoping, the
``iter_rows`` chokepoint filter, and end-to-end consistency through the L1 mirror
— all offline on in-memory SQLite (no network), using the shared
``source_session`` fixture + ``FakeGraphClient`` recorder from ``conftest``.

The seeded scenario (current season auto-detects to 2026, window = 2023–2026,
``n_per_gender = 2``) is engineered so every selection rule is observable:

* **m1** (id 1) — 3 in-window events → the clear top man.
* **m2** (id 2) and **m3** (id 3) — tied at 2 in-window events; the id tie-break
  selects **m2**. Each has a *private* event with a unique opponent (x2 / x3), so
  whichever man is in the cohort pulls his opponent into the full field — making
  the tie-break directly observable (x2 in, x3 out).
* **m4** (id 4) — competes only in 2026 (no history) → not eligible, but appears
  as an opponent in a cohort event, so the full-field rule still includes him.
* **m5** (id 5) — history only, isolated in an out-of-window event → excluded.
* **xold** (id 8) — opponent only in a pre-window (2019) event → excluded by the
  season bound.
* **f1 / f2** (ids 11/12) — both eligible women; selected per-gender even though
  their in-window counts (2 each) match the *excluded* man m3 — proving ranking
  is within gender, not global.
"""

from __future__ import annotations

from datetime import date

import pytest

from climber_network.config import CohortParams
from climber_network.source import cohort, pg
from climber_network.source.cohort import CohortScope, resolve_scope
from sync.pg_to_neo4j import sync_graph, validate_counts
from tests.conftest import FakeGraphClient

# Event ids 1xx, round ids 2xx, athlete ids in 1–12 — disjoint ranges so an
# assertion reads unambiguously.
_M1, _M2, _M3, _M4, _M5, _X2, _X3, _XOLD = 1, 2, 3, 4, 5, 6, 7, 8
_F1, _F2 = 11, 12

_E_PRE, _E_OLD, _E_SH1, _E2PRIV, _E_SH2, _E3PRIV, _E_CUR = 101, 102, 103, 104, 105, 106, 107


def _seed_cohort_world(session: pg.Session) -> None:
    """Seed the engineered multi-season competition described in the module docstring."""
    session.add_all(
        [
            pg.Athlete(id=_M1, name="m1", gender="M"),
            pg.Athlete(id=_M2, name="m2", gender="M"),
            pg.Athlete(id=_M3, name="m3", gender="M"),
            pg.Athlete(id=_M4, name="m4", gender="M"),
            pg.Athlete(id=_M5, name="m5", gender="M"),
            pg.Athlete(id=_X2, name="x2", gender="M"),
            pg.Athlete(id=_X3, name="x3", gender="M"),
            pg.Athlete(id=_XOLD, name="xold", gender="M"),
            pg.Athlete(id=_F1, name="f1", gender="F"),
            pg.Athlete(id=_F2, name="f2", gender="F"),
        ]
    )

    def _event(eid: int, season: int) -> pg.Event:
        return pg.Event(
            id=eid,
            name=f"Event {eid}",
            tier="world_cup",
            season=season,
            start_date=date(season, 6, 1),
            discipline="L",
        )

    session.add_all(
        [
            _event(_E_PRE, 2019),
            _event(_E_OLD, 2022),
            _event(_E_SH1, 2024),
            _event(_E2PRIV, 2024),
            _event(_E_SH2, 2025),
            _event(_E3PRIV, 2025),
            _event(_E_CUR, 2026),
        ]
    )
    # One final round per event; round id = event id + 100. Round.gender is unused
    # by cohort selection (which keys off Athlete.gender).
    session.add_all(
        [
            pg.Round(id=eid + 100, event_id=eid, round_type="final", gender="M")
            for eid in (_E_PRE, _E_OLD, _E_SH1, _E2PRIV, _E_SH2, _E3PRIV, _E_CUR)
        ]
    )

    # (round_id, [athlete_ids]) — the full field of each round.
    rosters: list[tuple[int, list[int]]] = [
        (_E_PRE + 100, [_M1, _XOLD]),
        (_E_OLD + 100, [_M5]),
        (_E_SH1 + 100, [_M1, _F1]),
        (_E2PRIV + 100, [_M2, _X2]),
        (_E_SH2 + 100, [_M1, _F2]),
        (_E3PRIV + 100, [_M3, _X3]),
        (_E_CUR + 100, [_M1, _M2, _M3, _M4, _F1, _F2]),
    ]
    rid = 1
    for round_id, athletes in rosters:
        for pos, athlete_id in enumerate(athletes, start=1):
            session.add(pg.Result(id=rid, round_id=round_id, athlete_id=athlete_id, rank=pos))
            rid += 1
    session.commit()


@pytest.fixture
def cohort_session(source_session: pg.Session) -> pg.Session:
    """A source session seeded with the engineered cohort scenario."""
    _seed_cohort_world(source_session)
    return source_session


_PARAMS = CohortParams(n_per_gender=2)  # window defaults to 4 → 2023–2026.


# ---------------------------------------------------------------------------
# resolve_scope — selection + full-field + season bound.
# ---------------------------------------------------------------------------


def test_resolve_scope_event_and_round_sets(cohort_session: pg.Session) -> None:
    scope = resolve_scope(cohort_session, _PARAMS)
    # In-window events a COHORT athlete entered: shared(103), m2-private(104),
    # shared(105), current(107). m3's private event 106 is absent (m3 lost the
    # tie-break); pre-window 101/102 are absent (season bound).
    assert scope.event_ids == frozenset({_E_SH1, _E2PRIV, _E_SH2, _E_CUR})
    assert scope.round_ids == frozenset({203, 204, 205, 207})


def test_resolve_scope_full_field_includes_non_cohort_opponents(
    cohort_session: pg.Session,
) -> None:
    scope = resolve_scope(cohort_session, _PARAMS)
    # Full field of the in-scope rounds: cohort + every opponent in those rounds.
    assert scope.athlete_ids == frozenset({_M1, _M2, _M3, _M4, _X2, _F1, _F2})
    # m3 (lost the tie-break) and m4 (not eligible) are still present as opponents
    # in the current event — the full-field rule keeps them.
    assert _M3 in scope.athlete_ids
    assert _M4 in scope.athlete_ids


def test_resolve_scope_tiebreak_is_observable(cohort_session: pg.Session) -> None:
    scope = resolve_scope(cohort_session, _PARAMS)
    # m2 beat m3 on the id tie-break → m2's private event (104) is in scope, so its
    # unique opponent x2 is pulled into the field; m3's private event (106) is not,
    # so x3 never appears. This is the only thing that distinguishes the two.
    assert _X2 in scope.athlete_ids
    assert _X3 not in scope.athlete_ids
    assert _E3PRIV not in scope.event_ids


def test_resolve_scope_excludes_ineligible_and_out_of_window(
    cohort_session: pg.Session,
) -> None:
    scope = resolve_scope(cohort_session, _PARAMS)
    # m5 (history only, isolated) and xold (pre-window opponent) never enter scope.
    assert _M5 not in scope.athlete_ids
    assert _XOLD not in scope.athlete_ids
    # The out-of-window events are excluded even though m1 (a cohort athlete)
    # competed in the 2019 one.
    assert _E_PRE not in scope.event_ids
    assert _E_OLD not in scope.event_ids


def test_resolve_scope_respects_per_gender_ranking(cohort_session: pg.Session) -> None:
    # Both women are selected at in-window count 2 — the same count that got m3
    # excluded among the men — proving the top-N is taken within each gender.
    scope = resolve_scope(cohort_session, _PARAMS)
    assert {_F1, _F2} <= scope.athlete_ids


def test_resolve_scope_explicit_current_season_shifts_window(
    cohort_session: pg.Session,
) -> None:
    # Anchor "this season" at 2025 instead of auto 2026: now 2026 events are in the
    # future and excluded, and eligibility requires activity in 2025 + prior.
    scope = resolve_scope(cohort_session, CohortParams(n_per_gender=2, current_season=2025))
    assert _E_CUR not in scope.event_ids
    assert all(eid != _E_CUR for eid in scope.event_ids)


def test_resolve_scope_empty_store_returns_empty(source_session: pg.Session) -> None:
    scope = resolve_scope(source_session, _PARAMS)
    assert scope == CohortScope(frozenset(), frozenset(), frozenset())


# ---------------------------------------------------------------------------
# filter_query / iter_rows — the chokepoint.
# ---------------------------------------------------------------------------


def test_filter_query_keys_each_model(cohort_session: pg.Session) -> None:
    scope = CohortScope(
        athlete_ids=frozenset({_M1, _M2}),
        event_ids=frozenset({_E_SH1}),
        round_ids=frozenset({203, 204}),
    )
    athletes = {a.id for a in pg.iter_rows(cohort_session, pg.Athlete, scope=scope)}
    events = {e.id for e in pg.iter_rows(cohort_session, pg.Event, scope=scope)}
    rounds = {r.id for r in pg.iter_rows(cohort_session, pg.Round, scope=scope)}
    results = {r.round_id for r in pg.iter_rows(cohort_session, pg.Result, scope=scope)}

    assert athletes == {_M1, _M2}  # Athlete keyed by id
    assert events == {_E_SH1}  # Event keyed by id
    assert rounds == {203, 204}  # Round keyed by id
    assert results <= {203, 204}  # Result keyed by round_id


def test_filter_query_keys_ratings_and_history(source_session: pg.Session) -> None:
    source_session.add_all(
        [
            pg.Rating(id=1, athlete_id=_M1, discipline="L", mu=1500.0, sigma=200.0),
            pg.Rating(id=2, athlete_id=_M5, discipline="L", mu=1400.0, sigma=200.0),
            pg.RatingHistory(
                id=1,
                athlete_id=_M1,
                event_id=_E_SH1,
                round_id=203,
                mu_before=1500.0,
                mu_after=1510.0,
                sigma_before=200.0,
                sigma_after=190.0,
            ),
            pg.RatingHistory(
                id=2,
                athlete_id=_M5,
                event_id=_E_OLD,
                round_id=202,
                mu_before=1400.0,
                mu_after=1390.0,
                sigma_before=200.0,
                sigma_after=205.0,
            ),
        ]
    )
    source_session.commit()
    scope = CohortScope(
        athlete_ids=frozenset({_M1}), event_ids=frozenset(), round_ids=frozenset({203})
    )
    ratings = {r.athlete_id for r in pg.iter_rows(source_session, pg.Rating, scope=scope)}
    history = {h.round_id for h in pg.iter_rows(source_session, pg.RatingHistory, scope=scope)}
    assert ratings == {_M1}  # Rating keyed by athlete_id
    assert history == {203}  # RatingHistory keyed by round_id


def test_iter_rows_scope_none_is_unchanged(cohort_session: pg.Session) -> None:
    # No scope → every row (10 athletes seeded).
    assert len(list(pg.iter_rows(cohort_session, pg.Athlete))) == 10
    # An explicit empty scope selects nothing — distinct from scope=None.
    empty = CohortScope(frozenset(), frozenset(), frozenset())
    assert list(pg.iter_rows(cohort_session, pg.Athlete, scope=empty)) == []


# ---------------------------------------------------------------------------
# scope_from_config — the CLI/env toggle.
# ---------------------------------------------------------------------------


def test_scope_from_config_override_false_disables(cohort_session: pg.Session) -> None:
    assert cohort.scope_from_config(cohort_session, override=False) is None


def test_scope_from_config_defers_to_env(
    cohort_session: pg.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COHORT_ENABLED", raising=False)
    assert cohort.scope_from_config(cohort_session, override=None) is None
    monkeypatch.setenv("COHORT_ENABLED", "true")
    scope = cohort.scope_from_config(cohort_session, override=None)
    assert scope is not None and scope.athlete_ids


def test_scope_from_config_override_true_resolves(
    cohort_session: pg.Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COHORT_ENABLED", raising=False)  # override wins over env
    scope = cohort.scope_from_config(cohort_session, override=True)
    assert scope is not None and scope.athlete_ids


# ---------------------------------------------------------------------------
# End-to-end through the L1 mirror — scope is internally consistent.
# ---------------------------------------------------------------------------


def test_sync_graph_with_scope_mirrors_only_the_slice(cohort_session: pg.Session) -> None:
    scope = resolve_scope(cohort_session, _PARAMS)
    client = FakeGraphClient()
    report = sync_graph(client, cohort_session, scope=scope)

    athlete_nodes = {nid for nid, label in client.node_labels.items() if label == "Athlete"}
    assert athlete_nodes == {vocab_ath for vocab_ath in (f"ath:{i}" for i in scope.athlete_ids)}
    assert "ath:5" not in athlete_nodes  # m5 (out of slice) is absent.
    # Counts come from the filtered reads, so validation still balances.
    assert report.node_athletes == report.src_athletes == len(scope.athlete_ids)
    validate_counts(report)  # raises on unexplained drift
