"""Tests for climber_network.elo.reps — representative-round selection + rating lookups.

These tests verify that:
* :func:`select_representative_rounds` picks the deepest round reached per
  (athlete, event), breaking ties on the larger ``round_id``, and correctly
  tallies dns / no-rank skips in the supplied counter.
* :func:`mu_before_lookup` returns the correct ``(athlete_id, round_id)``→ value map.
* :func:`sigma_before_lookup` returns the correct ``(athlete_id, round_id)``→ value map.

Fixtures and seeding patterns follow the conventions established in
``tests/conftest.py`` and ``tests/test_validate_elo.py``.  The ``source_session``
and ``seeded_session`` fixtures come from ``conftest.py`` (in-memory SQLite).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

import pytest

from climber_network.elo.reps import (
    ROUND_DEPTH,
    RepRound,
    mu_before_lookup,
    select_representative_rounds,
    sigma_before_lookup,
)
from climber_network.source import pg

# ---------------------------------------------------------------------------
# Seeding helper — a small but representative competition.
# ---------------------------------------------------------------------------


def _seed(session: pg.Session) -> None:
    """Seed a 3-round, 4-athlete competition with rating history for tests."""
    session.add_all(
        [
            pg.Athlete(id=1, name="Ada", gender="F", nationality="USA"),
            pg.Athlete(id=2, name="Bea", gender="F", nationality="GBR"),
            pg.Athlete(id=3, name="Cleo", gender="F", nationality="JPN"),
            pg.Athlete(id=4, name="Dot", gender="F", nationality="AUT"),
        ]
    )
    session.add(
        pg.Event(
            id=1,
            name="World Cup Innsbruck",
            tier="world_cup",
            country="AUT",
            season=2024,
            start_date=date(2024, 6, 1),
            discipline="L",
        )
    )
    session.add_all(
        [
            pg.Round(id=1, event_id=1, round_type="qualification", gender="F", athlete_count=4),
            pg.Round(id=2, event_id=1, round_type="semi", gender="F", athlete_count=3),
            pg.Round(id=3, event_id=1, round_type="final", gender="F", athlete_count=2),
        ]
    )
    session.add_all(
        [
            # Qualification (round 1): all four athletes.
            pg.Result(id=1, round_id=1, athlete_id=1, rank=1),
            pg.Result(id=2, round_id=1, athlete_id=2, rank=2),
            pg.Result(id=3, round_id=1, athlete_id=3, rank=3),
            pg.Result(id=4, round_id=1, athlete_id=4, rank=4),
            # Semi (round 2): athletes 1, 2, 3.
            pg.Result(id=5, round_id=2, athlete_id=1, rank=1),
            pg.Result(id=6, round_id=2, athlete_id=2, rank=2),
            pg.Result(id=7, round_id=2, athlete_id=3, rank=3),
            # Final (round 3): athletes 1, 2.
            pg.Result(id=8, round_id=3, athlete_id=1, rank=2),
            pg.Result(id=9, round_id=3, athlete_id=2, rank=1),
        ]
    )
    # Rating history: point-in-time mu/sigma per (athlete, round).
    session.add_all(
        [
            # Qualification (round 1).
            pg.RatingHistory(
                id=1,
                athlete_id=1,
                event_id=1,
                round_id=1,
                mu_before=1700.0,
                mu_after=1710.0,
                sigma_before=100.0,
                sigma_after=98.0,
            ),
            pg.RatingHistory(
                id=2,
                athlete_id=2,
                event_id=1,
                round_id=1,
                mu_before=1500.0,
                mu_after=1495.0,
                sigma_before=110.0,
                sigma_after=109.0,
            ),
            pg.RatingHistory(
                id=3,
                athlete_id=3,
                event_id=1,
                round_id=1,
                mu_before=1450.0,
                mu_after=1445.0,
                sigma_before=120.0,
                sigma_after=119.0,
            ),
            pg.RatingHistory(
                id=4,
                athlete_id=4,
                event_id=1,
                round_id=1,
                mu_before=1400.0,
                mu_after=1395.0,
                sigma_before=130.0,
                sigma_after=129.0,
            ),
            # Semi (round 2).
            pg.RatingHistory(
                id=5,
                athlete_id=1,
                event_id=1,
                round_id=2,
                mu_before=1710.0,
                mu_after=1720.0,
                sigma_before=98.0,
                sigma_after=96.0,
            ),
            pg.RatingHistory(
                id=6,
                athlete_id=2,
                event_id=1,
                round_id=2,
                mu_before=1495.0,
                mu_after=1500.0,
                sigma_before=109.0,
                sigma_after=107.0,
            ),
            pg.RatingHistory(
                id=7,
                athlete_id=3,
                event_id=1,
                round_id=2,
                mu_before=1445.0,
                mu_after=1440.0,
                sigma_before=119.0,
                sigma_after=117.0,
            ),
            # Final (round 3).
            pg.RatingHistory(
                id=8,
                athlete_id=1,
                event_id=1,
                round_id=3,
                mu_before=1720.0,
                mu_after=1715.0,
                sigma_before=96.0,
                sigma_after=95.0,
            ),
            pg.RatingHistory(
                id=9,
                athlete_id=2,
                event_id=1,
                round_id=3,
                mu_before=1500.0,
                mu_after=1510.0,
                sigma_before=107.0,
                sigma_after=105.0,
            ),
        ]
    )
    session.commit()


# ---------------------------------------------------------------------------
# ROUND_DEPTH constant.
# ---------------------------------------------------------------------------


def test_round_depth_map_values() -> None:
    """ROUND_DEPTH assigns expected ordinals to each round type."""
    assert ROUND_DEPTH["qualification"] == 0
    assert ROUND_DEPTH["qual"] == 0
    assert ROUND_DEPTH["semi"] == 1
    assert ROUND_DEPTH["semifinal"] == 1
    assert ROUND_DEPTH["final"] == 2
    # final must be strictly deeper than semi, semi than qualification.
    assert ROUND_DEPTH["final"] > ROUND_DEPTH["semi"] > ROUND_DEPTH["qualification"]


# ---------------------------------------------------------------------------
# RepRound dataclass.
# ---------------------------------------------------------------------------


def test_rep_round_is_frozen() -> None:
    """RepRound is a frozen dataclass — attribute assignment must raise."""
    rep = RepRound(
        athlete_id=1,
        event_id=1,
        round_id=3,
        round_type="final",
        discipline="L",
        actual_rank=1,
        expected_rank=math.nan,
        elo_residual=math.nan,
    )
    with pytest.raises((AttributeError, TypeError)):
        rep.athlete_id = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# select_representative_rounds — deepest round selection.
# ---------------------------------------------------------------------------


def test_selects_deepest_round_per_athlete_event(source_session: pg.Session) -> None:
    """Athletes 1 & 2 reached the final; athlete 3 only the semi; athlete 4 only qual."""
    _seed(source_session)
    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(source_session, skipped)

    by_athlete = {rep.athlete_id: rep for rep in reps}

    assert by_athlete[1].round_id == 3
    assert by_athlete[1].round_type == "final"
    assert by_athlete[2].round_id == 3
    assert by_athlete[2].round_type == "final"
    assert by_athlete[3].round_id == 2
    assert by_athlete[3].round_type == "semi"
    assert by_athlete[4].round_id == 1
    assert by_athlete[4].round_type == "qualification"


def test_actual_rank_carried_through(source_session: pg.Session) -> None:
    """The actual_rank field on RepRound matches the result row for the chosen round."""
    _seed(source_session)
    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(source_session, skipped)
    by_athlete = {rep.athlete_id: rep for rep in reps}

    # In the final, athlete 2 ranks 1st and athlete 1 ranks 2nd.
    assert by_athlete[2].actual_rank == 1
    assert by_athlete[1].actual_rank == 2
    # Athlete 4 had rank 4 in qualification.
    assert by_athlete[4].actual_rank == 4


def test_expected_rank_and_residual_initialised_to_nan(source_session: pg.Session) -> None:
    """select_representative_rounds does not fill expected_rank / elo_residual."""
    _seed(source_session)
    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(source_session, skipped)
    for rep in reps:
        assert math.isnan(rep.expected_rank)
        assert math.isnan(rep.elo_residual)


def test_discipline_from_event(source_session: pg.Session) -> None:
    """The discipline field on RepRound is read from the Event row."""
    _seed(source_session)
    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(source_session, skipped)
    for rep in reps:
        assert rep.discipline == "L"  # Event 1 is discipline "L" (Lead).


def test_event_id_correct(source_session: pg.Session) -> None:
    """All reps should map to event 1."""
    _seed(source_session)
    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(source_session, skipped)
    for rep in reps:
        assert rep.event_id == 1


# ---------------------------------------------------------------------------
# skip-reason counting.
# ---------------------------------------------------------------------------


def test_dns_result_counted_as_skip(source_session: pg.Session) -> None:
    """A DNS result increments the 'result_no_rank_or_dns' skip counter."""
    session = source_session
    session.add(pg.Athlete(id=1, name="Ada", gender="F", nationality="USA"))
    session.add(
        pg.Event(
            id=1,
            name="Test Event",
            tier="world_cup",
            country="AUT",
            season=2024,
            start_date=date(2024, 6, 1),
            discipline="L",
        )
    )
    session.add(pg.Round(id=1, event_id=1, round_type="qualification", gender="F", athlete_count=1))
    # DNS result — athlete did not start.
    session.add(pg.Result(id=1, round_id=1, athlete_id=1, rank=None, dns=True))
    session.commit()

    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(session, skipped)

    assert reps == []
    assert skipped["result_no_rank_or_dns"] == 1


def test_null_rank_result_counted_as_skip(source_session: pg.Session) -> None:
    """A result with rank=None (and not DNS) is also skipped with the same reason key."""
    session = source_session
    session.add(pg.Athlete(id=1, name="Ada", gender="F", nationality="USA"))
    session.add(
        pg.Event(
            id=1,
            name="Test Event",
            tier="world_cup",
            country="AUT",
            season=2024,
            start_date=date(2024, 6, 1),
            discipline="L",
        )
    )
    session.add(pg.Round(id=1, event_id=1, round_type="qualification", gender="F", athlete_count=1))
    session.add(pg.Result(id=1, round_id=1, athlete_id=1, rank=None, dns=False))
    session.commit()

    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(session, skipped)

    assert reps == []
    assert skipped["result_no_rank_or_dns"] == 1


def test_result_with_missing_round_counted_as_skip(source_session: pg.Session) -> None:
    """A result whose round_id has no matching Round row increments 'result_round_missing'."""
    session = source_session
    session.add(pg.Athlete(id=1, name="Ada", gender="F", nationality="USA"))
    session.add(
        pg.Event(
            id=1,
            name="Test Event",
            tier="world_cup",
            country="AUT",
            season=2024,
            start_date=date(2024, 6, 1),
            discipline="L",
        )
    )
    # No Round row for round_id=99.
    session.add(pg.Result(id=1, round_id=99, athlete_id=1, rank=1))
    session.commit()

    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(session, skipped)

    assert reps == []
    assert skipped["result_round_missing"] == 1


# ---------------------------------------------------------------------------
# src_rounds_out / src_results_out optional output parameters.
# ---------------------------------------------------------------------------


def test_src_counts_populated_via_out_lists(source_session: pg.Session) -> None:
    """src_rounds_out and src_results_out are updated to reflect rows read."""
    _seed(source_session)
    skipped: dict[str, int] = defaultdict(int)
    rounds_out: list[int] = [0]
    results_out: list[int] = [0]
    select_representative_rounds(
        source_session,
        skipped,
        src_rounds_out=rounds_out,
        src_results_out=results_out,
    )
    # 3 rounds and 9 results were seeded.
    assert rounds_out[0] == 3
    assert results_out[0] == 9


# ---------------------------------------------------------------------------
# Tiebreak — larger round_id wins when depth is equal.
# ---------------------------------------------------------------------------


def test_tiebreak_prefers_larger_round_id(source_session: pg.Session) -> None:
    """When two rounds have the same type/depth, the larger round_id is chosen."""
    session = source_session
    session.add(pg.Athlete(id=1, name="Ada", gender="F", nationality="USA"))
    session.add(
        pg.Event(
            id=1,
            name="Test Event",
            tier="world_cup",
            country="AUT",
            season=2024,
            start_date=date(2024, 6, 1),
            discipline="B",
        )
    )
    # Two qualification rounds for the same event — same depth.
    session.add(
        pg.Round(id=10, event_id=1, round_type="qualification", gender="F", athlete_count=1)
    )
    session.add(
        pg.Round(id=20, event_id=1, round_type="qualification", gender="F", athlete_count=1)
    )
    session.add(pg.Result(id=1, round_id=10, athlete_id=1, rank=1))
    session.add(pg.Result(id=2, round_id=20, athlete_id=1, rank=2))
    session.commit()

    skipped: dict[str, int] = defaultdict(int)
    reps = select_representative_rounds(session, skipped)

    assert len(reps) == 1
    # round_id=20 is larger — should win the tiebreak.
    assert reps[0].round_id == 20


# ---------------------------------------------------------------------------
# mu_before_lookup.
# ---------------------------------------------------------------------------


def test_mu_before_lookup_returns_expected_values(source_session: pg.Session) -> None:
    """mu_before_lookup maps each (athlete_id, round_id) to the correct mu_before."""
    _seed(source_session)
    mu_map = mu_before_lookup(source_session)

    # Spot-check a few seeded values.
    assert mu_map[(1, 1)] == pytest.approx(1700.0)  # athlete 1, round 1 (qual)
    assert mu_map[(2, 1)] == pytest.approx(1500.0)  # athlete 2, round 1
    assert mu_map[(1, 3)] == pytest.approx(1720.0)  # athlete 1, round 3 (final)
    assert mu_map[(2, 3)] == pytest.approx(1500.0)  # athlete 2, round 3
    assert mu_map[(3, 2)] == pytest.approx(1445.0)  # athlete 3, round 2 (semi)


def test_mu_before_lookup_covers_all_seeded_rows(source_session: pg.Session) -> None:
    """mu_before_lookup returns one entry per seeded rating_history row."""
    _seed(source_session)
    mu_map = mu_before_lookup(source_session)
    # 9 RatingHistory rows were seeded (4 qual + 3 semi + 2 final).
    assert len(mu_map) == 9


def test_mu_before_lookup_empty_when_no_history(source_session: pg.Session) -> None:
    """mu_before_lookup returns an empty dict when no rating_history rows exist."""
    mu_map = mu_before_lookup(source_session)
    assert mu_map == {}


# ---------------------------------------------------------------------------
# sigma_before_lookup.
# ---------------------------------------------------------------------------


def test_sigma_before_lookup_returns_expected_values(source_session: pg.Session) -> None:
    """sigma_before_lookup maps each (athlete_id, round_id) to the correct sigma_before."""
    _seed(source_session)
    sigma_map = sigma_before_lookup(source_session)

    # Spot-check seeded sigma_before values.
    assert sigma_map[(1, 1)] == pytest.approx(100.0)  # athlete 1, round 1 (qual)
    assert sigma_map[(2, 1)] == pytest.approx(110.0)  # athlete 2, round 1
    assert sigma_map[(3, 1)] == pytest.approx(120.0)  # athlete 3, round 1
    assert sigma_map[(4, 1)] == pytest.approx(130.0)  # athlete 4, round 1
    assert sigma_map[(1, 3)] == pytest.approx(96.0)  # athlete 1, round 3 (final)
    assert sigma_map[(2, 3)] == pytest.approx(107.0)  # athlete 2, round 3


def test_sigma_before_lookup_covers_all_seeded_rows(source_session: pg.Session) -> None:
    """sigma_before_lookup returns one entry per seeded rating_history row."""
    _seed(source_session)
    sigma_map = sigma_before_lookup(source_session)
    assert len(sigma_map) == 9


def test_sigma_before_lookup_empty_when_no_history(source_session: pg.Session) -> None:
    """sigma_before_lookup returns an empty dict when no rating_history rows exist."""
    sigma_map = sigma_before_lookup(source_session)
    assert sigma_map == {}


def test_sigma_differs_from_mu(source_session: pg.Session) -> None:
    """sigma_before and mu_before return distinct lookups (different column values)."""
    _seed(source_session)
    mu_map = mu_before_lookup(source_session)
    sigma_map = sigma_before_lookup(source_session)
    # Keys should be identical sets.
    assert set(mu_map.keys()) == set(sigma_map.keys())
    # Values must differ (sigma_before != mu_before in all seeded rows).
    for key in mu_map:
        assert mu_map[key] != sigma_map[key], f"Expected mu != sigma for key {key}"
