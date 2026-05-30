"""Tests for the L3c advancement-projection build (sync.advancement).

These mirror the montecarlo sync test setup: the shared in-memory SQLite
``source_session`` fixture seeded with a qualification(4) → semi(2) → final
round, Results, and RatingHistory rows (mu_before/sigma_before per
athlete-round). NO live Neo4j is hit.

Assertions cover:
* All five advancement props are stamped on representative Performances.
* Props satisfy ``1 >= p_make_final >= p_podium >= p_win >= 0``.
* ``advancement_surprise >= 0``.
* The favourite (high mu) has a higher ``p_win_event`` than a weaker athlete.
* Deterministic output under fixed params.
* Idempotent re-run (same ``client.nodes`` state).
* Athletes missing ``mu_before`` in the entry round are counted as skipped.
"""

from __future__ import annotations

import math
from datetime import date

from climber_network import vocab
from climber_network.config import MonteCarloParams
from climber_network.source import pg
from sync.advancement import advancement
from tests.conftest import FakeGraphClient

# Fast, deterministic params (fewer sims than the production default).
_PARAMS = MonteCarloParams(n_sims=8000, seed=2025, model="gaussian")

_ADV_PROPS = {
    "p_make_final",
    "p_podium_event",
    "p_win_event",
    "advancement_surprise",
    "mc_model_version",
}


def _seed(session: pg.Session) -> None:
    """Event 1 (Lead): qual(4) → semi(2) → final; athlete 1 is the strong favourite.

    Round layout:
      - round_id=1  qualification  athlete_count=4
      - round_id=2  semi           athlete_count=2
      - round_id=3  final          athlete_count=2   (last round, advance_count ignored)

    Results:
      - Athletes 1,2,3,4 all compete in qual.
      - Athletes 1,2 advance to semi (semi has athlete_count=2); 3 & 4 don't.
      - Athletes 1,2 compete in final.
      - Athletes 3 & 4 → rep in qual (round 1); athletes 1 & 2 → rep in final (round 3).

    RatingHistory (mu_before/sigma_before) supplied for all athletes in their rounds.
    """
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
            # qual: 4 slots; the advance_count for semi (=2) comes from round_id=2.
            pg.Round(id=1, event_id=1, round_type="qualification", gender="F", athlete_count=4),
            # semi: athlete_count=2 drives the qual→semi cut; advance_count for final from rnd 3.
            pg.Round(id=2, event_id=1, round_type="semi", gender="F", athlete_count=2),
            # final: athlete_count=2; advance_count=0 (last round, ignored by simulator).
            pg.Round(id=3, event_id=1, round_type="final", gender="F", athlete_count=2),
        ]
    )
    session.add_all(
        [
            # Qualification — all four athletes.
            pg.Result(id=1, round_id=1, athlete_id=1, rank=1),
            pg.Result(id=2, round_id=1, athlete_id=2, rank=2),
            pg.Result(id=3, round_id=1, athlete_id=3, rank=3),
            pg.Result(id=4, round_id=1, athlete_id=4, rank=4),
            # Semi — top-2 advance.
            pg.Result(id=5, round_id=2, athlete_id=1, rank=1),
            pg.Result(id=6, round_id=2, athlete_id=2, rank=2),
            # Final — athlete 2 upsets the favourite.
            pg.Result(id=7, round_id=3, athlete_id=2, rank=1),
            pg.Result(id=8, round_id=3, athlete_id=1, rank=2),
        ]
    )
    # Point-in-time mu_before / sigma_before per (athlete, round).
    # Athlete 1 is the strong favourite (mu=1700); athlete 4 is the weakest (mu=1350).
    mus = {1: 1700.0, 2: 1500.0, 3: 1400.0, 4: 1350.0}
    rid = 0
    rows: list[pg.RatingHistory] = []
    for round_id, athletes in {1: (1, 2, 3, 4), 2: (1, 2), 3: (1, 2)}.items():
        for aid in athletes:
            rid += 1
            rows.append(
                pg.RatingHistory(
                    id=rid,
                    athlete_id=aid,
                    event_id=1,
                    round_id=round_id,
                    mu_before=mus[aid],
                    mu_after=mus[aid],
                    sigma_before=100.0,
                    sigma_after=98.0,
                )
            )
    session.add_all(rows)
    session.commit()


def _perf_id(round_id: int, athlete_id: int) -> str:
    return vocab.perf(vocab.rnd(round_id), vocab.ath(athlete_id))


# ---------------------------------------------------------------------------
# Test: five props are stamped on representative Performances.
# ---------------------------------------------------------------------------


def test_advancement_props_stamped_on_representative_performances(
    source_session: pg.Session,
) -> None:
    """All five advancement props appear on every representative Performance."""
    _seed(source_session)
    client = FakeGraphClient()
    report = advancement(client, source_session, params=_PARAMS)

    # One rep per (athlete, event):
    #   athletes 1 & 2 → final (round 3, they competed in qual+semi+final)
    #   athlete 3 → qual (round 1, only competed in qual — never reached semi)
    #   athlete 4 → qual (round 1, only competed in qual)
    assert report.performances_written == 4
    expected_ids = {
        _perf_id(3, 1),
        _perf_id(3, 2),
        _perf_id(1, 3),
        _perf_id(1, 4),
    }
    assert expected_ids <= set(client.nodes)
    for pid in expected_ids:
        props = client.nodes[pid]
        assert _ADV_PROPS <= set(props), f"Missing props on {pid}: {set(props)}"
        assert client.node_labels[pid] == "Performance"
        assert props["mc_model_version"] == "mc-v1"


# ---------------------------------------------------------------------------
# Test: monotonicity contract and non-negativity of surprise.
# ---------------------------------------------------------------------------


def test_advancement_props_satisfy_monotonicity(source_session: pg.Session) -> None:
    """Props satisfy 1 >= p_make_final >= p_podium >= p_win >= 0 and surprise >= 0."""
    _seed(source_session)
    client = FakeGraphClient()
    advancement(client, source_session, params=_PARAMS)

    for pid, props in client.nodes.items():
        if "p_make_final" not in props:
            continue
        p_make_final = props["p_make_final"]
        p_podium = props["p_podium_event"]
        p_win = props["p_win_event"]
        surprise = props["advancement_surprise"]

        assert 0.0 <= p_win <= p_podium <= p_make_final <= 1.0, (
            f"{pid}: monotonicity violated — "
            f"p_make_final={p_make_final} p_podium={p_podium} p_win={p_win}"
        )
        assert surprise >= 0.0, f"{pid}: advancement_surprise={surprise} < 0"


# ---------------------------------------------------------------------------
# Test: favourite has higher p_win_event than weaker athlete.
# ---------------------------------------------------------------------------


def test_favourite_has_higher_p_win_than_weaker(source_session: pg.Session) -> None:
    """Athlete 1 (mu=1700) must have higher p_win_event than athlete 4 (mu=1350)."""
    _seed(source_session)
    client = FakeGraphClient()
    advancement(client, source_session, params=_PARAMS)

    # Athlete 1's rep is final (round 3); athlete 4's rep is qual (round 1).
    fav_props = client.nodes[_perf_id(3, 1)]
    weak_props = client.nodes[_perf_id(1, 4)]

    assert fav_props["p_win_event"] > weak_props["p_win_event"], (
        f"Expected favourite p_win={fav_props['p_win_event']} "
        f"> weak p_win={weak_props['p_win_event']}"
    )
    assert fav_props["p_make_final"] > weak_props["p_make_final"]


# ---------------------------------------------------------------------------
# Test: determinism under a fixed seed.
# ---------------------------------------------------------------------------


def test_deterministic_under_fixed_seed(source_session: pg.Session) -> None:
    """Two runs with the same params produce identical client.nodes."""
    _seed(source_session)
    a = FakeGraphClient()
    b = FakeGraphClient()
    advancement(a, source_session, params=_PARAMS)
    advancement(b, source_session, params=_PARAMS)
    assert a.nodes == b.nodes


# ---------------------------------------------------------------------------
# Test: idempotent re-run (MERGE semantics — second run leaves nodes unchanged).
# ---------------------------------------------------------------------------


def test_idempotent_rerun(source_session: pg.Session) -> None:
    """A second run with the same params yields the exact same node state."""
    _seed(source_session)
    client = FakeGraphClient()
    advancement(client, source_session, params=_PARAMS)
    first = {k: dict(v) for k, v in client.nodes.items()}
    advancement(client, source_session, params=_PARAMS)
    assert client.nodes == first


# ---------------------------------------------------------------------------
# Test: athletes without mu_before in the entry round are skipped and counted.
# ---------------------------------------------------------------------------


def _seed_zero_athlete_count(session: pg.Session) -> None:
    """Same layout as :func:`_seed` but the semi round's ``athlete_count`` is 0.

    The semi (round_id=2) carries a stale/zero ``athlete_count``, yet its roster
    holds 2 distinct non-DNS Results (athletes 1 & 2). The fix must derive the
    qual→semi ``advance_count`` from that roster (=2), not from the recorded 0,
    so the field is NOT collapsed to everyone-eliminated.
    """
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
            # Stale/zero athlete_count — the simulator must fall back to the roster.
            pg.Round(id=2, event_id=1, round_type="semi", gender="F", athlete_count=0),
            pg.Round(id=3, event_id=1, round_type="final", gender="F", athlete_count=2),
        ]
    )
    session.add_all(
        [
            pg.Result(id=1, round_id=1, athlete_id=1, rank=1),
            pg.Result(id=2, round_id=1, athlete_id=2, rank=2),
            pg.Result(id=3, round_id=1, athlete_id=3, rank=3),
            pg.Result(id=4, round_id=1, athlete_id=4, rank=4),
            # Semi roster: 2 distinct non-DNS athletes despite athlete_count=0.
            pg.Result(id=5, round_id=2, athlete_id=1, rank=1),
            pg.Result(id=6, round_id=2, athlete_id=2, rank=2),
            pg.Result(id=7, round_id=3, athlete_id=2, rank=1),
            pg.Result(id=8, round_id=3, athlete_id=1, rank=2),
        ]
    )
    mus = {1: 1700.0, 2: 1500.0, 3: 1400.0, 4: 1350.0}
    rid = 0
    rows: list[pg.RatingHistory] = []
    for round_id, athletes in {1: (1, 2, 3, 4), 2: (1, 2), 3: (1, 2)}.items():
        for aid in athletes:
            rid += 1
            rows.append(
                pg.RatingHistory(
                    id=rid,
                    athlete_id=aid,
                    event_id=1,
                    round_id=round_id,
                    mu_before=mus[aid],
                    mu_after=mus[aid],
                    sigma_before=100.0,
                    sigma_after=98.0,
                )
            )
    session.add_all(rows)
    session.commit()


def test_advance_count_derived_from_roster_when_athlete_count_zero(
    source_session: pg.Session,
) -> None:
    """A zero next-round ``athlete_count`` must not collapse the field to 0 advancers.

    With the semi round's recorded ``athlete_count`` set to 0 but 2 distinct
    non-DNS Results in its roster, the qual→semi cut advances 2 athletes. The
    favourite therefore has a strictly positive p_make_final and a finite
    advancement_surprise — the bug symptom (everyone eliminated → all zeros and
    inflated surprise) does not occur.
    """
    _seed_zero_athlete_count(source_session)
    client = FakeGraphClient()
    advancement(client, source_session, params=_PARAMS)

    fav_props = client.nodes[_perf_id(3, 1)]
    assert fav_props["p_make_final"] > 0.0
    assert fav_props["p_podium_event"] > 0.0
    assert fav_props["p_win_event"] > 0.0
    assert math.isfinite(fav_props["advancement_surprise"])


def test_zero_athlete_count_matches_populated_roster_count(source_session: pg.Session) -> None:
    """Deriving advance_count from the roster reproduces the populated-count result.

    The semi roster has exactly 2 distinct non-DNS athletes, so setting
    ``athlete_count`` to either 2 (populated) or 0 (stale, fall back to roster)
    must yield identical stamped props — confirming the populated-path behavior
    is unchanged and the fallback is equivalent.
    """
    a = FakeGraphClient()
    _seed(source_session)
    advancement(a, source_session, params=_PARAMS)

    b = FakeGraphClient()
    # Fresh session via a second fixture-style build is awkward; instead mutate
    # the existing rows to athlete_count=0 and re-run — the roster is unchanged.
    semi = source_session.get(pg.Round, 2)
    assert semi is not None
    semi.athlete_count = 0
    source_session.commit()
    advancement(b, source_session, params=_PARAMS)

    assert a.nodes == b.nodes


def test_athletes_missing_mu_are_skipped(source_session: pg.Session) -> None:
    """An athlete with no mu_before in the entry round is counted as skipped."""
    # Seed without rating history for athlete 4 in round 1 (entry round).
    session = source_session
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
            name="World Cup Test",
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
            pg.Round(id=2, event_id=1, round_type="semi", gender="F", athlete_count=2),
            pg.Round(id=3, event_id=1, round_type="final", gender="F", athlete_count=2),
        ]
    )
    session.add_all(
        [
            pg.Result(id=1, round_id=1, athlete_id=1, rank=1),
            pg.Result(id=2, round_id=1, athlete_id=2, rank=2),
            pg.Result(id=3, round_id=1, athlete_id=3, rank=3),
            pg.Result(id=4, round_id=1, athlete_id=4, rank=4),  # no RatingHistory for athlete 4
            pg.Result(id=5, round_id=2, athlete_id=1, rank=1),
            pg.Result(id=6, round_id=2, athlete_id=2, rank=2),
            pg.Result(id=7, round_id=3, athlete_id=1, rank=1),
            pg.Result(id=8, round_id=3, athlete_id=2, rank=2),
        ]
    )
    # Rating history omits athlete 4 entirely — they have no mu_before in entry round.
    mus = {1: 1700.0, 2: 1500.0, 3: 1400.0}
    rid = 0
    rows: list[pg.RatingHistory] = []
    for round_id, athletes in {1: (1, 2, 3), 2: (1, 2), 3: (1, 2)}.items():
        for aid in athletes:
            rid += 1
            rows.append(
                pg.RatingHistory(
                    id=rid,
                    athlete_id=aid,
                    event_id=1,
                    round_id=round_id,
                    mu_before=mus[aid],
                    mu_after=mus[aid],
                    sigma_before=100.0,
                    sigma_after=98.0,
                )
            )
    session.add_all(rows)
    session.commit()

    client = FakeGraphClient()
    report = advancement(client, session, params=_PARAMS)

    # Athlete 4 lacks mu_before in entry round → counted as skipped.
    assert report.skipped.get("no_mu_before_entry_round", 0) >= 1
    # The three athletes with mu_before produce reps and get stamped.
    assert report.performances_written == 3
    # Athlete 4's rep (qual round) should NOT be in the written nodes.
    assert _perf_id(1, 4) not in client.nodes
