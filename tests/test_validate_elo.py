"""Tests for the P3d expected_rank / elo_residual precompute (sync.validate_elo).

These tests use:
* the shared in-memory SQLite ``source_session`` fixture (defined in
  ``tests/conftest.py``), seeded here with results + rating_history (mu_before)
  across a qualification + semi + final round;
* the shared ``FakeGraphClient`` (also in ``conftest.py``) — its ``run_read``
  returns canned RestednessState rows (keyed on the exact ``REST_QUERY``) and
  every ``merge_node`` Performance prop-update is captured. NO live Neo4j is hit.

They assert representative-round selection (final preferred, else deepest),
expected_rank / elo_residual computation + correct Performance ids, and the
Pearson correlation (including the graceful n=0 path with no RestednessState).
"""

from __future__ import annotations

from datetime import date

import pytest

from climber_network import vocab
from climber_network.elo.expected import expected_finish_ranks
from climber_network.source import pg
from sync.validate_elo import (
    REST_QUERY,
    build_correlation_report,
    pearson,
    validate_elo,
)
from tests.conftest import FakeGraphClient

# ---------------------------------------------------------------------------
# Seeding helpers — a small competition with point-in-time mu_before.
# ---------------------------------------------------------------------------

# Event 1 (Lead) has three rounds for two athletes (1, 2):
#   round 1 qualification, round 2 semi, round 3 final.
# Athlete 1 reaches the final (rep = final); athlete 3 only reaches semi
# (rep = semi); athlete 4 only qualification (rep = qualification, deepest).
# mu_before values are point-in-time per round.


def _seed(session: pg.Session) -> None:
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
            # Qualification (round 1): all four.
            pg.Result(id=1, round_id=1, athlete_id=1, rank=1),
            pg.Result(id=2, round_id=1, athlete_id=2, rank=2),
            pg.Result(id=3, round_id=1, athlete_id=3, rank=3),
            pg.Result(id=4, round_id=1, athlete_id=4, rank=4),
            # Semi (round 2): athletes 1, 2, 3.
            pg.Result(id=5, round_id=2, athlete_id=1, rank=1),
            pg.Result(id=6, round_id=2, athlete_id=2, rank=2),
            pg.Result(id=7, round_id=2, athlete_id=3, rank=3),
            # Final (round 3): athletes 1, 2 — but athlete 2 finishes ahead (upset).
            pg.Result(id=8, round_id=3, athlete_id=1, rank=2),
            pg.Result(id=9, round_id=3, athlete_id=2, rank=1),
        ]
    )
    # rating_history.mu_before — point-in-time μ as of each round.
    # Athlete 1 is the strong favourite (high mu) everywhere.
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
                sigma_before=100.0,
                sigma_after=99.0,
            ),
            pg.RatingHistory(
                id=3,
                athlete_id=3,
                event_id=1,
                round_id=1,
                mu_before=1450.0,
                mu_after=1445.0,
                sigma_before=100.0,
                sigma_after=99.0,
            ),
            pg.RatingHistory(
                id=4,
                athlete_id=4,
                event_id=1,
                round_id=1,
                mu_before=1400.0,
                mu_after=1395.0,
                sigma_before=100.0,
                sigma_after=99.0,
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
                sigma_before=99.0,
                sigma_after=98.0,
            ),
            pg.RatingHistory(
                id=7,
                athlete_id=3,
                event_id=1,
                round_id=2,
                mu_before=1445.0,
                mu_after=1440.0,
                sigma_before=99.0,
                sigma_after=98.0,
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
                sigma_before=98.0,
                sigma_after=96.0,
            ),
        ]
    )
    session.commit()


def _perf_id(round_id: int, athlete_id: int) -> str:
    return vocab.perf(vocab.rnd(round_id), vocab.ath(athlete_id))


# ---------------------------------------------------------------------------
# Representative-round selection.
# ---------------------------------------------------------------------------


def test_representative_round_prefers_final_then_deepest(source_session: pg.Session) -> None:
    _seed(source_session)
    client = FakeGraphClient()
    report = validate_elo(client, source_session)

    by_athlete = {rep.athlete_id: rep for rep in report.reps}

    # Athletes 1 & 2 reached the final → rep is the final (round 3).
    assert by_athlete[1].round_id == 3
    assert by_athlete[1].round_type == "final"
    assert by_athlete[2].round_id == 3
    # Athlete 3's deepest round is the semi (round 2).
    assert by_athlete[3].round_id == 2
    assert by_athlete[3].round_type == "semi"
    # Athlete 4 only made qualification (round 1).
    assert by_athlete[4].round_id == 1
    assert by_athlete[4].round_type == "qualification"


# ---------------------------------------------------------------------------
# expected_rank / elo_residual computation + Performance writes.
# ---------------------------------------------------------------------------


def test_expected_rank_matches_pure_formula(source_session: pg.Session) -> None:
    _seed(source_session)
    client = FakeGraphClient()
    report = validate_elo(client, source_session)

    # The final's roster is athletes 1 & 2 with their round-3 mu_before.
    final_roster = [("1", 1720.0), ("2", 1500.0)]
    expected = expected_finish_ranks(final_roster)

    by_athlete = {rep.athlete_id: rep for rep in report.reps}
    assert by_athlete[1].expected_rank == pytest.approx(expected["1"])
    assert by_athlete[2].expected_rank == pytest.approx(expected["2"])

    # Athlete 1 (strong favourite) was expected near rank 1 but finished 2nd:
    # actual - expected > 0 (worse than expected). Athlete 2 over-performed (<0).
    assert by_athlete[1].elo_residual == pytest.approx(2.0 - expected["1"])
    assert by_athlete[1].elo_residual > 0
    assert by_athlete[2].elo_residual < 0


def test_derived_props_written_to_correct_performance_ids(source_session: pg.Session) -> None:
    _seed(source_session)
    client = FakeGraphClient()
    report = validate_elo(client, source_session)

    # Performance nodes were stamped for each representative round.
    assert report.performances_written == report.rep_rounds == 4

    # Athlete 1's rep is the final (round 3): props live on perf:rnd:3:ath:1.
    pid = _perf_id(3, 1)
    assert client.node_labels[pid] == "Performance"
    assert "expected_rank" in client.nodes[pid]
    assert "elo_residual" in client.nodes[pid]
    # Final favourite (athlete 1) finished 2nd → positive (worse-than-expected).
    assert client.nodes[pid]["elo_residual"] > 0

    # Athlete 3's rep is the semi (round 2), athlete 4's is qualification (round 1).
    assert _perf_id(2, 3) in client.nodes
    assert _perf_id(1, 4) in client.nodes
    # Athlete 3 did NOT get a final/semi-only Performance written for round 3.
    assert _perf_id(3, 3) not in client.nodes


def test_idempotent_rerun(source_session: pg.Session) -> None:
    _seed(source_session)
    client = FakeGraphClient()
    validate_elo(client, source_session)
    first = dict(client.nodes)
    validate_elo(client, source_session)
    # MERGE semantics: same keyed nodes, identical props (no drift).
    assert client.nodes == first


# ---------------------------------------------------------------------------
# Correlation — hand-set + graceful n=0.
# ---------------------------------------------------------------------------


def test_pearson_hand_set() -> None:
    # Perfect positive / negative / undefined cases.
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)
    assert pearson([1.0], [1.0]) is None  # < 2 pairs
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # zero variance


def test_correlation_negative_signal() -> None:
    # Hand-set reps + RestednessState rows where lower rested → larger residual
    # (worse than expected): a clean negative Pearson correlation.
    from sync.validate_elo import RepRound

    reps = [
        RepRound(
            athlete_id=aid,
            event_id=1,
            round_id=3,
            round_type="final",
            discipline="L",
            actual_rank=1,
            expected_rank=1.0,
            elo_residual=res,
        )
        for aid, res in [(1, -2.0), (2, -1.0), (3, 1.0), (4, 2.0)]
    ]
    rest_rows = [
        {
            "athlete_id": 1,
            "event_id": 1,
            "rested_index": 0.9,
            "discipline": "L",
            "travel_direction": "east",
        },
        {
            "athlete_id": 2,
            "event_id": 1,
            "rested_index": 0.7,
            "discipline": "L",
            "travel_direction": "east",
        },
        {
            "athlete_id": 3,
            "event_id": 1,
            "rested_index": 0.3,
            "discipline": "L",
            "travel_direction": "west",
        },
        {
            "athlete_id": 4,
            "event_id": 1,
            "rested_index": 0.1,
            "discipline": "L",
            "travel_direction": "west",
        },
    ]
    client = FakeGraphClient(read_results={REST_QUERY: rest_rows})

    block = build_correlation_report(client, reps)
    assert block["overall"]["n"] == 4
    assert block["overall"]["pearson_r"] is not None
    # Lower rested ↔ higher (worse) residual → negative correlation (success signal).
    assert block["overall"]["pearson_r"] < 0
    # Breakdown buckets populated.
    assert block["by_discipline"]["L"]["n"] == 4
    assert set(block["by_travel_direction"]) == {"east", "west"}
    assert block["by_travel_direction"]["east"]["n"] == 2


def test_correlation_graceful_when_no_restedness(source_session: pg.Session) -> None:
    _seed(source_session)
    # No RestednessState rows seeded → run_read returns [] for REST_QUERY.
    client = FakeGraphClient()
    report = validate_elo(client, source_session)

    corr = report.correlation
    assert corr["overall"]["n"] == 0
    assert corr["overall"]["pearson_r"] is None
    assert corr["by_discipline"] == {}
    assert corr["by_travel_direction"] == {}
