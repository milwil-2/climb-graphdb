"""Tests for the L3b Monte-Carlo placement build (sync.montecarlo).

These mirror the validate_elo test setup: the shared in-memory SQLite
``source_session`` fixture seeded with a qualification + semi + final round, and
the shared ``FakeGraphClient`` (canned ``run_read`` keyed on ``REST_QUERY``, every
``merge_nodes`` Performance prop-update captured). NO live Neo4j is hit.

They assert the additive MC props are stamped on the right Performance ids, that
``expected_rank_mc`` converges to the closed-form ``expected_rank``, determinism
under a fixed seed, idempotent re-run, and the rested_index ↔ result_percentile
correlation (including the graceful n=0 path).
"""

from __future__ import annotations

from datetime import date

import pytest

from climber_network import vocab
from climber_network.config import MonteCarloParams
from climber_network.elo.expected import expected_finish_ranks
from climber_network.source import pg
from sync.montecarlo import REST_QUERY, build_correlation_report, monte_carlo
from tests.conftest import FakeGraphClient

# Fast, deterministic params (fewer sims than the production default).
# Gaussian is the production default (mirrors climbing-elo's projection model);
# the plackett_luce params are used only for the closed-form convergence check.
_PARAMS_GAUSS = MonteCarloParams(n_sims=8000, seed=2024, model="gaussian")
_PARAMS_PL = MonteCarloParams(n_sims=8000, seed=2024, model="plackett_luce", sample_sigma=False)

_MC_PROPS = {
    "expected_rank_mc",
    "elo_residual_mc",
    "result_percentile",
    "surprisal",
    "p_win",
    "p_podium",
    "rank_std",
    "pmf_entropy",
    "mc_model_version",
}


def _seed(session: pg.Session) -> None:
    """Event 1 (Lead): qual(1)/semi(2)/final(3); athlete 1 is the strong favourite."""
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
            pg.Result(id=1, round_id=1, athlete_id=1, rank=1),
            pg.Result(id=2, round_id=1, athlete_id=2, rank=2),
            pg.Result(id=3, round_id=1, athlete_id=3, rank=3),
            pg.Result(id=4, round_id=1, athlete_id=4, rank=4),
            pg.Result(id=5, round_id=2, athlete_id=1, rank=1),
            pg.Result(id=6, round_id=2, athlete_id=2, rank=2),
            pg.Result(id=7, round_id=2, athlete_id=3, rank=3),
            # Final: athlete 2 upsets the favourite (rank 1), athlete 1 second.
            pg.Result(id=8, round_id=3, athlete_id=1, rank=2),
            pg.Result(id=9, round_id=3, athlete_id=2, rank=1),
        ]
    )
    # Point-in-time mu_before / sigma_before per (athlete, round).
    mus = {1: 1700.0, 2: 1500.0, 3: 1450.0, 4: 1400.0}
    rid = 0
    rows: list[pg.RatingHistory] = []
    for round_id, athletes in {1: (1, 2, 3, 4), 2: (1, 2, 3), 3: (1, 2)}.items():
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


def test_mc_props_stamped_on_representative_performances(source_session: pg.Session) -> None:
    _seed(source_session)
    client = FakeGraphClient()
    report = monte_carlo(client, source_session, params=_PARAMS_GAUSS)

    # One rep per (athlete, event): athlete 1 & 2 → final, 3 → semi, 4 → qual.
    assert report.rep_rounds == 4
    assert report.performances_written == 4
    expected_ids = {
        _perf_id(3, 1),
        _perf_id(3, 2),
        _perf_id(2, 3),
        _perf_id(1, 4),
    }
    assert expected_ids <= set(client.nodes)
    for pid in expected_ids:
        props = client.nodes[pid]
        assert _MC_PROPS <= set(props)
        assert client.node_labels[pid] == "Performance"
        assert props["mc_model_version"] == "mc-v1"
        assert 0.0 <= props["result_percentile"] <= 1.0
        assert props["surprisal"] >= 0.0


def test_expected_rank_mc_converges_to_closed_form(source_session: pg.Session) -> None:
    _seed(source_session)
    client = FakeGraphClient()
    # The plackett_luce model shares expected.py's logistic link, so its MC mean
    # converges to the closed form (the default gaussian model is a different family).
    monte_carlo(client, source_session, params=_PARAMS_PL)

    # Final round roster {1:1700, 2:1500} → compare MC mean to the exact closed form.
    closed = expected_finish_ranks([("1", 1700.0), ("2", 1500.0)])
    assert client.nodes[_perf_id(3, 1)]["expected_rank_mc"] == pytest.approx(closed["1"], abs=0.1)
    assert client.nodes[_perf_id(3, 2)]["expected_rank_mc"] == pytest.approx(closed["2"], abs=0.1)


def test_deterministic_under_fixed_seed(source_session: pg.Session) -> None:
    _seed(source_session)
    a = FakeGraphClient()
    b = FakeGraphClient()
    monte_carlo(a, source_session, params=_PARAMS_GAUSS)
    monte_carlo(b, source_session, params=_PARAMS_GAUSS)
    assert a.nodes == b.nodes


def test_idempotent_rerun(source_session: pg.Session) -> None:
    _seed(source_session)
    client = FakeGraphClient()
    monte_carlo(client, source_session, params=_PARAMS_GAUSS)
    first = {k: dict(v) for k, v in client.nodes.items()}
    monte_carlo(client, source_session, params=_PARAMS_GAUSS)
    assert client.nodes == first


def test_correlation_report_joins_rested_index(source_session: pg.Session) -> None:
    _seed(source_session)
    # Canned RestednessState rows keyed on the exact REST_QUERY.
    rest_rows = [
        {
            "athlete_id": 1,
            "event_id": 1,
            "rested_index": 0.9,
            "discipline": "L",
            "travel_direction": "E",
        },
        {
            "athlete_id": 2,
            "event_id": 1,
            "rested_index": 0.4,
            "discipline": "L",
            "travel_direction": "W",
        },
        {
            "athlete_id": 3,
            "event_id": 1,
            "rested_index": 0.7,
            "discipline": "L",
            "travel_direction": "none",
        },
    ]
    client = FakeGraphClient(read_results={REST_QUERY: rest_rows})
    report = monte_carlo(client, source_session, params=_PARAMS_GAUSS)

    overall = report.correlation["overall"]
    assert overall["n"] == 3
    # pearson_r is defined for n>=2 with variance; a float (sign not asserted on tiny n).
    assert overall["pearson_r"] is None or isinstance(overall["pearson_r"], float)
    assert "L" in report.correlation["by_discipline"]


def test_correlation_graceful_when_no_restedness(source_session: pg.Session) -> None:
    _seed(source_session)
    client = FakeGraphClient()  # no REST_QUERY rows seeded
    report = monte_carlo(client, source_session, params=_PARAMS_GAUSS)
    assert report.correlation["overall"] == {"pearson_r": None, "n": 0}


def test_build_correlation_report_standalone_empty() -> None:
    client = FakeGraphClient()
    out = build_correlation_report(client, [])
    assert out["overall"] == {"pearson_r": None, "n": 0}
