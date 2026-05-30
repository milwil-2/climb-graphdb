"""Tests for the pure season-aggregate module (climber_network.elo.season).

Stdlib-only: no graph, no source DB. Exercises grouping by
(athlete_id, season, discipline), None-skipping means, the upset threshold,
``over_under`` summation, ``mean_over_under`` per-event normalization,
deterministic sort order, and the season-driver report shape + a known-sign
correlation on synthetic data.
"""

from __future__ import annotations

from climber_network.elo.season import (
    PerformanceRecord,
    SeasonAggregate,
    aggregate_seasons,
    season_drivers_report,
)


def _rec(
    athlete_id: str = "ath:1", season: int = 2024, discipline: str = "L", **kw: object
) -> PerformanceRecord:
    return PerformanceRecord(athlete_id=athlete_id, season=season, discipline=discipline, **kw)  # type: ignore[arg-type]


def test_groups_by_athlete_season_discipline() -> None:
    records = [
        _rec("ath:1", 2024, "L", elo_residual=1.0),
        _rec("ath:1", 2024, "L", elo_residual=3.0),
        _rec("ath:1", 2025, "L", elo_residual=2.0),
        _rec("ath:1", 2024, "B", elo_residual=5.0),
        _rec("ath:2", 2024, "L", elo_residual=7.0),
    ]
    aggs = aggregate_seasons(records)
    keys = [(a.athlete_id, a.season, a.discipline, a.n_events) for a in aggs]
    assert keys == [
        ("ath:1", 2024, "B", 1),
        ("ath:1", 2024, "L", 2),
        ("ath:1", 2025, "L", 1),
        ("ath:2", 2024, "L", 1),
    ]


def test_sorted_output_is_deterministic() -> None:
    a = aggregate_seasons(
        [
            _rec("ath:2", 2024, "L", elo_residual=1.0),
            _rec("ath:1", 2025, "B", elo_residual=1.0),
            _rec("ath:1", 2024, "L", elo_residual=1.0),
        ]
    )
    b = aggregate_seasons(
        [
            _rec("ath:1", 2024, "L", elo_residual=1.0),
            _rec("ath:1", 2025, "B", elo_residual=1.0),
            _rec("ath:2", 2024, "L", elo_residual=1.0),
        ]
    )
    assert [(x.athlete_id, x.season, x.discipline) for x in a] == [
        (x.athlete_id, x.season, x.discipline) for x in b
    ]


def test_means_skip_none() -> None:
    records = [
        _rec(elo_residual=2.0, result_percentile=None, p_win=0.5, rank_std=1.0),
        _rec(elo_residual=None, result_percentile=0.4, p_win=None, rank_std=3.0),
        _rec(elo_residual=4.0, result_percentile=0.6, p_win=0.7, rank_std=None),
    ]
    (agg,) = aggregate_seasons(records)
    # elo_residual over [2,4]; result_percentile over [0.4,0.6]; p_win over [0.5,0.7].
    assert agg.mean_elo_residual == 3.0
    assert agg.mean_result_percentile == 0.5
    assert agg.season_skill == 0.6
    assert agg.season_consistency == 2.0  # rank_std over [1,3]


def test_mean_none_when_no_data() -> None:
    (agg,) = aggregate_seasons([_rec()])
    assert agg.mean_elo_residual is None
    assert agg.mean_result_percentile is None
    assert agg.mean_surprisal is None
    assert agg.season_skill is None
    assert agg.season_consistency is None
    assert agg.mean_rested_index is None
    assert agg.over_under == 0.0
    # n_events == 1 (one record) but no residuals -> mean_over_under == 0.0.
    assert agg.mean_over_under == 0.0


def test_n_upsets_honors_threshold() -> None:
    records = [
        _rec(surprisal=1.0),
        _rec(surprisal=2.0),  # not strictly greater than default 2.0
        _rec(surprisal=2.5),
        _rec(surprisal=None),
        _rec(surprisal=9.0),
    ]
    (agg,) = aggregate_seasons(records)
    assert agg.n_upsets == 2  # 2.5 and 9.0

    (agg_low,) = aggregate_seasons(records, upset_threshold=1.5)
    assert agg_low.n_upsets == 3  # 2.0, 2.5, 9.0


def test_over_under_sums_available_residuals() -> None:
    records = [
        _rec(elo_residual=1.5),
        _rec(elo_residual=-0.5),
        _rec(elo_residual=None),
        _rec(elo_residual=2.0),
    ]
    (agg,) = aggregate_seasons(records)
    assert agg.over_under == 3.0
    assert agg.n_events == 4
    # mean_over_under is the cumulative sum normalized by event count (3.0 / 4).
    assert agg.mean_over_under == 0.75


def test_mean_over_under_is_sum_over_n_events() -> None:
    # A season with known residuals over N events: over_under == sum,
    # mean_over_under == sum / N, both fields present and distinct.
    records = [_rec(elo_residual=r) for r in (2.0, 4.0, 6.0)]
    (agg,) = aggregate_seasons(records)
    assert agg.n_events == 3
    assert agg.over_under == 12.0
    assert agg.mean_over_under == 4.0
    assert agg.over_under != agg.mean_over_under


def test_empty_input() -> None:
    assert aggregate_seasons([]) == []
    report = season_drivers_report([])
    assert report["overall"] == {"pearson_r": None, "n": 0}
    assert report["by_discipline"] == {}
    assert "negative correlation" in report["success_signal"]


def test_drivers_report_shape_and_known_sign() -> None:
    # Synthetic: as mean_rested_index rises, mean_over_under falls -> negative r.
    # The cumulative over_under is set to a constant so a *positive* r would only
    # appear if the report (incorrectly) correlated the sum instead.
    aggs: list[SeasonAggregate] = []
    for i, rested in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        aggs.append(
            SeasonAggregate(
                athlete_id=f"ath:{i}",
                season=2024,
                discipline="L",
                n_events=3,
                mean_elo_residual=None,
                mean_result_percentile=None,
                mean_surprisal=None,
                season_skill=None,
                season_consistency=None,
                mean_rested_index=rested,
                n_upsets=0,
                over_under=42.0,  # constant: only mean_over_under carries signal
                mean_over_under=10.0 - 10.0 * rested,  # perfectly anti-correlated
            )
        )
    report = season_drivers_report(aggs)
    overall = report["overall"]
    assert overall["n"] == 5
    assert overall["pearson_r"] is not None
    assert overall["pearson_r"] < 0
    assert "L" in report["by_discipline"]
    assert report["by_discipline"]["L"]["n"] == 5


def test_drivers_report_skips_aggregates_without_rested_index() -> None:
    aggs = [
        SeasonAggregate(
            athlete_id="ath:1",
            season=2024,
            discipline="L",
            n_events=1,
            mean_elo_residual=None,
            mean_result_percentile=None,
            mean_surprisal=None,
            season_skill=None,
            season_consistency=None,
            mean_rested_index=None,  # excluded
            n_upsets=0,
            over_under=5.0,
            mean_over_under=5.0,
        ),
        SeasonAggregate(
            athlete_id="ath:2",
            season=2024,
            discipline="L",
            n_events=1,
            mean_elo_residual=None,
            mean_result_percentile=None,
            mean_surprisal=None,
            season_skill=None,
            season_consistency=None,
            mean_rested_index=0.5,
            n_upsets=0,
            over_under=3.0,
            mean_over_under=3.0,
        ),
    ]
    report = season_drivers_report(aggs)
    # Only one aggregate has rested_index -> n=1 -> pearson undefined (None).
    assert report["overall"] == {"pearson_r": None, "n": 1}
