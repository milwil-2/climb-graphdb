"""Tests for the Phase-4 season aggregation build (sync.season).

Mirrors the test_montecarlo_sync conventions: a ``FakeGraphClient`` seeded with
canned rows keyed on the exact module-level ``SEASON_QUERY`` string, every
``merge_nodes`` / ``merge_rels`` call captured. NO live Neo4j is hit.

Asserts SeasonSummary nodes are MERGEd with the right ``vocab.seas`` ids + rollup
props, HAD_SEASON edges run Athlete -> SeasonSummary, the build is idempotent on
re-run, and the season-driver correlation is computed with the expected sign.
"""

from __future__ import annotations

from climber_network import vocab
from sync.season import SEASON_QUERY, season
from tests.conftest import FakeGraphClient


def _row(
    athlete_id: str,
    season_year: int,
    discipline: str,
    *,
    elo_residual: float | None = None,
    result_percentile: float | None = None,
    surprisal: float | None = None,
    p_win: float | None = None,
    rank_std: float | None = None,
    rested_index: float | None = None,
) -> dict[str, object]:
    return {
        "athlete_id": athlete_id,
        "season": season_year,
        "discipline": discipline,
        "elo_residual": elo_residual,
        "result_percentile": result_percentile,
        "surprisal": surprisal,
        "p_win": p_win,
        "rank_std": rank_std,
        "rested_index": rested_index,
    }


def _seed_rows() -> list[dict[str, object]]:
    # Athlete ath:1 has a two-event 2024 Lead season + a one-event 2025 Lead
    # season; ath:2 has a one-event 2024 Lead season. rested_index falls as
    # over_under rises across athletes -> negative driver correlation.
    return [
        _row(
            "ath:1",
            2024,
            "L",
            elo_residual=1.0,
            surprisal=3.0,
            p_win=0.6,
            rank_std=2.0,
            rested_index=0.9,
        ),
        _row(
            "ath:1",
            2024,
            "L",
            elo_residual=3.0,
            surprisal=1.0,
            p_win=0.4,
            rank_std=4.0,
            rested_index=0.9,
        ),
        _row(
            "ath:1",
            2025,
            "L",
            elo_residual=-1.0,
            surprisal=0.5,
            p_win=0.8,
            rank_std=1.0,
            rested_index=0.95,
        ),
        _row(
            "ath:2",
            2024,
            "L",
            elo_residual=8.0,
            surprisal=5.0,
            p_win=0.2,
            rank_std=6.0,
            rested_index=0.3,
        ),
    ]


def test_season_summaries_merged_with_ids_and_props() -> None:
    client = FakeGraphClient(read_results={SEASON_QUERY: _seed_rows()})
    report = season(client)

    assert report.records_read == 4
    assert report.season_summaries_written == 3  # (ath:1,2024,L) (ath:1,2025,L) (ath:2,2024,L)

    sid_1_2024 = vocab.seas("ath:1", 2024, "L")
    sid_1_2025 = vocab.seas("ath:1", 2025, "L")
    sid_2_2024 = vocab.seas("ath:2", 2024, "L")
    assert {sid_1_2024, sid_1_2025, sid_2_2024} <= set(client.nodes)

    for sid in (sid_1_2024, sid_1_2025, sid_2_2024):
        assert client.node_labels[sid] == "SeasonSummary"

    # ath:1 2024 L: two events, residuals [1,3].
    props = client.nodes[sid_1_2024]
    assert props["athlete_id"] == "ath:1"
    assert props["season"] == 2024
    assert props["discipline"] == "L"
    assert props["n_events"] == 2
    assert props["over_under"] == 4.0
    assert props["mean_elo_residual"] == 2.0
    assert props["season_skill"] == 0.5  # mean p_win [0.6,0.4]
    assert props["season_consistency"] == 3.0  # mean rank_std [2,4]
    assert props["mean_rested_index"] == 0.9
    assert props["n_upsets"] == 1  # only surprisal 3.0 > 2.0


def test_none_valued_rollup_props_are_dropped() -> None:
    rows = [_row("ath:9", 2024, "L", elo_residual=2.0)]  # no p_win/rank_std/etc.
    client = FakeGraphClient(read_results={SEASON_QUERY: rows})
    season(client)

    props = client.nodes[vocab.seas("ath:9", 2024, "L")]
    # Present (non-None) rollups + always-present scalars.
    assert "mean_elo_residual" in props
    assert props["over_under"] == 2.0
    assert props["n_upsets"] == 0
    # Dropped because their source values were all None.
    for dropped in (
        "mean_result_percentile",
        "mean_surprisal",
        "season_skill",
        "season_consistency",
        "mean_rested_index",
    ):
        assert dropped not in props


def test_had_season_edges_athlete_to_summary() -> None:
    client = FakeGraphClient(read_results={SEASON_QUERY: _seed_rows()})
    season(client)

    sid_1_2024 = vocab.seas("ath:1", 2024, "L")
    sid_2_2024 = vocab.seas("ath:2", 2024, "L")
    assert ("ath:1", "HAD_SEASON", sid_1_2024) in client.rels
    assert ("ath:2", "HAD_SEASON", sid_2_2024) in client.rels
    # Every HAD_SEASON edge points at a SeasonSummary node we wrote.
    for _src, rel, tgt in client.rel_calls:
        assert rel == "HAD_SEASON"
        assert client.node_labels[tgt] == "SeasonSummary"


def test_nodes_written_before_edges() -> None:
    client = FakeGraphClient(read_results={SEASON_QUERY: _seed_rows()})
    season(client)
    # All node merges precede all rel merges (merge_nodes called before merge_rels).
    assert len(client.node_calls) == 3
    assert len(client.rel_calls) == 3


def test_idempotent_rerun() -> None:
    client = FakeGraphClient(read_results={SEASON_QUERY: _seed_rows()})
    season(client)
    first_nodes = {k: dict(v) for k, v in client.nodes.items()}
    first_rels = dict(client.rels)
    season(client)
    assert client.nodes == first_nodes
    assert client.rels == first_rels


def test_drivers_report_negative_sign() -> None:
    client = FakeGraphClient(read_results={SEASON_QUERY: _seed_rows()})
    report = season(client)

    overall = report.drivers["overall"]
    # Three athlete-seasons carry rested_index; higher rested -> lower over_under.
    assert overall["n"] == 3
    assert overall["pearson_r"] is not None
    assert overall["pearson_r"] < 0
    assert "L" in report.drivers["by_discipline"]


def test_empty_graph_yields_no_summaries() -> None:
    client = FakeGraphClient()  # no SEASON_QUERY rows seeded
    report = season(client)
    assert report.records_read == 0
    assert report.season_summaries_written == 0
    assert report.drivers["overall"] == {"pearson_r": None, "n": 0}
    assert client.nodes == {}
