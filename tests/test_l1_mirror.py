"""Tests for the P1 L1 competition mirror (sync.pg_to_neo4j).

These tests use:
* an in-memory SQLite database as the source fixture (a few athletes, events,
  rounds, results, ratings — including final, semi, and qualification rounds);
* a FAKE GraphClient that records every merge_node / merge_rel call, so NO live
  Neo4j connection is ever made.

They assert node/edge creation + ids, the FACED scope (final/semi only,
aggregated per ordered pair), idempotency (run twice → no duplicate logical
MERGEs), count-validation pass/fail, and that an out-of-vocab label/rel raises.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from climber_network import vocab
from climber_network.source import pg
from climber_network.vocab import assert_label, assert_rel
from sync.pg_to_neo4j import (
    CountValidationError,
    sync_graph,
    validate_counts,
)

# ---------------------------------------------------------------------------
# Fake GraphClient — records merges, validates vocab like the real client.
# ---------------------------------------------------------------------------


class FakeGraphClient:
    """Records merge calls; mirrors the real client's vocab-gating behaviour."""

    def __init__(self) -> None:
        # node_id -> latest props (MERGE semantics: keyed, last write wins).
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_labels: dict[str, str] = {}
        # (src, rel, tgt) -> latest props.
        self.rels: dict[tuple[str, str, str], dict[str, Any] | None] = {}
        # Raw call logs for duplicate / ordering assertions.
        self.node_calls: list[tuple[str, str]] = []
        self.rel_calls: list[tuple[str, str, str]] = []

    def merge_node(self, label: str, node_id: str, props: dict[str, Any]) -> None:
        assert_label(label)  # raises ValueError on out-of-vocab label
        self.nodes[node_id] = dict(props)
        self.node_labels[node_id] = label
        self.node_calls.append((label, node_id))

    def merge_rel(
        self,
        src_id: str,
        rel_type: str,
        tgt_id: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        assert_rel(rel_type)  # raises ValueError on out-of-vocab rel
        self.rels[(src_id, rel_type, tgt_id)] = dict(props) if props else None
        self.rel_calls.append((src_id, rel_type, tgt_id))


# ---------------------------------------------------------------------------
# SQLite source fixture.
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> pg.Engine:
    eng = pg.make_engine("sqlite:///:memory:")
    pg.Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def seeded_engine(engine: pg.Engine) -> pg.Engine:
    """Seed a small but representative competition.

    Event 1 (Lead) has three rounds:
      - round 1 qualification (4 athletes) — must NOT produce FACED
      - round 2 semi          (3 athletes) — FACED
      - round 3 final         (2 athletes) — FACED
    Event 2 (Boulder) has one final (round 4, 2 athletes) on a later date —
    contributes a second FACED round to a pair, exercising aggregation.
    One result is DNS (documented filter → no Performance).
    """
    with pg.Session(engine) as s:
        s.add_all(
            [
                pg.Athlete(id=1, name="Ada", gender="F", nationality="USA"),
                pg.Athlete(id=2, name="Bea", gender="F", nationality="GBR"),
                pg.Athlete(id=3, name="Cleo", gender="F", nationality="JPN"),
                pg.Athlete(id=4, name="Dot", gender="F", nationality="AUT"),
            ]
        )
        s.add_all(
            [
                pg.Event(
                    id=1,
                    name="World Cup Innsbruck",
                    tier="world_cup",
                    country="AUT",
                    season=2024,
                    start_date=date(2024, 6, 1),
                    discipline="L",
                ),
                pg.Event(
                    id=2,
                    name="World Cup Bern",
                    tier="world_cup",
                    country="CHE",
                    season=2024,
                    start_date=date(2024, 7, 1),
                    discipline="B",
                ),
            ]
        )
        s.add_all(
            [
                pg.Round(id=1, event_id=1, round_type="qualification", gender="F", athlete_count=4),
                pg.Round(id=2, event_id=1, round_type="semi", gender="F", athlete_count=3),
                pg.Round(id=3, event_id=1, round_type="final", gender="F", athlete_count=2),
                pg.Round(id=4, event_id=2, round_type="final", gender="F", athlete_count=2),
            ]
        )
        s.add_all(
            [
                # Qualification (round 1): 4 athletes — excluded from FACED.
                pg.Result(id=1, round_id=1, athlete_id=1, rank=1, score_normalized=1.0),
                pg.Result(id=2, round_id=1, athlete_id=2, rank=2, score_normalized=0.9),
                pg.Result(id=3, round_id=1, athlete_id=3, rank=3, score_normalized=0.8),
                pg.Result(id=4, round_id=1, athlete_id=4, rank=4, dns=True),  # DNS filter
                # Semi (round 2): athletes 1,2,3 → FACED.
                pg.Result(id=5, round_id=2, athlete_id=1, rank=1),
                pg.Result(id=6, round_id=2, athlete_id=2, rank=2),
                pg.Result(id=7, round_id=2, athlete_id=3, rank=3),
                # Final (round 3): athletes 1,2 → FACED.
                pg.Result(id=8, round_id=3, athlete_id=1, rank=1),
                pg.Result(id=9, round_id=3, athlete_id=2, rank=2),
                # Event 2 Final (round 4): athletes 1,2 again → aggregates with round 3.
                pg.Result(id=10, round_id=4, athlete_id=1, rank=2),
                pg.Result(id=11, round_id=4, athlete_id=2, rank=1),
            ]
        )
        s.add_all(
            [
                pg.Rating(id=1, athlete_id=1, discipline="L", mu=1600.0, sigma=200.0, n_events=5),
                pg.Rating(id=2, athlete_id=2, discipline="L", mu=1550.0, sigma=210.0, n_events=4),
            ]
        )
        s.commit()
    return engine


# ---------------------------------------------------------------------------
# Node / edge / id assertions.
# ---------------------------------------------------------------------------


def _run(engine: pg.Engine, client: FakeGraphClient):
    with pg.read_session(engine) as session:
        return sync_graph(client, session)


def test_nodes_created_with_correct_ids(seeded_engine: pg.Engine) -> None:
    client = FakeGraphClient()
    _run(seeded_engine, client)

    # Athletes
    assert client.node_labels[vocab.ath(1)] == "Athlete"
    assert client.nodes[vocab.ath(1)]["name"] == "Ada"
    assert all(vocab.ath(i) in client.nodes for i in (1, 2, 3, 4))

    # Events + disciplines
    assert client.node_labels[vocab.evt(1)] == "Event"
    assert client.nodes[vocab.evt(1)]["discipline"] == "L"
    assert client.node_labels[vocab.disc("L")] == "Discipline"
    assert client.node_labels[vocab.disc("B")] == "Discipline"

    # Rounds
    assert client.node_labels[vocab.rnd(2)] == "Round"
    assert client.nodes[vocab.rnd(2)]["round_type"] == "semi"

    # Performance id keyed on round + athlete; DNS row produced none.
    perf_id = vocab.perf(vocab.rnd(3), vocab.ath(1))
    assert client.node_labels[perf_id] == "Performance"
    dns_perf = vocab.perf(vocab.rnd(1), vocab.ath(4))
    assert dns_perf not in client.nodes

    # Ratings
    rat_id = vocab.rat(vocab.ath(1), "L")
    assert client.node_labels[rat_id] == "Rating"
    assert client.nodes[rat_id]["mu"] == 1600.0


def test_structural_edges(seeded_engine: pg.Engine) -> None:
    client = FakeGraphClient()
    _run(seeded_engine, client)

    perf_id = vocab.perf(vocab.rnd(3), vocab.ath(1))
    assert (vocab.ath(1), "COMPETED_IN", perf_id) in client.rels
    assert (perf_id, "OF_ROUND", vocab.rnd(3)) in client.rels
    assert (vocab.rnd(3), "OF_EVENT", vocab.evt(1)) in client.rels
    assert (vocab.evt(1), "IN_DISCIPLINE", vocab.disc("L")) in client.rels
    assert (vocab.ath(1), "HAS_RATING", vocab.rat(vocab.ath(1), "L")) in client.rels


# ---------------------------------------------------------------------------
# FACED scope + aggregation.
# ---------------------------------------------------------------------------


def test_faced_only_final_and_semi(seeded_engine: pg.Engine) -> None:
    client = FakeGraphClient()
    _run(seeded_engine, client)

    faced_pairs = {(s, t) for (s, r, t) in client.rels if r == "FACED"}

    # Athletes 1 and 2 faced in: semi(2), final(3), final(4) → present both ways.
    assert (vocab.ath(1), vocab.ath(2)) in faced_pairs
    assert (vocab.ath(2), vocab.ath(1)) in faced_pairs

    # Athlete 3 only appears in qualification + semi. The semi pairs it with 1,2.
    assert (vocab.ath(1), vocab.ath(3)) in faced_pairs

    # Athlete 4 appears ONLY in qualification → never in any FACED edge.
    for s, t in faced_pairs:
        assert vocab.ath(4) not in (s, t)


def test_faced_aggregated_per_pair(seeded_engine: pg.Engine) -> None:
    client = FakeGraphClient()
    _run(seeded_engine, client)

    # 1↔2 faced across rounds 2 (semi), 3 (final), 4 (final) = 3 rounds,
    # collapsed into ONE aggregated directed edge each way.
    props = client.rels[(vocab.ath(1), "FACED", vocab.ath(2))]
    assert props is not None
    assert props["count"] == 3
    assert props["round_ids"] == [2, 3, 4]
    assert props["first_date"] == "2024-06-01"
    assert props["last_date"] == "2024-07-01"

    # 1↔3 only faced in the semi (round 2).
    props13 = client.rels[(vocab.ath(1), "FACED", vocab.ath(3))]
    assert props13 is not None
    assert props13["count"] == 1
    assert props13["round_ids"] == [2]

    # Exactly one logical edge per ordered pair (no per-round duplicates).
    faced_calls = [c for c in client.rel_calls if c[1] == "FACED"]
    assert len(faced_calls) == len(set(faced_calls))


# ---------------------------------------------------------------------------
# Idempotency.
# ---------------------------------------------------------------------------


def test_idempotent_rerun(seeded_engine: pg.Engine) -> None:
    first = FakeGraphClient()
    _run(seeded_engine, first)
    second = FakeGraphClient()
    _run(seeded_engine, second)

    # Same logical node/edge sets and same number of MERGE calls on a re-run:
    # MERGE is keyed, so the graph state is identical (0 net changes).
    assert first.nodes.keys() == second.nodes.keys()
    assert first.rels.keys() == second.rels.keys()
    assert len(first.node_calls) == len(second.node_calls)
    assert len(first.rel_calls) == len(second.rel_calls)

    # No node id is MERGEd more than once within a single run (dedup-aware).
    node_ids = [nid for (_label, nid) in first.node_calls]
    assert len(node_ids) == len(set(node_ids))


# ---------------------------------------------------------------------------
# Count validation pass / fail.
# ---------------------------------------------------------------------------


def test_count_validation_passes(seeded_engine: pg.Engine) -> None:
    client = FakeGraphClient()
    report = _run(seeded_engine, client)
    # 11 results, 1 DNS → 10 performances.
    assert report.src_results == 11
    assert report.filtered["performance_skipped_dns"] == 1
    assert report.node_performances == 10
    validate_counts(report)  # must not raise


def test_count_validation_fails_on_drift(seeded_engine: pg.Engine) -> None:
    client = FakeGraphClient()
    report = _run(seeded_engine, client)
    # Inject unexplained drift: pretend a node went missing.
    report.node_athletes -= 1
    with pytest.raises(CountValidationError, match="athletes"):
        validate_counts(report)


def test_count_validation_fails_on_performance_drift(seeded_engine: pg.Engine) -> None:
    client = FakeGraphClient()
    report = _run(seeded_engine, client)
    # Drop a DNS-filter count without changing performances → mismatch surfaces.
    report.filtered["performance_skipped_dns"] = 0
    with pytest.raises(CountValidationError, match="performances"):
        validate_counts(report)


# ---------------------------------------------------------------------------
# Out-of-vocab guard.
# ---------------------------------------------------------------------------


def test_out_of_vocab_label_raises() -> None:
    client = FakeGraphClient()
    with pytest.raises(ValueError, match="Unknown node label"):
        client.merge_node("Widget", "x:1", {})


def test_out_of_vocab_rel_raises() -> None:
    client = FakeGraphClient()
    with pytest.raises(ValueError, match="Unknown relationship type"):
        client.merge_rel("a:1", "KNOWS", "b:2")
