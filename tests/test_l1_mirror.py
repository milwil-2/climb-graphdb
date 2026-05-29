"""Tests for the P1 L1 competition mirror (sync.pg_to_neo4j).

These tests use:
* the shared in-memory SQLite ``seeded_session`` fixture (a few athletes,
  events, rounds, results, ratings — including final, semi, and qualification
  rounds), defined in ``tests/conftest.py``;
* the shared ``FakeGraphClient`` (also in ``conftest.py``) that records every
  merge_node / merge_rel call, so NO live Neo4j connection is ever made.

They assert node/edge creation + ids, the FACED scope (final/semi only,
aggregated per ordered pair), idempotency (run twice → no duplicate logical
MERGEs), count-validation pass/fail, and that an out-of-vocab label/rel raises.
"""

from __future__ import annotations

import pytest

from climber_network import vocab
from climber_network.source import pg
from sync.pg_to_neo4j import (
    CountValidationError,
    sync_graph,
    validate_counts,
)
from tests.conftest import FakeGraphClient

# ---------------------------------------------------------------------------
# Node / edge / id assertions.
# ---------------------------------------------------------------------------


def _run(session: pg.Session, client: FakeGraphClient):
    return sync_graph(client, session)


def test_nodes_created_with_correct_ids(seeded_session: pg.Session) -> None:
    client = FakeGraphClient()
    _run(seeded_session, client)

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


def test_structural_edges(seeded_session: pg.Session) -> None:
    client = FakeGraphClient()
    _run(seeded_session, client)

    perf_id = vocab.perf(vocab.rnd(3), vocab.ath(1))
    assert (vocab.ath(1), "COMPETED_IN", perf_id) in client.rels
    assert (perf_id, "OF_ROUND", vocab.rnd(3)) in client.rels
    assert (vocab.rnd(3), "OF_EVENT", vocab.evt(1)) in client.rels
    assert (vocab.evt(1), "IN_DISCIPLINE", vocab.disc("L")) in client.rels
    assert (vocab.ath(1), "HAS_RATING", vocab.rat(vocab.ath(1), "L")) in client.rels


# ---------------------------------------------------------------------------
# FACED scope + aggregation.
# ---------------------------------------------------------------------------


def test_faced_only_final_and_semi(seeded_session: pg.Session) -> None:
    client = FakeGraphClient()
    _run(seeded_session, client)

    faced_pairs = {(s, t) for (s, r, t) in client.rels if r == "FACED"}

    # Athletes 1 and 2 faced in: semi(2), final(3), final(4) → present both ways.
    assert (vocab.ath(1), vocab.ath(2)) in faced_pairs
    assert (vocab.ath(2), vocab.ath(1)) in faced_pairs

    # Athlete 3 only appears in qualification + semi. The semi pairs it with 1,2.
    assert (vocab.ath(1), vocab.ath(3)) in faced_pairs

    # Athlete 4 appears ONLY in qualification → never in any FACED edge.
    for s, t in faced_pairs:
        assert vocab.ath(4) not in (s, t)


def test_faced_aggregated_per_pair(seeded_session: pg.Session) -> None:
    client = FakeGraphClient()
    _run(seeded_session, client)

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


def test_idempotent_rerun(seeded_session: pg.Session) -> None:
    first = FakeGraphClient()
    _run(seeded_session, first)
    second = FakeGraphClient()
    _run(seeded_session, second)

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


def test_count_validation_passes(seeded_session: pg.Session) -> None:
    client = FakeGraphClient()
    report = _run(seeded_session, client)
    # 11 results, 1 DNS → 10 performances.
    assert report.src_results == 11
    assert report.filtered["performance_skipped_dns"] == 1
    assert report.node_performances == 10
    validate_counts(report)  # must not raise


def test_count_validation_fails_on_drift(seeded_session: pg.Session) -> None:
    client = FakeGraphClient()
    report = _run(seeded_session, client)
    # Inject unexplained drift: pretend a node went missing.
    report.node_athletes -= 1
    with pytest.raises(CountValidationError, match="athletes"):
        validate_counts(report)


def test_count_validation_fails_on_performance_drift(seeded_session: pg.Session) -> None:
    client = FakeGraphClient()
    report = _run(seeded_session, client)
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
