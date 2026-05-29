"""Shared pytest fixtures for the climb-graphdb test suite.

This module provides the reusable test scaffolding referenced throughout
``tests/``:

* throwaway environment defaults set BEFORE any app/config import, so merely
  importing a test module can never crash on a missing ``DATABASE_URL`` /
  ``NEO4J_*`` (CI has no real credentials);
* a fresh-per-test in-memory SQLite source-DB session built from
  ``source.pg`` (disposed on teardown — also silences the historical
  ``ResourceWarning: unclosed database``);
* a ``FakeGraphClient`` recorder (the sync ``GraphWriter`` shape) that
  validates labels/rels via ``vocab.assert_label`` / ``assert_rel`` exactly
  like the real ``GraphClient``;
* a ``FakeNeo4jDriver`` (the ``api.db`` driver shape) plus domain factory
  fixtures that seed a representative competition.

Nothing here imports the sibling ``climbing_elo`` / ``knowledge_graph``
projects (hard isolation rule, see CLAUDE.md).
"""

from __future__ import annotations

import os

# --- Throwaway env defaults (MUST run before importing config/app/pg) --------
# These are dummy values: tests never hit a live database or graph. Using
# setdefault means a real local .env still wins for the rare `-m network` run.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# Dummy Neo4j creds — the fallback when no real .env is present. They point at a
# dead localhost, so a `-m network` test that reaches a live driver with these
# would error on connect; ``has_live_neo4j_creds`` lets such tests skip cleanly
# instead (see the ``live_neo4j`` fixture below).
_DUMMY_NEO4J = {
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USER": "test-instance-id",
    "NEO4J_PASSWORD": "test-password",
}
for _key, _val in _DUMMY_NEO4J.items():
    os.environ.setdefault(_key, _val)


def has_live_neo4j_creds() -> bool:
    """True when NEO4J_* point at a real instance, not the dummy fallbacks.

    A real local ``.env`` overrides the ``setdefault`` dummies above, so any
    env var differing from its dummy value signals live credentials.
    """
    return any(os.environ.get(key) != val for key, val in _DUMMY_NEO4J.items())


from collections.abc import Iterator  # noqa: E402
from datetime import date  # noqa: E402
from typing import Any  # noqa: E402

import pytest  # noqa: E402

from climber_network.source import pg  # noqa: E402
from climber_network.vocab import assert_label, assert_rel  # noqa: E402


@pytest.fixture
def live_neo4j() -> None:
    """Skip a ``-m network`` test unless real NEO4J_* credentials are present.

    Without a real ``.env`` the dummy fallbacks point at a dead localhost, so a
    live driver call would error on connect; this skips cleanly instead.
    """
    if not has_live_neo4j_creds():
        pytest.skip("no live NEO4J_* credentials (using dummy test defaults)")


# ---------------------------------------------------------------------------
# Fake graph writer — sync.GraphWriter shape (merge_node / merge_rel).
# ---------------------------------------------------------------------------


class FakeGraphClient:
    """Records merge calls; mirrors the real client's vocab-gating behaviour.

    Implements the structural ``sync.pg_to_neo4j.GraphWriter`` protocol plus a
    canned ``run_read`` for read-query tests. Every label / relationship type
    passes through ``assert_label`` / ``assert_rel`` so out-of-vocab values
    raise here exactly as they would against live Neo4j.
    """

    def __init__(self, read_results: dict[str, list[dict[str, Any]]] | None = None) -> None:
        # node_id -> latest props (MERGE semantics: keyed, last write wins).
        self.nodes: dict[str, dict[str, Any]] = {}
        self.node_labels: dict[str, str] = {}
        # (src, rel, tgt) -> latest props.
        self.rels: dict[tuple[str, str, str], dict[str, Any] | None] = {}
        # Raw call logs for duplicate / ordering assertions.
        self.node_calls: list[tuple[str, str]] = []
        self.rel_calls: list[tuple[str, str, str]] = []
        # Canned cypher -> rows mapping for run_read query tests.
        self._read_results = read_results or {}

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

    def merge_nodes(self, label: str, rows: list[dict[str, Any]]) -> None:
        """Batch variant — records each row exactly as ``merge_node`` would."""
        for row in rows:
            self.merge_node(label, row["id"], row["props"])

    def merge_rels(self, rel_type: str, rows: list[dict[str, Any]]) -> None:
        """Batch variant — records each row exactly as ``merge_rel`` would."""
        for row in rows:
            self.merge_rel(row["src_id"], rel_type, row["tgt_id"], row.get("props"))

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Return canned rows for *cypher* (empty list if unseeded)."""
        return self._read_results.get(cypher, [])


# ---------------------------------------------------------------------------
# Fake Neo4j driver — api.db driver shape (.session() -> .run(...).single()).
# ---------------------------------------------------------------------------


class _FakeCountRecord:
    def __init__(self, value: int) -> None:
        self._value = value

    def __getitem__(self, key: str) -> int:
        # The count queries only ever read the "c" column.
        return self._value


class _FakeRowResult:
    """Result wrapping seeded read rows: iterable + ``single()``-able."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)

    def single(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None


class _FakeCountResult:
    def __init__(self, value: int) -> None:
        self._value = value

    def single(self) -> _FakeCountRecord:
        return _FakeCountRecord(self._value)


class _FakeSession:
    def __init__(
        self,
        nodes: int,
        relationships: int,
        read_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._nodes = nodes
        self._relationships = relationships
        self._read_results = read_results or {}

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run(self, cypher: str, **params: Any) -> Any:
        # Seeded read queries (api.rag) take precedence, keyed by exact cypher.
        if cypher in self._read_results:
            return _FakeRowResult(self._read_results[cypher])
        # Otherwise behave as a count query (graph_stats / health).
        value = self._relationships if "[r]" in cypher or "()-[" in cypher else self._nodes
        return _FakeCountResult(value)


class FakeNeo4jDriver:
    """Stand-in for the live Neo4j driver used by ``api.db._get_driver``.

    Returns seeded node / relationship counts so ``graph_stats()`` and
    ``health()`` can be exercised without a live database. An optional
    ``read_results`` map (exact-cypher -> rows) lets ``api.db.run_read`` based
    callers (``api.rag``) be exercised offline too.
    """

    def __init__(
        self,
        nodes: int = 0,
        relationships: int = 0,
        read_results: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.nodes = nodes
        self.relationships = relationships
        self.read_results = read_results or {}

    def session(self) -> _FakeSession:
        return _FakeSession(self.nodes, self.relationships, self.read_results)


# ---------------------------------------------------------------------------
# Source-DB fixtures — fresh in-memory SQLite per test.
# ---------------------------------------------------------------------------


@pytest.fixture
def source_engine() -> Iterator[pg.Engine]:
    """Yield a fresh in-memory SQLite engine with the read-model schema created.

    Teardown disposes the engine so no connection is left open (fixes the
    historical ``ResourceWarning: unclosed database``).
    """
    engine = pg.make_engine("sqlite://")
    pg.Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def source_session(source_engine: pg.Engine) -> Iterator[pg.Session]:
    """Yield a writable Session bound to the fresh source engine (for seeding)."""
    session = pg.Session(source_engine)
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Domain factory fixtures — build representative source rows.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_athlete(source_session: pg.Session) -> pg.Athlete:
    """Seed and return a single athlete with a minimal event/round/result.

    Shape: one athlete, one Lead event, one final round, one result.
    """
    athlete = pg.Athlete(id=1, name="Ada", gender="F", nationality="USA")
    event = pg.Event(
        id=1,
        name="World Cup Innsbruck",
        tier="world_cup",
        country="AUT",
        season=2024,
        start_date=date(2024, 6, 1),
        discipline="L",
    )
    rnd = pg.Round(id=1, event_id=1, round_type="final", gender="F", athlete_count=1)
    result = pg.Result(id=1, round_id=1, athlete_id=1, rank=1, score_normalized=1.0)
    source_session.add_all([athlete, event, rnd, result])
    source_session.commit()
    return athlete


@pytest.fixture
def seeded_session(source_session: pg.Session) -> pg.Session:
    """Seed a small but representative competition and return the session.

    Event 1 (Lead) has three rounds:
      - round 1 qualification (4 athletes) — must NOT produce FACED
      - round 2 semi          (3 athletes) — FACED
      - round 3 final         (2 athletes) — FACED
    Event 2 (Boulder) has one final (round 4, 2 athletes) on a later date —
    contributes a second FACED round to a pair, exercising aggregation.
    One result is DNS (documented filter → no Performance).
    """
    source_session.add_all(
        [
            pg.Athlete(id=1, name="Ada", gender="F", nationality="USA"),
            pg.Athlete(id=2, name="Bea", gender="F", nationality="GBR"),
            pg.Athlete(id=3, name="Cleo", gender="F", nationality="JPN"),
            pg.Athlete(id=4, name="Dot", gender="F", nationality="AUT"),
        ]
    )
    source_session.add_all(
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
    source_session.add_all(
        [
            pg.Round(id=1, event_id=1, round_type="qualification", gender="F", athlete_count=4),
            pg.Round(id=2, event_id=1, round_type="semi", gender="F", athlete_count=3),
            pg.Round(id=3, event_id=1, round_type="final", gender="F", athlete_count=2),
            pg.Round(id=4, event_id=2, round_type="final", gender="F", athlete_count=2),
        ]
    )
    source_session.add_all(
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
    source_session.add_all(
        [
            pg.Rating(id=1, athlete_id=1, discipline="L", mu=1600.0, sigma=200.0, n_events=5),
            pg.Rating(id=2, athlete_id=2, discipline="L", mu=1550.0, sigma=210.0, n_events=4),
        ]
    )
    source_session.commit()
    return source_session
