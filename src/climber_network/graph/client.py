"""climber_network.graph.client — Neo4j driver wrapper with injection-safe helpers.

Gotchas
-------
* **certifi / macOS TLS**: Python framework builds on macOS ship without a
  system CA bundle, so ``neo4j+s://`` (Aura) connections fail with
  ``SSL: CERTIFICATE_VERIFY_FAILED``. Setting ``SSL_CERT_FILE`` to certifi's
  bundle at import time fixes this. The ``setdefault`` call is harmless on
  Linux / Vercel (where the system bundle is present and already set).

* **Aura username**: The Neo4j Aura username is the *instance id*, not the
  literal string ``neo4j``. Always read ``NEO4J_USER`` from the environment
  via ``config.NEO4J_USER()``.

* **Serverless warm-reuse**: ``get_client()`` returns a module-level
  singleton. On Vercel / Lambda the process may handle multiple requests
  without restarting; reusing the driver avoids connection-pool churn.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import certifi

from climber_network import config
from climber_network.vocab import assert_label, assert_rel

# Must be set before the neo4j driver is first imported/used.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from neo4j import (  # noqa: E402  (import after SSL_CERT_FILE env patch)
    Driver,
    GraphDatabase,
    ManagedTransaction,
)

#: Max rows per UNWIND transaction — keeps Aura transactions bounded in size
#: while still collapsing tens of thousands of writes into a handful of trips.
_BATCH_SIZE = 5_000


def _chunked(rows: list[dict[str, Any]], size: int = _BATCH_SIZE) -> Iterator[list[dict[str, Any]]]:
    """Yield *rows* in successive chunks of at most *size* items."""
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


# ---------------------------------------------------------------------------
# GraphClient
# ---------------------------------------------------------------------------


class GraphClient:
    """Thin wrapper around the Neo4j driver with injection-safe merge helpers.

    Usage::

        client = GraphClient()
        client.verify_connectivity()
        client.merge_node("Athlete", "ath:42", {"name": "Adam Ondra"})
        client.close()
    """

    def __init__(self) -> None:
        self._driver: Driver | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_driver(self) -> Driver:
        """Return the lazily-created driver, constructing it on first call."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                config.NEO4J_URI(),
                auth=(config.NEO4J_USER(), config.NEO4J_PASSWORD()),
            )
        return self._driver

    def verify_connectivity(self) -> None:
        """Raise if the Neo4j server is unreachable."""
        self._get_driver().verify_connectivity()

    def close(self) -> None:
        """Close the underlying driver and release connection-pool resources."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _run_write(self, cypher: str, **params: Any) -> None:
        """Run *cypher* in a managed write transaction (auto-retries transients).

        ``execute_write`` transparently retries the unit of work on transient
        failures — notably ``SessionExpired`` when Aura drops a Bolt connection
        mid-run — acquiring a fresh connection each attempt.
        """

        def _work(tx: ManagedTransaction) -> None:
            tx.run(cypher, **params).consume()

        with self._get_driver().session() as session:
            session.execute_write(_work)

    def merge_node(self, label: str, node_id: str, props: dict[str, Any]) -> None:
        """MERGE a node by *node_id* and SET all *props*.

        *label* is validated against VALID_NODE_LABELS before interpolation —
        calling ``assert_label`` is the injection-safety gate.
        """
        safe_label = assert_label(label)
        # SET n:Entity stamps the shared label that backs the entity_id index,
        # so relationship MERGEs can match this node by id without its label.
        cypher = f"MERGE (n:{safe_label} {{id:$id}}) SET n:Entity, n += $props"
        self._run_write(cypher, id=node_id, props=props)

    def merge_nodes(self, label: str, rows: list[dict[str, Any]]) -> None:
        """Batch-MERGE many nodes of *label* via chunked UNWIND transactions.

        *rows* is a list of ``{"id": <str>, "props": <dict>}``. Chunked managed
        transactions replace N per-call round-trips — far faster and more
        resilient on Aura than calling :meth:`merge_node` in a loop.
        """
        if not rows:
            return
        safe_label = assert_label(label)
        cypher = (
            f"UNWIND $rows AS row MERGE (n:{safe_label} {{id: row.id}}) "
            "SET n:Entity, n += row.props"
        )
        for chunk in _chunked(rows):
            self._run_write(cypher, rows=chunk)

    def merge_rel(
        self,
        src_id: str,
        rel_type: str,
        tgt_id: str,
        props: dict[str, Any] | None = None,
    ) -> None:
        """MERGE a relationship between two nodes (matched by id).

        *rel_type* is validated against VALID_REL_TYPES before interpolation.
        If *props* is provided, SET them on the relationship after the merge.
        Runs inside a managed transaction (retries on transient failures).
        """
        safe_rel = assert_rel(rel_type)
        if props:
            cypher = (
                "MATCH (a:Entity {id:$src_id}), (b:Entity {id:$tgt_id}) "
                f"MERGE (a)-[r:{safe_rel}]->(b) "
                "SET r += $props"
            )
            self._run_write(cypher, src_id=src_id, tgt_id=tgt_id, props=props)
        else:
            cypher = (
                "MATCH (a:Entity {id:$src_id}), (b:Entity {id:$tgt_id}) "
                f"MERGE (a)-[:{safe_rel}]->(b)"
            )
            self._run_write(cypher, src_id=src_id, tgt_id=tgt_id)

    def merge_rels(self, rel_type: str, rows: list[dict[str, Any]]) -> None:
        """Batch-MERGE many relationships of *rel_type* via chunked UNWIND txns.

        *rows* is a list of ``{"src_id": <str>, "tgt_id": <str>, "props": <dict>}``
        (``props`` optional / may be ``{}``). Chunked managed transactions
        replace N per-call round-trips.
        """
        if not rows:
            return
        safe_rel = assert_rel(rel_type)
        cypher = (
            "UNWIND $rows AS row "
            "MATCH (a:Entity {id: row.src_id}), (b:Entity {id: row.tgt_id}) "
            f"MERGE (a)-[r:{safe_rel}]->(b) "
            "SET r += coalesce(row.props, {})"
        )
        for chunk in _chunked(rows):
            self._run_write(cypher, rows=chunk)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Execute a read-only Cypher query and return results as plain dicts."""
        with self._get_driver().session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    def graph_stats(self) -> dict[str, int]:
        """Return total node and relationship counts.

        Returns:
            ``{"nodes": <int>, "relationships": <int>}``
        """
        with self._get_driver().session() as session:
            node_rec = session.run("MATCH (n) RETURN count(n) AS c").single()
            rel_rec = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
        nodes: int = node_rec["c"] if node_rec is not None else 0
        rels: int = rel_rec["c"] if rel_rec is not None else 0
        return {"nodes": nodes, "relationships": rels}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: GraphClient | None = None


def get_client() -> GraphClient:
    """Return the module-level GraphClient singleton.

    The singleton is created on first call and reused for the lifetime of the
    process — safe for serverless warm invocations (Vercel / Lambda).
    """
    global _client
    if _client is None:
        _client = GraphClient()
    return _client
