"""Self-contained Neo4j access layer for the Climber Network API.

This module is intentionally self-contained: it does NOT import from
``src/climber_network/graph/client`` so that the Vercel Python runtime can
import ``api/`` without needing the full package installed.  It MAY import
lightweight helpers from ``src/climber_network`` (e.g. config).

Gotchas (same as graph/client.py)
----------------------------------
* **certifi / macOS TLS**: Set ``SSL_CERT_FILE`` to certifi's bundle before
  constructing any ``GraphDatabase.driver``. The ``setdefault`` call is a
  no-op on Linux / Vercel where a system bundle is already present.
* **Aura username**: The Neo4j Aura username is the *instance id*, not the
  literal string ``neo4j``.  Always read from the environment.
* **Singleton**: The module-level ``_driver`` is reused across requests on
  serverless warm invocations to avoid connection-pool churn.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv

# Load .env from repo root so this module works when invoked by uvicorn locally
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# macOS framework Python lacks a CA bundle; point TLS at certifi so
# neo4j+s:// connections to Aura verify correctly.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from neo4j import GraphDatabase  # noqa: E402

_driver = None


def _get_driver():  # type: ignore[return]
    """Return the cached Neo4j driver, constructing it on first call."""
    global _driver
    if _driver is None:
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def health() -> dict[str, object]:
    """Return ``{"status": "ok"}`` plus live node/relationship counts."""
    with _get_driver().session() as s:
        nodes: int = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels: int = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return {"status": "ok", "nodes": nodes, "relationships": rels}


def graph_stats() -> dict[str, int]:
    """Return ``{"nodes": <int>, "relationships": <int>}``."""
    with _get_driver().session() as s:
        nodes: int = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels: int = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return {"nodes": nodes, "relationships": rels}


def run_read(cypher: str, **params: Any) -> list[dict[str, Any]]:
    """Execute a read-only Cypher query and return rows as plain dicts.

    This is the single read accessor used by ``api.rag``; tests swap
    ``_driver`` for a fake whose ``session().run(...)`` yields seeded records,
    so this function is exercised end-to-end without a live database.
    """
    with _get_driver().session() as s:
        result = s.run(cypher, **params)
        return [dict(record) for record in result]
