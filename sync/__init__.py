"""sync — Re-runnable ingest jobs that build the Neo4j graph from upstream data.

Each module here reads a source (read-only) and idempotently MERGEs nodes and
relationships into Neo4j via ``climber_network.graph.client.GraphClient``. No
module in this package may import the sibling ``climbing_elo`` /
``knowledge_graph`` projects (isolation rule, see CLAUDE.md).
"""

from __future__ import annotations
