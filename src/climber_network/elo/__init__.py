"""climber_network.elo — Self-contained expected-rank library.

Pure-Python (``math`` only) logic for turning a roster of athlete strength
ratings (``mu``) into expected finishing ranks. No graph, database, numpy, or
external-service dependency. This intentionally re-implements the small bit of
probability math needed here rather than importing any sibling project.
"""

from __future__ import annotations

from climber_network.elo.expected import (
    expected_finish_ranks,
    expected_rank_for,
)

__all__ = ["expected_finish_ranks", "expected_rank_for"]
