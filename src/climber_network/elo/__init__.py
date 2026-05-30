"""climber_network.elo — Self-contained expected-rank library.

Pure-Python (``math`` only) logic for turning a roster of athlete strength
ratings (``mu``) into expected finishing ranks. No graph, database, numpy, or
external-service dependency. This intentionally re-implements the small bit of
probability math needed here rather than importing any sibling project.

Submodules are explicitly re-exported below so that
``from climber_network.elo import montecarlo`` resolves **deterministically**
under mypy. Without the re-export, ``from <pkg> import <submodule>`` triggers an
order-dependent ``Module "climber_network.elo" has no attribute "<submodule>"``
flake (see issue #52); listing each submodule in ``__all__`` makes it a real,
always-resolved package attribute (the same pattern ``source``/``geo`` use).
When you add a new submodule here, add it to the re-export list and ``__all__``
— ``tests/test_elo_package_exports.py`` enforces this.
"""

from __future__ import annotations

from climber_network.elo import (
    advancement,
    calibration,
    expected,
    montecarlo,
    reps,
    rested,
    season,
    weightfit,
)
from climber_network.elo.expected import (
    expected_finish_ranks,
    expected_rank_for,
)

__all__ = [
    "advancement",
    "calibration",
    "expected",
    "expected_finish_ranks",
    "expected_rank_for",
    "montecarlo",
    "reps",
    "rested",
    "season",
    "weightfit",
]
