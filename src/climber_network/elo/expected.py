"""Expected finishing ranks from pairwise strength ratings.

Given a roster of athletes, each with a real-valued strength ``mu`` (in the
familiar Elo / Bradley-Terry scale where a difference of ``scale`` log-units
corresponds to a fixed win probability), we model every head-to-head ordering
independently with a logistic link and sum the implied pairwise rankings.

Model
-----
For two athletes ``i`` and ``j`` with strengths ``mu_i`` and ``mu_j``, the
probability that ``j`` finishes *ahead of* ``i`` (i.e. ``j`` beats ``i``) is

    P(j ahead of i) = logistic((mu_j - mu_i) / scale)
                    = 1 / (1 + exp(-(mu_j - mu_i) / scale))

The **expected rank** of athlete ``i`` (rank 1 = winner) is then one plus the
expected number of opponents who finish ahead of them:

    E[rank_i] = 1 + Σ_{j != i} P(j ahead of i)

This is the standard linearity-of-expectation result: each opponent ``j``
contributes an indicator ``1{j ahead of i}`` to ``i``'s rank, whose expectation
is exactly ``P(j ahead of i)``. The constant ``1`` accounts for the rank of the
best possible finish.

Properties
----------
* **Monotonicity** — a larger ``mu`` strictly lowers (improves) expected rank,
  since every pairwise term decreases as ``mu_i`` grows.
* **Sum identity** — because ``P(j ahead of i) + P(i ahead of j) = 1`` for every
  unordered pair, the expected ranks of an ``N``-athlete field always sum to
  ``N * (N + 1) / 2`` (the sum ``1 + 2 + ... + N``), regardless of the ratings.
* **Ties** — equal ``mu`` gives ``P = 0.5`` for that pair, contributing ``0.5``
  to each side; a fully equal field therefore yields ``E[rank] = (N + 1) / 2``
  for everyone. Duplicate ``mu`` values are handled gracefully by the same math
  (no special-casing required).

Scale
-----
``scale`` is the logistic temperature. The default of ``173.7`` matches the
Glicko convention ``400 / ln(10)``, under which a 400-point rating advantage
corresponds to a 10:1 (~0.909) head-to-head win probability — the classic Elo
calibration. Callers may pass any positive ``scale`` to widen or sharpen the
spread of probabilities.
"""

from __future__ import annotations

import math

__all__ = ["expected_finish_ranks", "expected_rank_for"]

# 400 / ln(10): an Elo gap of 400 points -> ~0.909 win probability.
DEFAULT_SCALE: float = 173.7


def _p_ahead(mu_other: float, mu_self: float, scale: float) -> float:
    """Logistic probability that ``mu_other`` finishes ahead of ``mu_self``.

    Computed in a numerically stable way (no ``exp`` overflow for large
    favourable gaps).
    """
    diff = (mu_other - mu_self) / scale
    if diff >= 0.0:
        return 1.0 / (1.0 + math.exp(-diff))
    # exp(diff) avoids overflow when diff is very negative.
    e = math.exp(diff)
    return e / (1.0 + e)


def expected_finish_ranks(
    roster: list[tuple[str, float]],
    scale: float = DEFAULT_SCALE,
) -> dict[str, float]:
    """Expected finishing rank for every athlete in ``roster``.

    Each entry of ``roster`` is a ``(athlete_id, mu)`` tuple where ``mu`` is the
    athlete's strength rating. Higher ``mu`` means stronger, so a higher ``mu``
    yields a *lower* (better) expected rank. Rank 1 is the winner.

    The expected rank of athlete ``i`` is::

        E[rank_i] = 1 + sum_{j != i} logistic((mu_j - mu_i) / scale)

    Parameters
    ----------
    roster:
        List of ``(athlete_id, mu)`` tuples. Athlete ids should be unique; if an
        id repeats, the later entry wins in the returned mapping (though all
        entries still contribute to every pairwise sum).
    scale:
        Positive logistic temperature. See module docstring; defaults to
        ``173.7`` (``400 / ln(10)``).

    Returns
    -------
    dict[str, float]
        Mapping from ``athlete_id`` to expected finishing rank. The values sum
        to ``N * (N + 1) / 2`` for a field of ``N`` athletes.

    Raises
    ------
    ValueError
        If ``scale`` is not strictly positive.

    Examples
    --------
    >>> ranks = expected_finish_ranks([("a", 100.0), ("b", 0.0)])
    >>> ranks["a"] < ranks["b"]
    True
    >>> round(ranks["a"] + ranks["b"], 6)
    3.0
    """
    if scale <= 0.0:
        msg = f"scale must be strictly positive, got {scale!r}"
        raise ValueError(msg)

    if not roster:
        return {}

    ranks: dict[str, float] = {}
    for athlete_id, mu_i in roster:
        expected = 1.0
        for other_id, mu_j in roster:
            if other_id == athlete_id:
                continue
            expected += _p_ahead(mu_j, mu_i, scale)
        ranks[athlete_id] = expected
    return ranks


def expected_rank_for(
    athlete_id: str,
    roster: list[tuple[str, float]],
    scale: float = DEFAULT_SCALE,
) -> float:
    """Expected finishing rank for a single ``athlete_id`` within ``roster``.

    Convenience wrapper around :func:`expected_finish_ranks`. ``athlete_id``
    must appear in ``roster``.

    Raises
    ------
    KeyError
        If ``athlete_id`` is not present in ``roster``.
    ValueError
        If ``scale`` is not strictly positive.
    """
    ranks = expected_finish_ranks(roster, scale=scale)
    if athlete_id not in ranks:
        msg = f"athlete_id {athlete_id!r} not found in roster"
        raise KeyError(msg)
    return ranks[athlete_id]
