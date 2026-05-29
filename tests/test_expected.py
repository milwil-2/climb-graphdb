"""Unit tests for climber_network.elo.expected.

All tests are pure-Python — no database, numpy, or external service required.
"""

from __future__ import annotations

import math

import pytest

from climber_network.elo.expected import (
    DEFAULT_SCALE,
    expected_finish_ranks,
    expected_rank_for,
)


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def test_empty_roster() -> None:
    assert expected_finish_ranks([]) == {}


def test_single_athlete_ranks_first() -> None:
    assert expected_finish_ranks([("solo", 1500.0)]) == {"solo": 1.0}


def test_monotonicity_higher_mu_means_lower_rank() -> None:
    """A strictly increasing mu must give strictly decreasing expected rank."""
    roster = [("a", 0.0), ("b", 100.0), ("c", 250.0), ("d", 500.0)]
    ranks = expected_finish_ranks(roster)
    # d strongest -> best (lowest) rank; a weakest -> worst (highest) rank.
    assert ranks["d"] < ranks["c"] < ranks["b"] < ranks["a"]


def test_ranks_sum_to_triangular_number() -> None:
    """Expected ranks of an N-field sum to N(N+1)/2 for any ratings."""
    roster = [("a", 12.0), ("b", -340.0), ("c", 1000.0), ("d", 5.5), ("e", 88.0)]
    n = len(roster)
    ranks = expected_finish_ranks(roster)
    assert ranks.keys() == {"a", "b", "c", "d", "e"}
    assert math.isclose(sum(ranks.values()), n * (n + 1) / 2, rel_tol=1e-12)


def test_equal_mu_field_all_ranks_are_midpoint() -> None:
    """A fully equal field yields E[rank] = (N + 1) / 2 for everyone."""
    n = 6
    roster = [(f"x{i}", 1500.0) for i in range(n)]
    ranks = expected_finish_ranks(roster)
    expected = (n + 1) / 2
    for value in ranks.values():
        assert math.isclose(value, expected, rel_tol=1e-12)


def test_duplicate_mu_handled_gracefully() -> None:
    """Ties (duplicate mu) contribute 0.5 each and remain symmetric."""
    roster = [("a", 100.0), ("b", 100.0), ("c", 0.0)]
    ranks = expected_finish_ranks(roster)
    # a and b are tied -> identical expected ranks, both ahead of c.
    assert math.isclose(ranks["a"], ranks["b"], rel_tol=1e-12)
    assert ranks["a"] < ranks["c"]
    assert math.isclose(sum(ranks.values()), 6.0, rel_tol=1e-12)


def test_hand_checked_two_athletes() -> None:
    """Two athletes: E[rank_i] = 1 + P(other ahead of i)."""
    scale = DEFAULT_SCALE
    mu_a, mu_b = 200.0, 50.0
    roster = [("a", mu_a), ("b", mu_b)]
    ranks = expected_finish_ranks(roster, scale=scale)

    p_b_ahead_a = _logistic((mu_b - mu_a) / scale)
    p_a_ahead_b = _logistic((mu_a - mu_b) / scale)
    assert math.isclose(ranks["a"], 1.0 + p_b_ahead_a, rel_tol=1e-12)
    assert math.isclose(ranks["b"], 1.0 + p_a_ahead_b, rel_tol=1e-12)
    assert math.isclose(sum(ranks.values()), 3.0, rel_tol=1e-12)


def test_hand_checked_three_athletes() -> None:
    """Three athletes: explicit pairwise sum against a fresh computation."""
    scale = 173.7
    mus = {"a": 300.0, "b": 100.0, "c": -50.0}
    roster = [(k, v) for k, v in mus.items()]
    ranks = expected_finish_ranks(roster, scale=scale)

    for i, mu_i in mus.items():
        expected = 1.0
        for j, mu_j in mus.items():
            if j == i:
                continue
            expected += _logistic((mu_j - mu_i) / scale)
        assert math.isclose(ranks[i], expected, rel_tol=1e-12)

    assert math.isclose(sum(ranks.values()), 6.0, rel_tol=1e-12)


def test_scale_affects_spread() -> None:
    """Smaller scale sharpens probabilities, widening the rank spread."""
    roster = [("strong", 200.0), ("weak", -200.0)]
    sharp = expected_finish_ranks(roster, scale=50.0)
    soft = expected_finish_ranks(roster, scale=400.0)
    # Sharper scale pushes the favourite closer to rank 1.
    assert sharp["strong"] < soft["strong"]
    assert sharp["weak"] > soft["weak"]


def test_non_positive_scale_raises() -> None:
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        expected_finish_ranks([("a", 1.0), ("b", 2.0)], scale=0.0)
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        expected_finish_ranks([("a", 1.0)], scale=-10.0)


def test_extreme_gap_numerically_stable() -> None:
    """Very large rating gaps must not overflow and stay in [1, N]."""
    roster = [("god", 1e9), ("mortal", -1e9)]
    ranks = expected_finish_ranks(roster, scale=1.0)
    assert math.isclose(ranks["god"], 1.0, abs_tol=1e-9)
    assert math.isclose(ranks["mortal"], 2.0, abs_tol=1e-9)


def test_expected_rank_for_matches_full_mapping() -> None:
    roster = [("a", 10.0), ("b", 20.0), ("c", 30.0)]
    full = expected_finish_ranks(roster)
    assert math.isclose(expected_rank_for("b", roster), full["b"], rel_tol=1e-12)


def test_expected_rank_for_missing_athlete_raises() -> None:
    roster = [("a", 10.0), ("b", 20.0)]
    with pytest.raises(KeyError, match="not found in roster"):
        expected_rank_for("zzz", roster)
