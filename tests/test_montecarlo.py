"""Unit tests for climber_network.elo.montecarlo.

Pure-Python only — no database, numpy, or external service. The convergence
test cross-checks the Monte-Carlo mean against the exact closed form in
``climber_network.elo.expected``.
"""

from __future__ import annotations

import math

import pytest

from climber_network.elo.expected import (
    DEFAULT_SCALE,
    expected_finish_ranks,
)
from climber_network.elo.montecarlo import (
    GAUSSIAN,
    PLACKETT_LUCE,
    expected_rank_mc,
    p_podium,
    p_win,
    placement_pmf,
    pmf_entropy,
    rank_std,
    result_percentile,
    summarize,
    surprisal,
)

# A small, well-separated roster reused across several tests.
ROSTER = [
    ("a", 300.0),
    ("b", 100.0),
    ("c", -50.0),
    ("d", -300.0),
]


def test_empty_roster() -> None:
    assert placement_pmf([]) == {}


def test_single_athlete() -> None:
    pmf = placement_pmf([("solo", 1500.0)])
    assert pmf == {"solo": [1.0]}


def test_pmf_sums_to_one_and_correct_length() -> None:
    pmfs = placement_pmf(ROSTER, n_sims=5000, seed=7)
    n = len(ROSTER)
    assert set(pmfs) == {"a", "b", "c", "d"}
    for pmf in pmfs.values():
        assert len(pmf) == n
        assert math.isclose(sum(pmf), 1.0, abs_tol=1e-9)
        assert all(p >= 0.0 for p in pmf)


def test_determinism_same_seed_identical() -> None:
    first = placement_pmf(ROSTER, n_sims=3000, seed=42)
    second = placement_pmf(ROSTER, n_sims=3000, seed=42)
    assert first == second


def test_different_seed_generally_differs() -> None:
    first = placement_pmf(ROSTER, n_sims=3000, seed=1)
    second = placement_pmf(ROSTER, n_sims=3000, seed=2)
    assert first != second


def test_convergence_to_closed_form() -> None:
    """Plackett-Luce MC mean rank must match the exact expected_finish_ranks form.

    Convergence to the logistic closed form is a property of the plackett_luce
    model (the shared link); the default gaussian model is a different family.
    """
    n = len(ROSTER)
    n_sims = 40000
    pmfs = placement_pmf(
        ROSTER,
        n_sims=n_sims,
        seed=2024,
        sample_sigma=False,
        scale=DEFAULT_SCALE,
        model=PLACKETT_LUCE,
    )
    closed = expected_finish_ranks(ROSTER, scale=DEFAULT_SCALE)

    total = 0.0
    for athlete_id, pmf in pmfs.items():
        mc_mean = expected_rank_mc(pmf)
        total += mc_mean
        assert math.isclose(mc_mean, closed[athlete_id], abs_tol=0.15)

    # Sum identity: ranks of an N-field sum to N(N+1)/2.
    assert math.isclose(total, n * (n + 1) / 2, abs_tol=1e-9)


def test_favourite_wins_most() -> None:
    pmfs = placement_pmf(ROSTER, n_sims=20000, seed=99)
    # 'a' is strongest, so it should win most often and 'd' least.
    assert p_win(pmfs["a"]) > p_win(pmfs["b"]) > p_win(pmfs["c"]) > p_win(pmfs["d"])


def test_result_percentile_monotonic_non_decreasing() -> None:
    pmfs = placement_pmf(ROSTER, n_sims=8000, seed=5)
    pmf = pmfs["b"]
    n = len(pmf)
    prev = -1.0
    for r in range(1, n + 1):
        val = result_percentile(pmf, r)
        assert 0.0 <= val <= 1.0 + 1e-12
        assert val >= prev - 1e-12
        prev = val
    # Full CDF at the last rank is 1.
    assert math.isclose(result_percentile(pmf, n), 1.0, abs_tol=1e-9)


def test_surprisal_last_place_for_certain_winner() -> None:
    """A near-certain winner finishing last is far more surprising than first."""
    roster = [("god", 5000.0), ("m1", 0.0), ("m2", -50.0), ("m3", -100.0)]
    pmfs = placement_pmf(roster, n_sims=20000, seed=11, sample_sigma=False)
    pmf = pmfs["god"]
    n = len(pmf)
    s_first = surprisal(pmf, 1)
    s_last = surprisal(pmf, n)
    assert s_last > s_first
    # Winning was nearly certain -> very low surprisal.
    assert s_first < 0.1
    assert s_last > s_first + 2.0


def test_pl_sample_sigma_widens_distribution() -> None:
    """Plackett-Luce: jittering a high-sigma athlete widens its rank distribution."""
    base = placement_pmf(ROSTER, n_sims=20000, seed=3, sample_sigma=False, model=PLACKETT_LUCE)
    sigmas = {"b": 600.0}
    jittered = placement_pmf(
        ROSTER, sigmas=sigmas, n_sims=20000, seed=3, sample_sigma=True, model=PLACKETT_LUCE
    )
    assert rank_std(jittered["b"]) > rank_std(base["b"])


def test_pl_sample_sigma_off_ignores_sigmas() -> None:
    """Plackett-Luce with sample_sigma off must ignore the sigmas map entirely."""
    sigmas = {"b": 600.0}
    without = placement_pmf(ROSTER, n_sims=4000, seed=8, sample_sigma=False, model=PLACKETT_LUCE)
    with_off = placement_pmf(
        ROSTER, sigmas=sigmas, n_sims=4000, seed=8, sample_sigma=False, model=PLACKETT_LUCE
    )
    assert without == with_off


def test_gaussian_larger_sigma_widens_distribution() -> None:
    """Gaussian (default): a larger per-athlete sigma widens its rank spread.

    Mirrors climbing-elo's projection model, where the Glicko-2 RD is the spread.
    """
    tight = placement_pmf(ROSTER, sigmas={"b": 50.0}, n_sims=20000, seed=3, model=GAUSSIAN)
    wide = placement_pmf(ROSTER, sigmas={"b": 800.0}, n_sims=20000, seed=3, model=GAUSSIAN)
    assert rank_std(wide["b"]) > rank_std(tight["b"])


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="model must be"):
        placement_pmf(ROSTER, model="bogus")


def test_p_podium_includes_top_three() -> None:
    pmf = [0.1, 0.2, 0.3, 0.25, 0.15]
    assert math.isclose(p_podium(pmf), 0.6, abs_tol=1e-12)
    assert math.isclose(p_win(pmf), 0.1, abs_tol=1e-12)


def test_entropy_uniform_is_log_n() -> None:
    n = 5
    pmf = [1.0 / n] * n
    assert math.isclose(pmf_entropy(pmf), math.log(n), abs_tol=1e-12)


def test_entropy_certain_is_zero() -> None:
    pmf = [1.0, 0.0, 0.0]
    assert math.isclose(pmf_entropy(pmf), 0.0, abs_tol=1e-12)


def test_rank_std_certain_is_zero() -> None:
    pmf = [1.0, 0.0, 0.0]
    assert math.isclose(rank_std(pmf), 0.0, abs_tol=1e-12)


def test_clamping_out_of_range_rank() -> None:
    pmf = [0.5, 0.3, 0.2]
    # Below range clamps to rank 1, above range clamps to N.
    assert math.isclose(result_percentile(pmf, 0), pmf[0], abs_tol=1e-12)
    assert math.isclose(result_percentile(pmf, 99), 1.0, abs_tol=1e-12)
    assert math.isclose(surprisal(pmf, -5), surprisal(pmf, 1), abs_tol=1e-12)
    assert math.isclose(surprisal(pmf, 99), surprisal(pmf, 3), abs_tol=1e-12)


def test_summarize_bundles_all_keys() -> None:
    pmfs = placement_pmf(ROSTER, n_sims=5000, seed=17)
    summary = summarize(pmfs["a"], actual_rank=1)
    assert set(summary) == {
        "expected_rank_mc",
        "result_percentile",
        "surprisal",
        "p_win",
        "p_podium",
        "rank_std",
        "pmf_entropy",
    }
    pmf = pmfs["a"]
    assert math.isclose(summary["expected_rank_mc"], expected_rank_mc(pmf))
    assert math.isclose(summary["p_win"], p_win(pmf))
    assert math.isclose(summary["p_podium"], p_podium(pmf))


def test_duplicate_id_last_wins() -> None:
    roster = [("dup", 100.0), ("other", 0.0), ("dup", -500.0)]
    pmfs = placement_pmf(roster, n_sims=4000, seed=4)
    # Only one 'dup' key, reflecting the last (weak) entry: should rarely win.
    assert set(pmfs) == {"dup", "other"}
    assert p_win(pmfs["dup"]) < p_win(pmfs["other"])


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        placement_pmf(ROSTER, scale=0.0)
    with pytest.raises(ValueError, match="n_sims must be strictly positive"):
        placement_pmf(ROSTER, n_sims=0)


# ---------------------------------------------------------------------------
# Gumbel-sort domain safety (issue #59)
# ---------------------------------------------------------------------------


class _ZeroFirstRandom:
    """Stand-in RNG whose first ``random()`` returns the 0.0 endpoint.

    ``random.Random.random()`` ranges over ``[0.0, 1.0)``, so ``0.0`` is a
    possible draw. Feeding it to the Gumbel transform ``-log(-log(u))`` hits
    ``math.log(0.0)`` -> ``ValueError`` (#59). This stub forces that draw.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._first = True

    def random(self) -> float:
        if self._first:
            self._first = False
            return 0.0
        return 0.5

    def gauss(self, mu: float, sigma: float) -> float:  # pragma: no cover - unused here
        return mu


def test_placement_pmf_plackett_luce_survives_zero_uniform(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Gumbel draw must tolerate random()==0.0 instead of raising math domain error.
    monkeypatch.setattr("climber_network.elo.montecarlo.random.Random", _ZeroFirstRandom)
    pmf = placement_pmf(
        [("a", 100.0), ("b", 0.0)],
        model=PLACKETT_LUCE,
        sample_sigma=False,
        n_sims=1,
    )
    assert set(pmf) == {"a", "b"}
    for probs in pmf.values():
        assert all(math.isfinite(p) for p in probs)
        assert math.isclose(sum(probs), 1.0)
