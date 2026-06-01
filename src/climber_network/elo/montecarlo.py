"""Monte-Carlo placement distributions from pairwise strength ratings.

This module is the distributional companion to
:func:`climber_network.elo.expected.expected_finish_ranks`. The closed form
gives each athlete's *expected* finishing rank exactly (by linearity of
expectation), but it cannot express the *shape* of the placement distribution —
the probability of winning, of a podium, the spread, the entropy, or how
surprising an actual result was. We recover all of that by simulation.

Generative model
-----------------
Two models are supported, selected by ``model``:

* ``"gaussian"`` (default) — a **Thurstonian** order-statistics model that
  mirrors climbing-elo's shipped projection
  (``engine.projections.compute_podium_probabilities``): each trial draws a
  performance ``perf_i ~ Normal(mu_i, sigma_i)`` and sorts by **descending**
  score, where ``sigma_i`` is the athlete's Glicko-2 rating deviation (φ, on the
  display scale). Spread — and therefore who upsets whom — is governed by
  ``sigma_i``. This is the production model, so the resulting ``p_win`` /
  ``p_podium`` are directly comparable to climbing-elo's predictions.

* ``"plackett_luce"`` — a Gumbel-sort realisation of Plackett-Luce:
  ``g_i = mu_i' / scale + Gumbel(0, 1)`` (``Gumbel(0, 1) = -log(-log(U))``),
  sorted descending. Its pairwise marginal that ``j`` beats ``i`` is the logistic
  ``logistic((mu_j - mu_i) / scale)`` — the *same* link ``expected.py`` uses — so
  its Monte-Carlo mean rank converges to the closed-form expected rank (a
  convergence test pins this down). This is the bridge to the closed-form
  ``expected_rank`` rather than the production placement model.

Why two: climbing-elo computes *ratings* with a Bradley-Terry / Plackett-Luce
logistic likelihood (matched by ``expected.py`` + the plackett_luce model) but
computes *placement predictions* with the gaussian model above. We keep the
closed-form ``expected_rank`` on the logistic link and default the MC placement
distribution to the gaussian model, so each piece matches its climbing-elo
counterpart.

Rating uncertainty
-------------------
The gaussian model is inherently uncertainty-driven (``sigma_i`` is the spread;
a missing sigma falls back to ``default_sigma``). For plackett_luce, setting
``sample_sigma`` jitters ``mu_i' ~ Normal(mu_i, sigma_i)`` per trial before the
Gumbel noise, widening the distribution for poorly-observed athletes.

Determinism
-----------
All randomness flows through a single seeded ``random.Random(seed)``, so the
same ``(roster, sigmas, seed, ...)`` always yields byte-identical output. This
is required for the downstream sync to stay idempotent. Callers should pass a
distinct, stable per-round seed (e.g. derived from a round id) so different
rounds get independent draws while each round stays reproducible.
"""

from __future__ import annotations

import math
import random

from climber_network.elo.expected import DEFAULT_SCALE

__all__ = [
    "DEFAULT_SCALE",
    "GAUSSIAN",
    "PLACKETT_LUCE",
    "expected_rank_mc",
    "p_podium",
    "p_win",
    "placement_pmf",
    "pmf_entropy",
    "rank_std",
    "result_percentile",
    "summarize",
    "surprisal",
]

# Floor for log of a probability, so surprisal/entropy never hit -inf.
_LOG_FLOOR: float = 1e-12
# Floor on a performance standard deviation, mirroring climbing-elo's SIGMA_FLOOR.
_SIGMA_FLOOR: float = 1e-6

#: Supported generative models for :func:`placement_pmf`.
GAUSSIAN: str = "gaussian"
PLACKETT_LUCE: str = "plackett_luce"


def _gumbel_noise(rng: random.Random) -> float:
    """A standard ``Gumbel(0, 1)`` draw, ``-log(-log(U))`` with ``U`` in ``(0, 1)``.

    ``random.Random.random()`` returns a float in ``[0.0, 1.0)``, so it can yield
    exactly ``0.0`` — which sends the inner ``math.log(0.0)`` to a domain error and
    crashes the Gumbel-sort simulation (issue #59). We resample the ``0.0``
    endpoint (unbiased, and in practice never loops) so the result is always
    finite. Shared by :func:`placement_pmf` and
    :func:`climber_network.elo.advancement.simulate_event_progression` so the
    guard lives in exactly one place.
    """
    u = rng.random()
    while u <= 0.0:
        u = rng.random()
    return -math.log(-math.log(u))


def placement_pmf(
    roster: list[tuple[str, float]],
    sigmas: dict[str, float] | None = None,
    *,
    n_sims: int = 20000,
    seed: int = 12345,
    scale: float = DEFAULT_SCALE,
    sample_sigma: bool = True,
    model: str = GAUSSIAN,
    default_sigma: float = 350.0,
) -> dict[str, list[float]]:
    """Monte-Carlo finishing-rank PMF for every athlete in ``roster``.

    Each entry of ``roster`` is an ``(athlete_id, mu)`` tuple where ``mu`` is the
    athlete's strength rating (higher is stronger). We run ``n_sims`` simulated
    orderings and tally how often each athlete lands in each finishing rank
    (rank 1 = winner).

    Two generative models are supported (see the module docstring):

    * ``"gaussian"`` (default) — **Thurstonian**: each trial draws a performance
      ``perf_i ~ Normal(mu_i, sigma_i)`` and sorts by descending score. This
      mirrors climbing-elo's shipped projection model
      (``engine.projections.compute_podium_probabilities``), where ``sigma_i`` is
      the Glicko-2 rating deviation. Spread is driven by ``sigma_i``; ``scale``
      and ``sample_sigma`` are unused.
    * ``"plackett_luce"`` — Gumbel-sort: ``g_i = mu_i'/scale + Gumbel(0, 1)``,
      sorted descending. Its pairwise marginal is the logistic
      ``logistic((mu_j - mu_i)/scale)`` — the *same* link as ``expected.py`` — so
      its mean rank converges to the closed-form expected rank (a test pins
      this). When ``sample_sigma`` is set, ``mu_i' ~ Normal(mu_i, sigma_i)``.

    Parameters
    ----------
    roster:
        List of ``(athlete_id, mu)`` tuples. If an id repeats, the later entry
        wins in the returned mapping (matching
        :func:`climber_network.elo.expected.expected_finish_ranks`), though all
        entries still compete in every trial.
    sigmas:
        Optional mapping ``athlete_id -> sigma`` (rating deviation). For the
        gaussian model this is the performance spread (missing entries fall back
        to ``default_sigma``); for plackett_luce it is the optional rating jitter
        (only consulted when ``sample_sigma`` is true).
    n_sims:
        Number of simulated orderings. Larger ``n_sims`` reduces sampling noise.
    seed:
        Seed for the single ``random.Random``. Same seed -> identical output.
        Pass a per-round seed for cross-round variety with reproducibility.
    scale:
        Positive logistic temperature for the plackett_luce model; defaults to
        ``expected.DEFAULT_SCALE`` (= ``config.GLICKO2_SCALE``). Unused for gaussian.
    sample_sigma:
        Plackett-Luce only: if true, jitter ``mu_i' ~ Normal(mu_i, sigma_i)``
        per trial before adding Gumbel noise.
    model:
        ``"gaussian"`` (default) or ``"plackett_luce"``.
    default_sigma:
        Gaussian only: performance sigma used for an athlete with no (or a
        non-positive) entry in ``sigmas``, so an unrated competitor still has a
        non-degenerate distribution.

    Returns
    -------
    dict[str, list[float]]
        Mapping from ``athlete_id`` to a PMF: a list of length ``N`` where index
        ``r - 1`` holds ``P(finish in rank r)``. Each list sums to ``1.0``.
        Special cases: empty roster -> ``{}``; single athlete -> ``{id: [1.0]}``.

    Raises
    ------
    ValueError
        If ``scale``/``n_sims`` is not strictly positive or ``model`` is unknown.
    """
    if scale <= 0.0:
        msg = f"scale must be strictly positive, got {scale!r}"
        raise ValueError(msg)
    if n_sims <= 0:
        msg = f"n_sims must be strictly positive, got {n_sims!r}"
        raise ValueError(msg)
    if model not in (GAUSSIAN, PLACKETT_LUCE):
        msg = f"model must be {GAUSSIAN!r} or {PLACKETT_LUCE!r}, got {model!r}"
        raise ValueError(msg)

    if not roster:
        return {}

    n = len(roster)

    # Single athlete: always rank 1, no randomness needed.
    if n == 1:
        return {roster[0][0]: [1.0]}

    ids = [athlete_id for athlete_id, _ in roster]
    mus = [mu for _, mu in roster]
    sig_map = sigmas or {}
    # Statistical Monte-Carlo sampling, not security/crypto. Seeded explicitly so
    # the downstream sync is idempotent (re-run = 0 net changes).
    rng = random.Random(seed)  # noqa: S311  # nosec B311
    indices = list(range(n))

    # counts[i][r] = number of trials athlete at index i finished in rank r+1.
    counts: list[list[int]] = [[0] * n for _ in range(n)]

    if model == GAUSSIAN:
        # Thurstonian: perf_i ~ N(mu_i, sigma_i); sigma drives the spread.
        sigs = [max(sig_map.get(a) or default_sigma, _SIGMA_FLOOR) for a in ids]
        for _ in range(n_sims):
            perf = [rng.gauss(mus[i], sigs[i]) for i in range(n)]
            order = sorted(indices, key=lambda i: perf[i], reverse=True)
            for rank_idx, athlete_idx in enumerate(order):
                counts[athlete_idx][rank_idx] += 1
    else:
        # Plackett-Luce via the Gumbel-sort trick.
        jit = [s if (sample_sigma and (s := sig_map.get(a, 0.0)) and s > 0.0) else 0.0 for a in ids]
        any_jitter = any(s > 0.0 for s in jit)
        for _ in range(n_sims):
            gumbels: list[float] = []
            for i in range(n):
                mu = rng.gauss(mus[i], jit[i]) if (any_jitter and jit[i] > 0.0) else mus[i]
                gumbels.append(mu / scale + _gumbel_noise(rng))
            order = sorted(indices, key=lambda i: gumbels[i], reverse=True)
            for rank_idx, athlete_idx in enumerate(order):
                counts[athlete_idx][rank_idx] += 1

    inv = 1.0 / n_sims
    # Last-wins on duplicate ids: iterate in roster order so later overwrites.
    return {athlete_id: [c * inv for c in counts[i]] for i, athlete_id in enumerate(ids)}


def _clamp_rank(actual_rank: int, n: int) -> int:
    """Clamp a 1-indexed rank into ``[1, n]``."""
    if actual_rank < 1:
        return 1
    if actual_rank > n:
        return n
    return actual_rank


def expected_rank_mc(pmf: list[float]) -> float:
    """Mean finishing rank ``sum_r r * pmf[r - 1]`` for one athlete's PMF."""
    return sum(rank * p for rank, p in enumerate(pmf, start=1))


def result_percentile(pmf: list[float], actual_rank: int) -> float:
    """CDF of the placement PMF evaluated at ``actual_rank``.

    Returns ``sum_{r <= actual_rank} pmf[r - 1]`` — the probability of finishing
    at least as well as ``actual_rank``. Low values mean the athlete
    overperformed relative to expectation; high values mean they underperformed.
    Always in ``[0, 1]``. ``actual_rank`` is clamped into ``[1, N]``.
    """
    n = len(pmf)
    if n == 0:
        return 0.0
    r = _clamp_rank(actual_rank, n)
    return sum(pmf[:r])


def surprisal(pmf: list[float], actual_rank: int) -> float:
    """Information content ``-log(P(actual_rank))`` in nats.

    Large for unlikely results, ``0`` for a certain one. The probability is
    floored at ``1e-12`` so the value is finite. ``actual_rank`` is clamped into
    ``[1, N]``.
    """
    n = len(pmf)
    if n == 0:
        return -math.log(_LOG_FLOOR)
    r = _clamp_rank(actual_rank, n)
    return -math.log(max(pmf[r - 1], _LOG_FLOOR))


def p_win(pmf: list[float]) -> float:
    """Probability of finishing first (``pmf[0]``)."""
    return pmf[0] if pmf else 0.0


def p_podium(pmf: list[float]) -> float:
    """Probability of a top-3 finish (``sum(pmf[:3])``)."""
    return sum(pmf[:3])


def rank_std(pmf: list[float]) -> float:
    """Standard deviation of the finishing rank under the PMF."""
    mean = expected_rank_mc(pmf)
    var = sum(p * (rank - mean) ** 2 for rank, p in enumerate(pmf, start=1))
    return math.sqrt(max(var, 0.0))


def pmf_entropy(pmf: list[float]) -> float:
    """Shannon entropy of the PMF in nats (natural log).

    Zero-probability ranks contribute nothing (``0 * log 0 := 0``).
    """
    total = 0.0
    for p in pmf:
        if p > 0.0:
            total -= p * math.log(p)
    return total


def summarize(pmf: list[float], actual_rank: int) -> dict[str, float]:
    """Bundle every distributional statistic for one athlete's result.

    Returns a dict with keys ``expected_rank_mc``, ``result_percentile``,
    ``surprisal``, ``p_win``, ``p_podium``, ``rank_std`` and ``pmf_entropy``.
    ``actual_rank`` is clamped into ``[1, N]`` by the percentile/surprisal
    helpers.
    """
    return {
        "expected_rank_mc": expected_rank_mc(pmf),
        "result_percentile": result_percentile(pmf, actual_rank),
        "surprisal": surprisal(pmf, actual_rank),
        "p_win": p_win(pmf),
        "p_podium": p_podium(pmf),
        "rank_std": rank_std(pmf),
        "pmf_entropy": pmf_entropy(pmf),
    }
