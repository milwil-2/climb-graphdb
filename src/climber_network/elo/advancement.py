"""Multi-round event-progression simulator for World-Cup climbing formats.

This module is the multi-round companion to
:mod:`climber_network.elo.montecarlo`. The single-round model in that module
gives a finishing-rank PMF for one ranked round; here we simulate a *staged*
event format — qualification -> semifinal -> final, with a cut after each
non-final round — to estimate per-athlete advancement, podium, and win
probabilities.

It mirrors climbing-elo's shipped projection
(``engine.projections.simulate_event_progression``) algorithm exactly, but
re-implements it here so this repo stays fully isolated from ``climbing_elo``
(no cross-import). Only the model-name constants are shared, via
:mod:`climber_network.elo.montecarlo`.

Algorithm (one trial)
----------------------
All athletes start in round 0. For each round in order:

* Draw a performance per *currently-active* athlete (same draw conventions as
  :func:`climber_network.elo.montecarlo.placement_pmf` — gaussian Thurstonian
  or Plackett-Luce Gumbel-sort).
* If this is **not** the last round, the top ``advance_count`` performers (by
  descending performance) advance to the next round and the rest are
  eliminated; those advancers are recorded as having "reached" the next round.
* In the **final** (last) round, tally top-1 (win) and top-3 (podium) among the
  finalists.

Probabilities are tally / ``n_sims``.

Monotonicity contract
----------------------
Because the final round is reached strictly less often than (or as often as)
any earlier round, and a podium is a superset of a win among finalists, the
returned :class:`ProgressionResult` satisfies, by construction::

    1.0 >= p_make_final >= p_podium >= p_win >= 0.0

Determinism
-----------
All randomness flows through a single seeded ``random.Random(seed)``, so the
same ``(athletes, rounds, sigmas, seed, ...)`` always yields identical output —
required for the downstream sync to stay idempotent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from climber_network.elo.expected import DEFAULT_SCALE
from climber_network.elo.montecarlo import GAUSSIAN, PLACKETT_LUCE

__all__ = [
    "DEFAULT_SCALE",
    "GAUSSIAN",
    "PLACKETT_LUCE",
    "ProgressionResult",
    "RoundSpec",
    "simulate_event_progression",
]

#: Floor on a performance standard deviation, mirroring climbing-elo's SIGMA_FLOOR
#: and :data:`climber_network.elo.montecarlo._SIGMA_FLOOR`.
_SIGMA_FLOOR: float = 1e-6


@dataclass(frozen=True)
class RoundSpec:
    """Configuration for a single event round.

    Attributes
    ----------
    round_type:
        Identifier string for the round, e.g. ``"qualification"``,
        ``"semifinal"`` / ``"semi"``, or ``"final"``.
    advance_count:
        How many athletes advance from this round to the next. Ignored for the
        last round in the format (everyone remaining competes for the podium).
        Clamped to the number currently active, so a value larger than the
        field simply advances everyone.
    """

    round_type: str
    advance_count: int


@dataclass(frozen=True)
class ProgressionResult:
    """Per-athlete output from :func:`simulate_event_progression`.

    Attributes
    ----------
    athlete_id:
        The athlete identifier.
    advance_probs:
        Mapping ``round_type -> P(reached that round)``. The first round is
        always ``1.0`` (all athletes start there). If a ``round_type`` repeats
        in the format, the first occurrence wins.
    p_make_final:
        Probability the athlete reached the final (last) round. If the format
        has a single round, that round *is* the final, so this is ``1.0``.
    p_podium:
        Probability the athlete finished top-3 in the final round.
    p_win:
        Probability the athlete finished 1st in the final round.

    Monotonicity contract (by construction)::

        1.0 >= p_make_final >= p_podium >= p_win >= 0.0
    """

    athlete_id: str
    advance_probs: dict[str, float] = field(default_factory=dict)
    p_make_final: float = 1.0
    p_podium: float = 0.0
    p_win: float = 0.0


def simulate_event_progression(
    athletes: list[tuple[str, float]],
    rounds: list[RoundSpec],
    sigmas: dict[str, float] | None = None,
    *,
    n_sims: int = 20000,
    seed: int = 12345,
    scale: float = DEFAULT_SCALE,
    sample_sigma: bool = True,
    model: str = GAUSSIAN,
    default_sigma: float = 350.0,
) -> dict[str, ProgressionResult]:
    """Monte-Carlo a multi-round event format for every athlete in the field.

    Each entry of ``athletes`` is an ``(athlete_id, mu)`` tuple giving the full
    starting field (``mu`` higher = stronger). ``rounds[0]`` is the entry round
    that everyone starts in; the cut to ``rounds[i + 1]`` keeps the top
    ``rounds[i].advance_count`` performers of round ``i`` (clamped to the number
    currently active). The last round has no cut — its top-1 / top-3 finishers
    are tallied as win / podium.

    Draw conventions mirror :func:`climber_network.elo.montecarlo.placement_pmf`:

    * ``"gaussian"`` (default) — Thurstonian: ``perf_i ~ Normal(mu_i, sigma_i)``
      with ``sigma_i = max(sigmas[id] or default_sigma, _SIGMA_FLOOR)``; sorted
      descending. ``scale`` / ``sample_sigma`` are unused.
    * ``"plackett_luce"`` — Gumbel-sort: ``g_i = mu_i'/scale + Gumbel(0, 1)``
      where ``Gumbel(0, 1) = -log(-log(U))`` and ``mu_i' ~ Normal(mu_i,
      sigma_i)`` per trial when ``sample_sigma`` is set; sorted descending.

    Parameters
    ----------
    athletes:
        Full starting field as ``(athlete_id, mu)`` tuples. If an id repeats,
        the later entry wins in the returned mapping (all entries still
        compete in every trial).
    rounds:
        Ordered list of :class:`RoundSpec`, first round to last. Must be
        non-empty.
    sigmas:
        Optional ``athlete_id -> sigma`` mapping. For gaussian this is the
        performance spread (missing -> ``default_sigma``); for plackett_luce it
        is the optional per-trial rating jitter (only when ``sample_sigma``).
    n_sims:
        Number of independent trials. Must be strictly positive.
    seed:
        Seed for the single ``random.Random``. Same seed -> identical output.
    scale:
        Positive logistic temperature for plackett_luce. Unused for gaussian.
    sample_sigma:
        Plackett-Luce only: jitter ``mu_i' ~ Normal(mu_i, sigma_i)`` per trial.
    model:
        ``"gaussian"`` (default) or ``"plackett_luce"``.
    default_sigma:
        Gaussian only: performance sigma for an athlete with no (or a
        non-positive) entry in ``sigmas``.

    Returns
    -------
    dict[str, ProgressionResult]
        One :class:`ProgressionResult` per athlete, keyed by id. Empty field ->
        ``{}``.

    Raises
    ------
    ValueError
        If ``rounds`` is empty, ``n_sims`` is not strictly positive, ``scale``
        is not strictly positive, or ``model`` is unknown.
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
    if not rounds:
        msg = "rounds must contain at least one RoundSpec"
        raise ValueError(msg)

    if not athletes:
        return {}

    n = len(athletes)
    n_rounds = len(rounds)
    final_idx = n_rounds - 1

    ids = [athlete_id for athlete_id, _ in athletes]
    mus = [mu for _, mu in athletes]
    sig_map = sigmas or {}

    # Statistical Monte-Carlo sampling, not security/crypto. Seeded explicitly so
    # the downstream sync is idempotent (re-run = 0 net changes).
    rng = random.Random(seed)  # noqa: S311  # nosec B311

    # Precompute per-athlete draw parameters once.
    if model == GAUSSIAN:
        sigs = [max(sig_map.get(a) or default_sigma, _SIGMA_FLOOR) for a in ids]
        jit: list[float] = []
        any_jitter = False
    else:
        sigs = []
        jit = [s if (sample_sigma and (s := sig_map.get(a, 0.0)) and s > 0.0) else 0.0 for a in ids]
        any_jitter = any(s > 0.0 for s in jit)

    def _draw(idx: int) -> float:
        """Draw one performance for athlete at full-field index ``idx``."""
        if model == GAUSSIAN:
            return rng.gauss(mus[idx], sigs[idx])
        mu = rng.gauss(mus[idx], jit[idx]) if (any_jitter and jit[idx] > 0.0) else mus[idx]
        u = rng.random()  # in (0, 1) -> Gumbel(0, 1) = -log(-log(U))
        return mu / scale - math.log(-math.log(u))

    # reached[i][round_idx] = number of trials athlete i reached that round.
    reached: list[list[int]] = [[0] * n_rounds for _ in range(n)]
    win_counts = [0] * n
    podium_counts = [0] * n

    # All athletes start in round 0.
    for i in range(n):
        reached[i][0] = n_sims

    for _ in range(n_sims):
        active = list(range(n))  # full-field indices currently competing
        for round_idx in range(n_rounds):
            n_active = len(active)
            if n_active == 0:
                break

            perf = {idx: _draw(idx) for idx in active}
            order = sorted(active, key=lambda idx: perf[idx], reverse=True)

            if round_idx < final_idx:
                k = min(rounds[round_idx].advance_count, n_active)
                if k < 0:
                    k = 0
                advancers = order[:k]
                next_round = round_idx + 1
                for idx in advancers:
                    reached[idx][next_round] += 1
                active = advancers
            else:
                # Final round: tally win (top-1) and podium (top-3).
                win_counts[order[0]] += 1
                for idx in order[: min(3, n_active)]:
                    podium_counts[idx] += 1

    inv = 1.0 / n_sims

    # Map round_type -> round_idx, first occurrence wins (mirrors climbing-elo).
    round_idx_by_type: dict[str, int] = {}
    for round_idx in range(n_rounds):
        round_idx_by_type.setdefault(rounds[round_idx].round_type, round_idx)

    # Build results. Last-wins on duplicate ids: iterate in field order so the
    # later entry overwrites the earlier one in the returned dict.
    results: dict[str, ProgressionResult] = {}
    for i, athlete_id in enumerate(ids):
        advance_probs = {
            round_type: reached[i][round_idx] * inv
            for round_type, round_idx in round_idx_by_type.items()
        }
        # p_make_final = P(reach the last round). For a single-round format the
        # entry round IS the final, so reached[i][final_idx] == n_sims -> 1.0.
        p_make_final = reached[i][final_idx] * inv
        p_podium = podium_counts[i] * inv
        p_win = win_counts[i] * inv
        results[athlete_id] = ProgressionResult(
            athlete_id=athlete_id,
            advance_probs=advance_probs,
            p_make_final=p_make_final,
            p_podium=p_podium,
            p_win=p_win,
        )

    return results
