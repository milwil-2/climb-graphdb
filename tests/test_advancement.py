"""Unit tests for climber_network.elo.advancement.

Pure-Python only — no database, numpy, or external service. Verifies the
multi-round progression simulator's monotonicity contract, favourite ordering,
determinism, degenerate formats, advance-count clamping, and both generative
models.
"""

from __future__ import annotations

import pytest

from climber_network.elo.advancement import (
    GAUSSIAN,
    PLACKETT_LUCE,
    ProgressionResult,
    RoundSpec,
    simulate_event_progression,
)

# A small, well-separated starting field reused across several tests.
FIELD = [
    ("a", 400.0),
    ("b", 200.0),
    ("c", 50.0),
    ("d", -100.0),
    ("e", -250.0),
    ("f", -400.0),
]

# Standard 3-round World-Cup-style format over the 6-athlete field above.
THREE_ROUNDS = [
    RoundSpec(round_type="qualification", advance_count=4),
    RoundSpec(round_type="semifinal", advance_count=2),
    RoundSpec(round_type="final", advance_count=2),
]


def _assert_monotonic(result: ProgressionResult, first_round_type: str) -> None:
    """Assert the by-construction monotonicity contract for one athlete."""
    assert result.advance_probs[first_round_type] == 1.0
    assert 1.0 >= result.p_make_final >= result.p_podium >= result.p_win >= 0.0


def test_empty_athletes_returns_empty() -> None:
    assert simulate_event_progression([], THREE_ROUNDS) == {}


def test_empty_rounds_raises() -> None:
    with pytest.raises(ValueError):
        simulate_event_progression(FIELD, [])


def test_monotonicity_all_athletes() -> None:
    results = simulate_event_progression(FIELD, THREE_ROUNDS, n_sims=4000, seed=1)
    assert set(results) == {a for a, _ in FIELD}
    for result in results.values():
        _assert_monotonic(result, "qualification")
        # advance_probs has one entry per distinct round_type.
        assert set(result.advance_probs) == {"qualification", "semifinal", "final"}
        for prob in result.advance_probs.values():
            assert 0.0 <= prob <= 1.0


def test_favourite_dominates_weak() -> None:
    results = simulate_event_progression(FIELD, THREE_ROUNDS, n_sims=8000, seed=7)
    fav = results["a"]
    weak = results["f"]
    assert fav.p_win > weak.p_win
    assert fav.p_podium > weak.p_podium
    assert fav.p_make_final > weak.p_make_final
    # Higher advance probability at each stage.
    for round_type in ("qualification", "semifinal", "final"):
        assert fav.advance_probs[round_type] >= weak.advance_probs[round_type]
    assert fav.advance_probs["semifinal"] > weak.advance_probs["semifinal"]
    assert fav.advance_probs["final"] > weak.advance_probs["final"]


def test_determinism_same_seed_identical() -> None:
    r1 = simulate_event_progression(FIELD, THREE_ROUNDS, n_sims=3000, seed=42)
    r2 = simulate_event_progression(FIELD, THREE_ROUNDS, n_sims=3000, seed=42)
    assert r1 == r2


def test_different_seed_differs() -> None:
    r1 = simulate_event_progression(FIELD, THREE_ROUNDS, n_sims=3000, seed=42)
    r2 = simulate_event_progression(FIELD, THREE_ROUNDS, n_sims=3000, seed=99)
    assert r1 != r2


def test_single_round_format_make_final_is_one() -> None:
    rounds = [RoundSpec(round_type="final", advance_count=99)]
    results = simulate_event_progression(FIELD, rounds, n_sims=5000, seed=3)
    total_win = 0.0
    total_podium = 0.0
    for result in results.values():
        assert result.p_make_final == 1.0
        assert result.advance_probs["final"] == 1.0
        _assert_monotonic(result, "final")
        total_win += result.p_win
        total_podium += result.p_podium
    # Exactly one winner and three podium slots per trial.
    assert total_win == pytest.approx(1.0, abs=1e-9)
    assert total_podium == pytest.approx(3.0, abs=1e-9)


def test_three_round_final_prob_relations() -> None:
    results = simulate_event_progression(FIELD, THREE_ROUNDS, n_sims=8000, seed=5)
    for result in results.values():
        # advance_probs["final"] is exactly P(reach final) = p_make_final.
        assert result.advance_probs["final"] == pytest.approx(result.p_make_final, abs=1e-12)
        assert result.p_podium <= result.p_make_final + 1e-12
        assert result.p_win <= result.p_make_final + 1e-12
    # Two athletes reach the final each trial -> reach-final probs sum to 2.
    total_final = sum(r.p_make_final for r in results.values())
    assert total_final == pytest.approx(2.0, abs=1e-9)
    # The final round has only 2 finalists, so podium (top-3) == make-final.
    for result in results.values():
        assert result.p_podium == pytest.approx(result.p_make_final, abs=1e-12)


def test_advance_count_larger_than_field_clamped() -> None:
    # advance_count exceeds the field: everyone advances, no crash.
    rounds = [
        RoundSpec(round_type="qualification", advance_count=100),
        RoundSpec(round_type="final", advance_count=100),
    ]
    results = simulate_event_progression(FIELD, rounds, n_sims=2000, seed=11)
    for result in results.values():
        # Everyone reaches the final since the qual cut keeps all of them.
        assert result.p_make_final == 1.0
        assert result.advance_probs["qualification"] == 1.0
        assert result.advance_probs["final"] == 1.0


@pytest.mark.parametrize("model", [GAUSSIAN, PLACKETT_LUCE])
def test_both_models_satisfy_monotonicity(model: str) -> None:
    results = simulate_event_progression(
        FIELD,
        THREE_ROUNDS,
        n_sims=4000,
        seed=21,
        model=model,
    )
    assert set(results) == {a for a, _ in FIELD}
    for result in results.values():
        _assert_monotonic(result, "qualification")


def test_duplicate_ids_last_wins() -> None:
    field = [("x", 100.0), ("y", 0.0), ("x", -500.0)]
    rounds = [RoundSpec(round_type="final", advance_count=3)]
    results = simulate_event_progression(field, rounds, n_sims=2000, seed=2)
    # Only one "x" key survives in the returned mapping.
    assert set(results) == {"x", "y"}


def test_invalid_args_raise() -> None:
    with pytest.raises(ValueError):
        simulate_event_progression(FIELD, THREE_ROUNDS, n_sims=0)
    with pytest.raises(ValueError):
        simulate_event_progression(FIELD, THREE_ROUNDS, scale=0.0)
    with pytest.raises(ValueError):
        simulate_event_progression(FIELD, THREE_ROUNDS, model="unknown")
