"""sync.analyze — L3b Phase 2: MC calibration + travel-weight fitting (read-only).

Two **read-only** analyses over the graph the Monte-Carlo (`sync.montecarlo`) and
travel (`sync.travel`) builds already populated. They write NOTHING to the graph
— each prints a report and can optionally dump it as JSON.

``calibrate``
    Probability-integral-transform (PIT) calibration of the Monte-Carlo placement
    model. From each representative ``Performance``'s stored ``result_percentile``
    (= P(finish ≤ actual)) and ``surprisal`` (= -log P(finish = actual)) we form
    the **randomized PIT** — using ``P(finish = actual) = exp(-surprisal)`` — and
    test uniformity (KS distance + ECE). A well-calibrated model ⇒ PIT ≈
    Uniform[0,1] (mean ≈ 0.5, KS/ECE ≈ 0). Lets us compare model variants.

``fit-weights``
    Grid-searches the L3 travel weights ``w1``/``w2`` to find the pair whose
    recomputed ``rested_index`` is **most negatively** correlated with an
    underperformance outcome (``elo_residual`` or ``result_percentile``) — a
    principled, data-driven alternative to the literature priors
    (``config.TravelParams.w1=0.7, w2=0.3``). It REPORTS recommended weights and
    the full grid curve; it does **NOT** mutate config.

Isolation / safety
    Reads the climber-network graph only (no source DB, no ``climbing_elo`` /
    ``knowledge_graph`` import). The Cypher is static (no label/rel interpolation);
    the one parameter, ``$outcome``, is an allow-listed property name.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

import typer

from climber_network.elo import calibration as cal
from climber_network.elo.weightfit import WeightSample, fit_weights

app = typer.Typer(add_completion=False, help="L3b Phase 2: MC calibration + travel-weight fitting.")


class GraphClientLike(Protocol):
    """Subset of GraphClient used by these analyses (read-only)."""

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Read queries (static — no label/rel interpolation).
# ---------------------------------------------------------------------------

#: Representative Performances carrying the MC outcome props, with discipline.
CALIBRATION_QUERY = (
    "MATCH (p:Performance) "
    "WHERE p.result_percentile IS NOT NULL AND p.surprisal IS NOT NULL "
    "OPTIONAL MATCH (p)-[:OF_ROUND]->(:Round)-[:OF_EVENT]->(e:Event) "
    "RETURN p.result_percentile AS result_percentile, p.surprisal AS surprisal, "
    "e.discipline AS discipline"
)

#: RestednessState components joined to the representative Performance outcome.
#: ``$outcome`` is an allow-listed property name (dynamic access ``p[$outcome]``).
WEIGHTFIT_QUERY = (
    "MATCH (a:Athlete)-[:HAD_STATE]->(rs:RestednessState)-[:AT_EVENT]->(e:Event) "
    "WHERE rs.jetlag_residual IS NOT NULL AND rs.travel_fatigue IS NOT NULL "
    "MATCH (a)-[:COMPETED_IN]->(p:Performance)-[:OF_ROUND]->(:Round)-[:OF_EVENT]->(e) "
    "WHERE p[$outcome] IS NOT NULL "
    "RETURN rs.jetlag_residual AS jetlag_residual, rs.travel_fatigue AS travel_fatigue, "
    "p[$outcome] AS outcome, rs.discipline AS discipline, rs.travel_direction AS travel_direction"
)

#: Allow-listed underperformance outcome fields for fit-weights (both: higher = worse).
OUTCOME_FIELDS: tuple[str, ...] = ("elo_residual", "result_percentile")
_DEFAULT_PIT_SEED = 12345


# ---------------------------------------------------------------------------
# Report builders — pure with respect to the injected client.
# ---------------------------------------------------------------------------


def build_calibration_report(
    client: GraphClientLike,
    *,
    seed: int = _DEFAULT_PIT_SEED,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute the randomized-PIT calibration report, overall + by discipline.

    Deterministic given ``seed`` (the randomized PIT draws one uniform per row).
    """
    rows = client.run_read(CALIBRATION_QUERY)
    rng = random.Random(seed)  # noqa: S311  # nosec B311 - PIT jitter, not security
    pits: list[float] = []
    by_discipline: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        rp = row.get("result_percentile")
        s = row.get("surprisal")
        if rp is None or s is None:
            continue
        point_mass = cal.point_mass_from_surprisal(float(s))
        pit = cal.randomized_pit(float(rp), point_mass, rng.random())
        pits.append(pit)
        discipline = row.get("discipline")
        if discipline:
            by_discipline[discipline].append(pit)
    return {
        "overall": cal.calibration_report(pits, n_bins=n_bins),
        "by_discipline": {
            code: cal.calibration_report(vals, n_bins=n_bins)
            for code, vals in sorted(by_discipline.items())
        },
    }


def build_weightfit_report(
    client: GraphClientLike,
    *,
    outcome: str = "elo_residual",
    grid_steps: int = 101,
) -> dict[str, Any]:
    """Grid-fit the travel weights against *outcome*; reports recommended weights.

    *outcome* must be one of :data:`OUTCOME_FIELDS` (both orient higher = worse).
    """
    if outcome not in OUTCOME_FIELDS:
        msg = f"outcome must be one of {OUTCOME_FIELDS}, got {outcome!r}"
        raise ValueError(msg)
    rows = client.run_read(WEIGHTFIT_QUERY, outcome=outcome)
    samples: list[WeightSample] = []
    for row in rows:
        jr = row.get("jetlag_residual")
        tf = row.get("travel_fatigue")
        out = row.get("outcome")
        if jr is None or tf is None or out is None:
            continue
        samples.append(
            WeightSample(
                jetlag_residual=float(jr),
                travel_fatigue=float(tf),
                outcome=float(out),
                discipline=row.get("discipline"),
                travel_direction=row.get("travel_direction"),
            )
        )
    report = fit_weights(samples, grid_steps=grid_steps)
    report["outcome_field"] = outcome
    return report


# ---------------------------------------------------------------------------
# Logging.
# ---------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = "+.4f") -> str:
    return "n/a" if value is None else format(value, spec)


def _log_calibration(report: dict[str, Any], console: Any) -> None:
    console.print("[bold]L3b — Monte-Carlo PIT calibration[/bold]")
    overall = report.get("overall", {})
    console.print(
        f"  overall: n={overall.get('n', 0)} "
        f"mean={_fmt(overall.get('mean'), '.4f')} "
        f"ks={_fmt(overall.get('ks'), '.4f')} ece={_fmt(overall.get('ece'), '.4f')} "
        "(calibrated ⇒ mean≈0.5, ks/ece≈0)"
    )
    by_disc = report.get("by_discipline", {})
    if by_disc:
        console.print("  by_discipline:")
        for code, blk in by_disc.items():
            console.print(
                f"    - {code}: n={blk.get('n', 0)} mean={_fmt(blk.get('mean'), '.4f')} "
                f"ks={_fmt(blk.get('ks'), '.4f')} ece={_fmt(blk.get('ece'), '.4f')}"
            )


def _log_weightfit(report: dict[str, Any], console: Any) -> None:
    console.print("[bold]L3b — travel-weight fit[/bold]")
    console.print(f"  outcome: {report.get('outcome_field')}  n={report.get('n', 0)}")
    best = report.get("best", {})
    current = report.get("current", {})
    console.print(
        f"  recommended: w1={_fmt(best.get('w1'), '.2f')} w2={_fmt(best.get('w2'), '.2f')} "
        f"(pearson {_fmt(best.get('pearson'))})"
    )
    console.print(
        f"  current prior: w1={_fmt(current.get('w1'), '.2f')} w2={_fmt(current.get('w2'), '.2f')} "
        f"(pearson {_fmt(current.get('pearson'))})  — config is NOT modified"
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

_OUT_OPT = typer.Option(None, "--out", help="Optional path to write the report as JSON.")
_SEED_OPT = typer.Option(_DEFAULT_PIT_SEED, "--seed", help="RNG seed for the randomized PIT.")
_OUTCOME_OPT = typer.Option(
    "elo_residual", "--outcome", help="Underperformance outcome: elo_residual | result_percentile."
)


def _maybe_write(out: Path | None, report: dict[str, Any], console: Any) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"[green]Report written to {out}.[/green]")


@app.command()
def calibrate(out: Path | None = _OUT_OPT, seed: int = _SEED_OPT) -> None:
    """PIT-calibration of the Monte-Carlo placement model (read-only)."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()
    report = build_calibration_report(get_client(), seed=seed)
    _log_calibration(report, console)
    _maybe_write(out, report, console)


@app.command("fit-weights")
def fit_weights_cmd(out: Path | None = _OUT_OPT, outcome: str = _OUTCOME_OPT) -> None:
    """Grid-fit the L3 travel weights against an underperformance outcome (read-only)."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()
    report = build_weightfit_report(get_client(), outcome=outcome)
    _log_weightfit(report, console)
    _maybe_write(out, report, console)


if __name__ == "__main__":
    app()
