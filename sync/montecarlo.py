"""sync.montecarlo — L3b: Monte-Carlo placement distribution (second outcome variable).

A *parallel* outcome variable that sits alongside the exact closed-form
``expected_rank`` / ``elo_residual`` written by :mod:`sync.validate_elo` — it never
replaces or mutates them. For each representative round this build simulates the
finishing order many times (Plackett-Luce / Gumbel-sort, optionally folding in
each athlete's rating ``sigma``) to produce a PMF over finishing positions, then
stamps **additional** distributional props onto the matching ``Performance`` node:

* ``expected_rank_mc`` — the MC mean rank (≈ the closed-form ``expected_rank``).
* ``elo_residual_mc`` — ``actual_rank - expected_rank_mc`` (the MC analogue of
  ``elo_residual``).
* ``result_percentile`` — ``P(finish <= actual_rank)`` under the PMF. The headline
  *calibrated* over/under-performance signal: low = over-performed, high = under-
  performed; bounded ``[0, 1]`` and directly calibratable. A 5th place is a much
  bigger "upset" for a near-certain winner than for a midfielder even at an
  identical raw residual — that distinction is exactly what this captures.
* ``surprisal`` — ``-log P(rank = actual_rank)`` (upset magnitude; always >= 0).
* ``p_win`` / ``p_podium`` / ``rank_std`` / ``pmf_entropy`` — skill vs consistency.

Correlation report
    Joins each representative ``result_percentile`` to the athlete's
    ``RestednessState.rested_index`` for that event and reports Pearson overall
    and by discipline / travel direction — the MC counterpart of the
    ``elo_residual`` report, so the two outcome variables can be compared head to
    head. Degrades gracefully to ``n = 0`` when no RestednessState exists yet.

Isolation / safety / idempotency
    Reads the upstream store READ-ONLY (never imports ``climbing_elo`` /
    ``knowledge_graph``); the MC math is self-contained stdlib. All randomness
    flows through a per-round seed derived from ``MC_PARAMS.seed`` so re-running is
    a logical no-op. Graph writes go through the vocab-gated, batched
    ``merge_nodes`` (``Performance`` label) keyed on ``perf:{round_id}:{athlete_id}``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import typer

import climber_network.elo.montecarlo as mc
from climber_network import vocab
from climber_network.config import MC_PARAMS, MonteCarloParams
from climber_network.elo.reps import (
    RepRound,
    mu_before_lookup,
    select_representative_rounds,
    sigma_before_lookup,
)
from climber_network.source import pg

# Shared Pearson helper + the RestednessState read query reused from the
# closed-form sync — both outcome reports share the same join.
from climber_network.stats import pearson
from sync.validate_elo import REST_QUERY

app = typer.Typer(
    add_completion=False, help="L3b: Monte-Carlo placement distribution + correlation."
)


# ---------------------------------------------------------------------------
# Structural type for the graph client — lets tests inject a fake recorder.
# ---------------------------------------------------------------------------


class GraphClientLike(Protocol):
    """Subset of GraphClient used by this build (structural typing)."""

    def merge_nodes(self, label: str, rows: list[dict[str, Any]]) -> None: ...

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Per-rep Monte-Carlo outcome.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McRep:
    """One representative round with its Monte-Carlo placement outcomes."""

    athlete_id: int
    event_id: int
    round_id: int
    round_type: str
    discipline: str | None
    actual_rank: int
    expected_rank_mc: float
    result_percentile: float
    surprisal: float
    p_win: float
    p_podium: float
    rank_std: float
    pmf_entropy: float

    @property
    def elo_residual_mc(self) -> float:
        """``actual_rank - expected_rank_mc`` — the MC analogue of elo_residual."""
        return float(self.actual_rank) - self.expected_rank_mc


@dataclass
class McReport:
    """Counts + correlation emitted during a Monte-Carlo build, for logging."""

    src_rounds: int = 0
    src_results: int = 0
    rep_rounds: int = 0
    performances_written: int = 0
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    reps: list[McRep] = field(default_factory=list)
    correlation: dict[str, Any] = field(default_factory=dict)

    def log(self, console: Any) -> None:
        """Print a human-readable summary of counts + the MC correlation block."""
        console.print("[bold]L3b — Monte-Carlo placement distribution[/bold]")
        console.print(f"  source: results={self.src_results:>6} rounds={self.src_rounds}")
        console.print(
            f"  representative rounds: {self.rep_rounds} "
            f"(Performance nodes written: {self.performances_written})"
        )
        if self.skipped:
            console.print("  [yellow]skipped:[/yellow]")
            for reason, n in sorted(self.skipped.items()):
                console.print(f"    - {reason}: {n}")
        corr = self.correlation
        if corr:
            overall = corr.get("overall", {})
            r = overall.get("pearson_r")
            shown = "n/a" if r is None else f"{r:+.4f}"
            console.print(
                f"  correlation (rested_index vs result_percentile): "
                f"r={shown} n={overall.get('n', 0)}"
            )
            for dim_key, label in (
                ("by_discipline", "by_discipline"),
                ("by_travel_direction", "by_travel_direction"),
            ):
                dim = corr.get(dim_key, {})
                if dim:
                    console.print(f"    {label}:")
                    for k, blk in sorted(dim.items()):
                        rk = blk.get("pearson_r")
                        sk = "n/a" if rk is None else f"{rk:+.4f}"
                        console.print(f"      - {k}: r={sk} n={blk.get('n', 0)}")


def _round_seed(base_seed: int, round_id: int) -> int:
    """Deterministic per-round seed so distinct rounds vary but each is reproducible."""
    return (base_seed * 1_000_003 + round_id) & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# Core computation — pure with respect to the inputs.
# ---------------------------------------------------------------------------


def compute_monte_carlo(
    reps: list[RepRound],
    mu_before: dict[tuple[int, int], float],
    sigma_before: dict[tuple[int, int], float],
    report: McReport,
    *,
    params: MonteCarloParams = MC_PARAMS,
) -> list[McRep]:
    """Simulate each representative round's PMF and derive per-rep MC outcomes.

    The roster for a round is every athlete in that *same round* who has a
    ``mu_before`` (so the simulated field matches the actual field). Reps whose
    own ``mu_before`` is missing are skipped and reported.
    """
    round_reps: dict[int, list[RepRound]] = defaultdict(list)
    for rep in reps:
        round_reps[rep.round_id].append(rep)

    out: list[McRep] = []
    for round_id, members in round_reps.items():
        roster: list[tuple[str, float]] = []
        sigmas: dict[str, float] = {}
        for rep in members:
            mu = mu_before.get((rep.athlete_id, round_id))
            if mu is None:
                continue
            aid = str(rep.athlete_id)
            roster.append((aid, mu))
            sig = sigma_before.get((rep.athlete_id, round_id))
            if sig is not None:
                sigmas[aid] = sig
        if not roster:
            continue
        pmfs = mc.placement_pmf(
            roster,
            sigmas,
            n_sims=params.n_sims,
            seed=_round_seed(params.seed, round_id),
            scale=params.scale,
            sample_sigma=params.sample_sigma,
            model=params.model,
            default_sigma=params.default_sigma,
        )
        for rep in members:
            if mu_before.get((rep.athlete_id, round_id)) is None:
                report.skipped["missing_mu_before"] += 1
                continue
            pmf = pmfs[str(rep.athlete_id)]
            summary = mc.summarize(pmf, rep.actual_rank)
            out.append(
                McRep(
                    athlete_id=rep.athlete_id,
                    event_id=rep.event_id,
                    round_id=rep.round_id,
                    round_type=rep.round_type,
                    discipline=rep.discipline,
                    actual_rank=rep.actual_rank,
                    expected_rank_mc=summary["expected_rank_mc"],
                    result_percentile=summary["result_percentile"],
                    surprisal=summary["surprisal"],
                    p_win=summary["p_win"],
                    p_podium=summary["p_podium"],
                    rank_std=summary["rank_std"],
                    pmf_entropy=summary["pmf_entropy"],
                )
            )
    return out


def _write_performances(
    client: GraphClientLike,
    reps: list[McRep],
    report: McReport,
    *,
    params: MonteCarloParams = MC_PARAMS,
) -> None:
    """MERGE the MC props onto each Performance node — additive, never touches elo_residual.

    Keyed on ``perf:{round_id}:{athlete_id}`` via a single batched ``merge_nodes``.
    Props are SET, so the closed-form ``expected_rank`` / ``elo_residual`` and the
    L1 fields on the node are preserved.
    """
    rows: list[dict[str, Any]] = [
        {
            "id": vocab.perf(vocab.rnd(rep.round_id), vocab.ath(rep.athlete_id)),
            "props": {
                "expected_rank_mc": rep.expected_rank_mc,
                "elo_residual_mc": rep.elo_residual_mc,
                "result_percentile": rep.result_percentile,
                "surprisal": rep.surprisal,
                "p_win": rep.p_win,
                "p_podium": rep.p_podium,
                "rank_std": rep.rank_std,
                "pmf_entropy": rep.pmf_entropy,
                "mc_model_version": params.model_version,
            },
        }
        for rep in reps
    ]
    client.merge_nodes("Performance", rows)
    report.performances_written = len(rows)


# ---------------------------------------------------------------------------
# Correlation — rested_index vs result_percentile (MC counterpart of the
# elo_residual report; reuses the shared pearson + REST_QUERY).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pair:
    """One joined (rested_index, result_percentile) sample with breakdown keys."""

    rested_index: float
    result_percentile: float
    discipline: str | None
    travel_direction: str | None


def _correlation_block(pairs: list[_Pair]) -> dict[str, Any]:
    """Build a ``{pearson_r, n}`` block from a list of joined pairs."""
    xs = [p.rested_index for p in pairs]
    ys = [p.result_percentile for p in pairs]
    return {"pearson_r": pearson(xs, ys), "n": len(pairs)}


def build_correlation_report(
    client: GraphClientLike,
    reps: list[McRep],
) -> dict[str, Any]:
    """Join RestednessState.rested_index to representative ``result_percentile``.

    Returns the same shape as :func:`sync.validate_elo.build_correlation_report`
    (overall / by_discipline / by_travel_direction) so the two outcome variables
    can be compared directly. Gracefully reports ``n = 0`` when no RestednessState
    nodes exist yet.
    """
    rest_rows = client.run_read(REST_QUERY)
    rested: dict[tuple[int, int], tuple[float, str | None]] = {}
    for row in rest_rows:
        athlete_id = row.get("athlete_id")
        event_id = row.get("event_id")
        rested_index = row.get("rested_index")
        if athlete_id is None or event_id is None or rested_index is None:
            continue
        rested[(int(athlete_id), int(event_id))] = (
            float(rested_index),
            row.get("travel_direction"),
        )

    pairs: list[_Pair] = []
    for rep in reps:
        match = rested.get((rep.athlete_id, rep.event_id))
        if match is None:
            continue
        rested_index, travel_direction = match
        pairs.append(
            _Pair(
                rested_index=rested_index,
                result_percentile=rep.result_percentile,
                discipline=rep.discipline or None,
                travel_direction=travel_direction,
            )
        )

    by_discipline: dict[str, list[_Pair]] = defaultdict(list)
    by_direction: dict[str, list[_Pair]] = defaultdict(list)
    for p in pairs:
        if p.discipline:
            by_discipline[p.discipline].append(p)
        if p.travel_direction:
            by_direction[p.travel_direction].append(p)

    return {
        "overall": _correlation_block(pairs),
        "by_discipline": {k: _correlation_block(v) for k, v in by_discipline.items()},
        "by_travel_direction": {k: _correlation_block(v) for k, v in by_direction.items()},
        "success_signal": (
            "negative correlation expected (lower rested → higher result_percentile / worse)"
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration — pure with respect to the (injected) client + session.
# ---------------------------------------------------------------------------


def monte_carlo(
    client: GraphClientLike,
    session: pg.Session,
    *,
    params: MonteCarloParams = MC_PARAMS,
) -> McReport:
    """Compute the MC placement outcomes + correlation report. Idempotent.

    1. Choose the representative round per (athlete, event) from the source data.
    2. Simulate each round's PMF (with point-in-time ``mu_before`` + ``sigma_before``).
    3. Stamp the additive MC props onto the matching Performance node (MERGE).
    4. Join RestednessState.rested_index (graph) and correlate against percentile.
    """
    report = McReport()

    # The out-lists are pre-sized one-element sinks: the helper assigns to [0].
    src_rounds_out = [0]
    src_results_out = [0]
    reps = select_representative_rounds(
        session,
        report.skipped,
        src_rounds_out=src_rounds_out,
        src_results_out=src_results_out,
    )
    report.src_rounds = src_rounds_out[0]
    report.src_results = src_results_out[0]

    mu_before = mu_before_lookup(session)
    sigma_before = sigma_before_lookup(session)
    mc_reps = compute_monte_carlo(reps, mu_before, sigma_before, report, params=params)
    report.rep_rounds = len(mc_reps)
    report.reps = mc_reps

    _write_performances(client, mc_reps, report, params=params)
    report.correlation = build_correlation_report(client, mc_reps)
    return report


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------

_OUT_OPT = typer.Option(
    None,
    "--out",
    help="Optional path to write the structured MC correlation report as JSON.",
)
_DB_OPT = typer.Option(
    None,
    "--database-url",
    help="Override the source connection URL (default: config.DATABASE_URL).",
)


@app.command()
def run(
    out: Path | None = _OUT_OPT,
    database_url: str | None = _DB_OPT,
) -> None:
    """Run the L3b Monte-Carlo build against the configured source DB + Neo4j."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()
    engine = pg.make_engine(database_url)
    client = get_client()
    with pg.read_session(engine) as session:
        report = monte_carlo(client, session)
    report.log(console)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.correlation, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"[green]MC correlation report written to {out}.[/green]")


if __name__ == "__main__":
    app()
