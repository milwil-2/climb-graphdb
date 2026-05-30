"""sync.validate_elo — P3d: expected_rank / elo_residual precompute + correlation report.

For every (athlete, event) this build picks one **representative round** — the
round whose result best summarizes how far the athlete got at that event — and
stamps two derived properties onto the matching ``Performance`` node:

* ``expected_rank`` — the athlete's expected finishing position in that round,
  computed by :func:`climber_network.elo.expected.expected_finish_ranks` over a
  roster of ``(athlete_id, mu)`` pairs. ``mu`` is each athlete's **pre-event**
  rating ``rating_history.mu_before`` as of that round (point-in-time, read-only
  from the source store).
* ``elo_residual`` — ``actual_rank - expected_rank``. A *positive* residual means
  the athlete finished **worse** than the model expected (a higher rank number is
  a worse placement); a *negative* residual means they over-performed.

Representative round
    Per (athlete, event), the athlete's ``final`` round if they reached it, else
    the deepest round they reached, ordered ``final > semi > qualification``.
    This is determined from the source ``results`` / ``rounds`` (read-only, data
    only — no climbing-elo code is imported). Rounds with no usable ``mu_before``
    for the athlete (or no actual rank) are skipped and reported.

Correlation report
    Joins each representative ``Performance.elo_residual`` (graph) to the
    athlete's ``RestednessState.rested_index`` for that event (graph, keyed
    ``rest:{ath_id}:{evt_id}``) and reports the Pearson correlation overall and
    broken down by discipline and by travel direction. The **success signal** is
    a *negative* correlation: a lower rested index (more jet-lagged / depleted)
    should coincide with a *positive* residual (worse-than-expected finish).

    If no ``RestednessState`` nodes exist yet (the travel sync has not run), the
    report degrades gracefully with ``n = 0`` and a ``null`` correlation.

Isolation / safety
    Reads the upstream climbing-elo store READ-ONLY over a DB connection only;
    never imports ``climbing_elo`` / ``knowledge_graph``. All graph writes go
    through the vocab-gated ``merge_node`` (``Performance`` label); ids come from
    :mod:`climber_network.vocab` builders. Writes are MERGE-keyed and therefore
    idempotent.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import typer

from climber_network import vocab
from climber_network.elo.expected import DEFAULT_SCALE, expected_finish_ranks
from climber_network.elo.reps import RepRound
from climber_network.elo.reps import mu_before_lookup as _mu_before_lookup_impl
from climber_network.elo.reps import (
    select_representative_rounds as _select_representative_rounds_impl,
)
from climber_network.source import pg
from climber_network.stats import pearson  # re-exported for callers/tests of this module

app = typer.Typer(add_completion=False, help="P3d: expected_rank / elo_residual + correlation.")


# ---------------------------------------------------------------------------
# Structural type for the graph client — lets tests inject a fake recorder.
# ---------------------------------------------------------------------------


class GraphClientLike(Protocol):
    """Subset of GraphClient used by this build (structural typing)."""

    def merge_node(self, label: str, node_id: str, props: dict[str, Any]) -> None: ...

    def merge_nodes(self, label: str, rows: list[dict[str, Any]]) -> None: ...

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Representative-round selection — deeper rounds preferred.
# ---------------------------------------------------------------------------

# ROUND_DEPTH and RepRound are re-exported from climber_network.elo.reps so
# that validate_elo callers (and tests) that import them from here continue to
# work unchanged.

#: Read query for RestednessState nodes already built by the travel sync (P3*).
#: Keyed ``rest:{ath_id}:{evt_id}``; we read the rested_index plus the
#: discipline / travel-direction breakdown dimensions for the report.
REST_QUERY = (
    "MATCH (r:RestednessState) "
    "RETURN r.athlete_id AS athlete_id, r.event_id AS event_id, "
    "r.rested_index AS rested_index, r.discipline AS discipline, "
    "r.travel_direction AS travel_direction"
)


# ---------------------------------------------------------------------------
# Representative round + roster assembly (read-only from the source session).
# ---------------------------------------------------------------------------


@dataclass
class ValidateReport:
    """Tallies + the per-(athlete,event) representative rounds and correlations."""

    src_results: int = 0
    src_rounds: int = 0

    rep_rounds: int = 0  # representative rounds chosen
    performances_written: int = 0  # Performance nodes stamped with derived props

    # Documented skips, keyed by reason → count.
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # The chosen representative rounds (for inspection / the correlation join).
    reps: list[RepRound] = field(default_factory=list)

    # Correlation block (populated by build_correlation_report).
    correlation: dict[str, Any] = field(default_factory=dict)

    def log(self, console: Any) -> None:
        """Print a human-readable summary of counts and correlation."""
        console.print("[bold]P3d — expected_rank / elo_residual[/bold]")
        console.print(f"  source: results={self.src_results} rounds={self.src_rounds}")
        console.print(
            f"  representative rounds chosen: {self.rep_rounds} "
            f"(Performance nodes written: {self.performances_written})"
        )
        if self.skipped:
            console.print("  [yellow]skipped:[/yellow]")
            for reason, n in sorted(self.skipped.items()):
                console.print(f"    - {reason}: {n}")
        if self.correlation:
            overall = self.correlation.get("overall", {})
            r = overall.get("pearson_r")
            n = overall.get("n", 0)
            shown = "null" if r is None else f"{r:+.4f}"
            console.print(f"  correlation (rested_index vs elo_residual): r={shown} n={n}")
            for key in ("by_discipline", "by_travel_direction"):
                groups = self.correlation.get(key, {})
                if groups:
                    console.print(f"    {key}:")
                    for label, block in sorted(groups.items()):
                        gr = block.get("pearson_r")
                        gn = block.get("n", 0)
                        gshown = "null" if gr is None else f"{gr:+.4f}"
                        console.print(f"      - {label}: r={gshown} n={gn}")


def _select_representative_rounds(
    session: pg.Session,
    report: ValidateReport,
) -> list[RepRound]:
    """Pick the representative round per (athlete, event) from the source data.

    Thin wrapper around :func:`climber_network.elo.reps.select_representative_rounds`
    that propagates counts into the :class:`ValidateReport`.
    """
    rounds_out: list[int] = [0]
    results_out: list[int] = [0]
    reps = _select_representative_rounds_impl(
        session,
        report.skipped,
        src_rounds_out=rounds_out,
        src_results_out=results_out,
    )
    report.src_rounds = rounds_out[0]
    report.src_results = results_out[0]
    return reps


def _mu_before_lookup(session: pg.Session) -> dict[tuple[int, int], float]:
    """Map (athlete_id, round_id) → pre-event ``mu_before`` from rating_history.

    Thin wrapper around :func:`climber_network.elo.reps.mu_before_lookup`.
    """
    return _mu_before_lookup_impl(session)


def _compute_expected(
    reps: list[RepRound],
    mu_before: dict[tuple[int, int], float],
    report: ValidateReport,
    *,
    scale: float = DEFAULT_SCALE,
) -> list[RepRound]:
    """Fill ``expected_rank`` / ``elo_residual`` on each representative round.

    The roster for a representative round is every athlete in that *same round*
    who also has a ``mu_before`` (so the expected-rank field matches the actual
    field). Reps whose own ``mu_before`` is missing are dropped (reported).
    """
    # Group the chosen reps by their round. A round may host reps for several
    # athletes; the roster (athletes-with-mu in that round) is shared across them.
    round_reps: dict[int, list[RepRound]] = defaultdict(list)
    for rep in reps:
        round_reps[rep.round_id].append(rep)

    completed: list[RepRound] = []
    for round_id, members in round_reps.items():
        roster: list[tuple[str, float]] = []
        for rep in members:
            mu = mu_before.get((rep.athlete_id, round_id))
            if mu is not None:
                roster.append((str(rep.athlete_id), mu))
        ranks = expected_finish_ranks(roster, scale=scale) if roster else {}
        for rep in members:
            key = (rep.athlete_id, round_id)
            mu = mu_before.get(key)
            if mu is None:
                report.skipped["missing_mu_before"] += 1
                continue
            expected_rank = ranks[str(rep.athlete_id)]
            residual = float(rep.actual_rank) - expected_rank
            completed.append(
                RepRound(
                    athlete_id=rep.athlete_id,
                    event_id=rep.event_id,
                    round_id=rep.round_id,
                    round_type=rep.round_type,
                    discipline=rep.discipline,
                    actual_rank=rep.actual_rank,
                    expected_rank=expected_rank,
                    elo_residual=residual,
                )
            )
    return completed


def _write_performances(
    client: GraphClientLike,
    reps: list[RepRound],
    report: ValidateReport,
) -> None:
    """MERGE ``expected_rank`` / ``elo_residual`` onto each Performance node.

    Idempotent: keyed on the deterministic ``perf:{round_id}:{athlete_id}`` id
    via a single batched ``merge_nodes`` call (UNWIND). Props are SET, so
    existing fields on the node are preserved.
    """
    rows: list[dict[str, Any]] = [
        {
            "id": vocab.perf(vocab.rnd(rep.round_id), vocab.ath(rep.athlete_id)),
            "props": {
                "expected_rank": rep.expected_rank,
                "elo_residual": rep.elo_residual,
            },
        }
        for rep in reps
    ]
    client.merge_nodes("Performance", rows)
    report.performances_written = len(rows)


# ---------------------------------------------------------------------------
# Correlation report — ``pearson`` lives in ``climber_network.stats`` (shared).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Pair:
    """One joined (rested_index, elo_residual) sample with breakdown keys."""

    rested_index: float
    elo_residual: float
    discipline: str | None
    travel_direction: str | None


def _correlation_block(pairs: list[_Pair]) -> dict[str, Any]:
    """Build a ``{pearson_r, n}`` block from a list of joined pairs."""
    xs = [p.rested_index for p in pairs]
    ys = [p.elo_residual for p in pairs]
    return {"pearson_r": pearson(xs, ys), "n": len(pairs)}


def build_correlation_report(
    client: GraphClientLike,
    reps: list[RepRound],
) -> dict[str, Any]:
    """Join RestednessState.rested_index to representative elo_residual; correlate.

    Returns a structured dict::

        {
          "overall": {"pearson_r": float|None, "n": int},
          "by_discipline": {code: {"pearson_r": ..., "n": ...}, ...},
          "by_travel_direction": {dir: {"pearson_r": ..., "n": ...}, ...},
          "success_signal": "negative correlation expected (lower rested → worse)",
        }

    Gracefully reports ``n = 0`` / ``pearson_r = None`` when no RestednessState
    nodes exist (the travel sync has not been run yet).
    """
    rest_rows = client.run_read(REST_QUERY)

    # Map (athlete_id, event_id) → (rested_index, travel_direction) from the graph.
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
                elo_residual=rep.elo_residual,
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
        "success_signal": "negative correlation expected (lower rested → worse-than-expected)",
    }


# ---------------------------------------------------------------------------
# Orchestration — pure with respect to the (injected) client + session.
# ---------------------------------------------------------------------------


def validate_elo(
    client: GraphClientLike,
    session: pg.Session,
    *,
    scale: float = DEFAULT_SCALE,
) -> ValidateReport:
    """Precompute expected_rank / elo_residual + the correlation report. Idempotent.

    1. Choose the representative round per (athlete, event) from the source data.
    2. Build each round's roster with point-in-time ``mu_before`` and compute the
       expected rank + residual.
    3. Stamp the derived props onto the matching Performance node (MERGE).
    4. Join RestednessState.rested_index (graph) and correlate.
    """
    report = ValidateReport()

    reps = _select_representative_rounds(session, report)
    mu_before = _mu_before_lookup(session)
    completed = _compute_expected(reps, mu_before, report, scale=scale)
    report.rep_rounds = len(completed)
    report.reps = completed

    _write_performances(client, completed, report)
    report.correlation = build_correlation_report(client, completed)
    return report


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------

_OUT_OPT = typer.Option(
    None,
    "--out",
    help="Optional path to write the structured correlation report as JSON.",
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
    """Run the P3d validation against the configured source DB + Neo4j."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()
    engine = pg.make_engine(database_url)
    client = get_client()
    with pg.read_session(engine) as session:
        report = validate_elo(client, session)
    report.log(console)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.correlation, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"[green]Correlation report written to {out}.[/green]")


if __name__ == "__main__":
    app()
