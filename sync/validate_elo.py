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
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import typer

from climber_network import vocab
from climber_network.elo.expected import DEFAULT_SCALE, expected_finish_ranks
from climber_network.source import pg

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

#: Round-type depth ordering. A larger number is a deeper (more selective) round,
#: so the representative round per (athlete, event) is the one with the max depth.
ROUND_DEPTH: dict[str, int] = {
    "qualification": 0,
    "qual": 0,
    "semi": 1,
    "semifinal": 1,
    "final": 2,
}

#: Read query for RestednessState nodes already built by the travel sync (P3*).
#: Keyed ``rest:{ath_id}:{evt_id}``; we read the rested_index plus the
#: discipline / travel-direction breakdown dimensions for the report.
REST_QUERY = (
    "MATCH (r:RestednessState) "
    "RETURN r.athlete_id AS athlete_id, r.event_id AS event_id, "
    "r.rested_index AS rested_index, r.discipline AS discipline, "
    "r.travel_direction AS travel_direction"
)


def _round_depth(round_type: str) -> int:
    """Return the selection depth of *round_type* (unknown types sort lowest)."""
    return ROUND_DEPTH.get(round_type.lower(), -1)


# ---------------------------------------------------------------------------
# Representative round + roster assembly (read-only from the source session).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepRound:
    """A chosen representative round for one (athlete, event)."""

    athlete_id: int
    event_id: int
    round_id: int
    round_type: str
    discipline: str
    actual_rank: int
    expected_rank: float
    elo_residual: float


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

    The deepest round each athlete reached (final > semi > qualification); ties
    on depth are broken by the larger ``round_id`` (deterministic). Rounds the
    athlete did not start / has no usable rank for are not eligible.
    """
    rounds = list(pg.iter_rows(session, pg.Round))
    results = list(pg.iter_rows(session, pg.Result))
    events = list(pg.iter_rows(session, pg.Event))
    report.src_rounds = len(rounds)
    report.src_results = len(results)

    rounds_by_id: dict[int, pg.Round] = {}
    for r in rounds:
        assert isinstance(r, pg.Round)
        rounds_by_id[r.id] = r
    event_discipline: dict[int, str] = {}
    for e in events:
        assert isinstance(e, pg.Event)
        event_discipline[e.id] = e.discipline

    # Best (deepest) eligible round per (athlete, event).
    best: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    # value tuple: (depth, round_id, actual_rank, round_id_for_tiebreak) — we keep
    # actual_rank alongside so we don't re-scan results later.
    for res in results:
        assert isinstance(res, pg.Result)
        rnd_row = rounds_by_id.get(res.round_id)
        if rnd_row is None:
            report.skipped["result_round_missing"] += 1
            continue
        if res.dns or res.rank is None:
            # Did not start, or no placement → not an eligible finish.
            report.skipped["result_no_rank_or_dns"] += 1
            continue
        event_id = rnd_row.event_id
        depth = _round_depth(rnd_row.round_type)
        key = (res.athlete_id, event_id)
        candidate = (depth, res.round_id, res.rank, res.round_id)
        current = best.get(key)
        if current is None or (depth, res.round_id) > (current[0], current[3]):
            best[key] = candidate

    reps: list[RepRound] = []
    for (athlete_id, event_id), (_depth, round_id, actual_rank, _tb) in best.items():
        rnd_row = rounds_by_id[round_id]
        reps.append(
            RepRound(
                athlete_id=athlete_id,
                event_id=event_id,
                round_id=round_id,
                round_type=rnd_row.round_type,
                discipline=event_discipline.get(event_id, ""),
                actual_rank=actual_rank,
                expected_rank=math.nan,  # filled in by _compute_expected.
                elo_residual=math.nan,
            )
        )
    return reps


def _mu_before_lookup(session: pg.Session) -> dict[tuple[int, int], float]:
    """Map (athlete_id, round_id) → pre-event ``mu_before`` from rating_history.

    Point-in-time μ as of that round, read READ-ONLY from the source store. If a
    duplicate (athlete, round) appears, the last row in pk order wins (stable).
    """
    out: dict[tuple[int, int], float] = {}
    for h in pg.iter_rows(session, pg.RatingHistory):
        assert isinstance(h, pg.RatingHistory)
        out[(h.athlete_id, h.round_id)] = h.mu_before
    return out


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
# Pearson correlation — pure stdlib (no numpy / scipy).
# ---------------------------------------------------------------------------


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient of paired samples, or ``None`` if undefined.

    Returns ``None`` when fewer than two pairs are supplied or when either series
    has zero variance (the coefficient is then mathematically undefined).
    """
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = 0.0
    syy = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx = x - mean_x
        dy = y - mean_y
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    if sxx <= 0.0 or syy <= 0.0:
        return None
    return sxy / math.sqrt(sxx * syy)


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
