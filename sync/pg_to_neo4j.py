"""sync.pg_to_neo4j — P1 L1 Competition mirror: climbing-elo Postgres → Neo4j.

Reads the upstream climbing-elo store (read-only) and idempotently MERGEs the
competition graph into Neo4j:

Nodes
    Athlete, Event, Round, Performance, Discipline, Rating

Edges
    (Athlete)-[:COMPETED_IN]->(Performance)-[:OF_ROUND]->(Round)-[:OF_EVENT]->(Event)
    (Event)-[:IN_DISCIPLINE]->(Discipline)
    (Athlete)-[:HAS_RATING]->(Rating)
    (Athlete)-[:FACED {count, round_ids, first_date, last_date}]->(Athlete)

FACED scope
    Materialized **only for ``final`` and ``semi`` rounds** (small fields), to
    avoid the clique explosion of qualification rounds (60+ athletes ⇒ ~3,500
    edges/round). One aggregated edge per *ordered* athlete pair, stored as two
    directed edges (a→b and b→a) for symmetric traversal.

All node ids come from ``climber_network.vocab`` builders and key on the
climbing-elo internal primary key (``ath:{id}``, ``evt:{id}``, …). Every label
and relationship type passes through ``assert_label`` / ``assert_rel`` (via the
GraphClient merge helpers) — the single injection-safety gate.

Idempotency
    Every write is a MERGE keyed on a deterministic id, so re-running the sync
    is a logical no-op (0 net changes).

Count validation
    After writing, Neo4j node/edge counts are compared against the source row
    counts. Documented filters (DNS performances skipped, non-final/semi rounds
    excluded from FACED) are subtracted by reason; any unexplained drift fails
    the sync.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

import typer

from climber_network import vocab
from climber_network.source import pg

app = typer.Typer(add_completion=False, help="L1 competition mirror: Postgres → Neo4j.")


# ---------------------------------------------------------------------------
# Structural type for the graph writer — lets tests inject a fake recorder.
# ---------------------------------------------------------------------------


class GraphWriter(Protocol):
    """Subset of GraphClient used by this sync (structural typing)."""

    def merge_node(self, label: str, node_id: str, props: dict[str, Any]) -> None: ...

    def merge_rel(
        self,
        src_id: str,
        rel_type: str,
        tgt_id: str,
        props: dict[str, Any] | None = None,
    ) -> None: ...

    def merge_nodes(self, label: str, rows: list[dict[str, Any]]) -> None: ...

    def merge_rels(self, rel_type: str, rows: list[dict[str, Any]]) -> None: ...


# ---------------------------------------------------------------------------
# Discipline vocabulary — codes mirror climbing-elo (PRD §8.1: L|B|S|BL).
# ---------------------------------------------------------------------------

DISCIPLINE_NAMES: dict[str, str] = {
    "L": "Lead",
    "B": "Boulder",
    "S": "Speed",
    "BL": "Boulder & Lead",
}

#: Round types that participate in FACED head-to-head materialization.
FACED_ROUND_TYPES: frozenset[str] = frozenset({"final", "semi"})


# ---------------------------------------------------------------------------
# Result of a sync run — counts, plus documented filters, for validation.
# ---------------------------------------------------------------------------


@dataclass
class SyncReport:
    """Tallies emitted during a sync, used for count validation and logging."""

    # Source row counts.
    src_athletes: int = 0
    src_events: int = 0
    src_rounds: int = 0
    src_results: int = 0
    src_ratings: int = 0

    # Nodes actually MERGEd into the graph (logical, dedup-aware).
    node_athletes: int = 0
    node_events: int = 0
    node_rounds: int = 0
    node_performances: int = 0
    node_ratings: int = 0
    node_disciplines: int = 0

    # Edges MERGEd.
    edge_competed_in: int = 0
    edge_of_round: int = 0
    edge_of_event: int = 0
    edge_in_discipline: int = 0
    edge_has_rating: int = 0
    edge_faced: int = 0

    # Documented filters, keyed by reason → count of dropped source rows.
    filtered: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def log(self, console: Any) -> None:
        """Print a human-readable summary of counts and applied filters."""
        console.print("[bold]L1 mirror — sync report[/bold]")
        console.print(f"  athletes:     src={self.src_athletes:>6}  nodes={self.node_athletes}")
        console.print(f"  events:       src={self.src_events:>6}  nodes={self.node_events}")
        console.print(f"  rounds:       src={self.src_rounds:>6}  nodes={self.node_rounds}")
        console.print(
            f"  results:      src={self.src_results:>6}  performances={self.node_performances}"
        )
        console.print(f"  ratings:      src={self.src_ratings:>6}  nodes={self.node_ratings}")
        console.print(f"  disciplines:  nodes={self.node_disciplines}")
        console.print(
            f"  edges: COMPETED_IN={self.edge_competed_in} OF_ROUND={self.edge_of_round} "
            f"OF_EVENT={self.edge_of_event} IN_DISCIPLINE={self.edge_in_discipline} "
            f"HAS_RATING={self.edge_has_rating} FACED={self.edge_faced}"
        )
        if self.filtered:
            console.print("  [yellow]filters applied:[/yellow]")
            for reason, n in sorted(self.filtered.items()):
                console.print(f"    - {reason}: {n}")


class CountValidationError(RuntimeError):
    """Raised when Neo4j counts diverge from source counts beyond documented filters."""


# ---------------------------------------------------------------------------
# Core sync logic — pure with respect to the (injected) writer + session.
# ---------------------------------------------------------------------------


def _iso(d: date | None) -> str | None:
    """Return an ISO date string, or None — Neo4j props stay JSON-serializable."""
    return d.isoformat() if d is not None else None


def sync_graph(writer: GraphWriter, session: pg.Session) -> SyncReport:
    """Mirror the L1 competition graph from *session* into *writer*. Idempotent.

    Returns a :class:`SyncReport` of source vs. graph counts (with documented
    filters) suitable for :func:`validate_counts`.
    """
    report = SyncReport()

    athletes = list(pg.iter_rows(session, pg.Athlete))
    events = list(pg.iter_rows(session, pg.Event))
    rounds = list(pg.iter_rows(session, pg.Round))
    results = list(pg.iter_rows(session, pg.Result))
    ratings = list(pg.iter_rows(session, pg.Rating))

    # All source rows are now fully materialized in memory. Detach them from the
    # Session (keeping their loaded values) and end the read transaction so the
    # upstream Postgres connection is NOT held idle-in-transaction during the
    # long Neo4j write phase below — Supabase terminates idle sessions, which
    # otherwise surfaces as "SSL SYSCALL error: Operation timed out" mid-run.
    session.expunge_all()
    session.rollback()

    report.src_athletes = len(athletes)
    report.src_events = len(events)
    report.src_rounds = len(rounds)
    report.src_results = len(results)
    report.src_ratings = len(ratings)

    # --- Athlete nodes -----------------------------------------------------
    athlete_rows: list[dict[str, Any]] = []
    for a in athletes:
        assert isinstance(a, pg.Athlete)
        athlete_rows.append(
            {
                "id": vocab.ath(a.id),
                "props": {
                    "name": a.name,
                    "year_of_birth": a.year_of_birth,
                    "nationality": a.nationality,
                    "gender": a.gender,
                    "photo_url": a.photo_url,
                    "height_cm": a.height_cm,
                    "weight_kg": a.weight_kg,
                    "wingspan_cm": a.wingspan_cm,
                    "retired_at": _iso(a.retired_at),
                },
            }
        )
    writer.merge_nodes("Athlete", athlete_rows)
    report.node_athletes = len(athlete_rows)

    # --- Discipline + Event nodes (and IN_DISCIPLINE edges) ----------------
    seen_disciplines: set[str] = set()
    discipline_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    in_discipline_rows: list[dict[str, Any]] = []
    for e in events:
        assert isinstance(e, pg.Event)
        code = e.discipline
        if code not in seen_disciplines:
            discipline_rows.append(
                {
                    "id": vocab.disc(code),
                    "props": {"code": code, "name": DISCIPLINE_NAMES.get(code, code)},
                }
            )
            seen_disciplines.add(code)

        event_rows.append(
            {
                "id": vocab.evt(e.id),
                "props": {
                    "name": e.name,
                    "tier": e.tier,
                    "country": e.country,
                    "season": e.season,
                    "start_date": _iso(e.start_date),
                    "discipline": code,
                },
            }
        )
        in_discipline_rows.append({"src_id": vocab.evt(e.id), "tgt_id": vocab.disc(code)})
    writer.merge_nodes("Discipline", discipline_rows)
    writer.merge_nodes("Event", event_rows)
    writer.merge_rels("IN_DISCIPLINE", in_discipline_rows)
    report.node_disciplines = len(discipline_rows)
    report.node_events = len(event_rows)
    report.edge_in_discipline = len(in_discipline_rows)

    # --- Round nodes (and OF_EVENT edges) ----------------------------------
    rounds_by_id: dict[int, pg.Round] = {}
    round_rows: list[dict[str, Any]] = []
    of_event_rows: list[dict[str, Any]] = []
    for r in rounds:
        assert isinstance(r, pg.Round)
        rounds_by_id[r.id] = r
        round_rows.append(
            {
                "id": vocab.rnd(r.id),
                "props": {
                    "round_type": r.round_type,
                    "gender": r.gender,
                    "athlete_count": r.athlete_count,
                    "event_id": r.event_id,
                },
            }
        )
        of_event_rows.append({"src_id": vocab.rnd(r.id), "tgt_id": vocab.evt(r.event_id)})
    writer.merge_nodes("Round", round_rows)
    writer.merge_rels("OF_EVENT", of_event_rows)
    report.node_rounds = len(round_rows)
    report.edge_of_event = len(of_event_rows)

    # Event start_date lookup for FACED first/last dates.
    event_date: dict[int, date | None] = {e.id: e.start_date for e in events}  # type: ignore[attr-defined]
    round_event: dict[int, int] = {r.id: r.event_id for r in rounds_by_id.values()}

    # --- Performance nodes + COMPETED_IN / OF_ROUND ------------------------
    # Group results by round for FACED aggregation; skip DNS (athlete did not
    # start) — they never competed, so no Performance / head-to-head is created.
    results_by_round: dict[int, list[pg.Result]] = defaultdict(list)
    performance_rows: list[dict[str, Any]] = []
    competed_in_rows: list[dict[str, Any]] = []
    of_round_rows: list[dict[str, Any]] = []
    for res in results:
        assert isinstance(res, pg.Result)
        if res.dns:
            report.filtered["performance_skipped_dns"] += 1
            continue
        perf_id = vocab.perf(vocab.rnd(res.round_id), vocab.ath(res.athlete_id))
        performance_rows.append(
            {
                "id": perf_id,
                "props": {
                    "rank": res.rank,
                    "score_normalized": res.score_normalized,
                    "dnf": bool(res.dnf),
                    "dns": bool(res.dns),
                },
            }
        )
        competed_in_rows.append({"src_id": vocab.ath(res.athlete_id), "tgt_id": perf_id})
        of_round_rows.append({"src_id": perf_id, "tgt_id": vocab.rnd(res.round_id)})
        results_by_round[res.round_id].append(res)

    writer.merge_nodes("Performance", performance_rows)
    writer.merge_rels("COMPETED_IN", competed_in_rows)
    writer.merge_rels("OF_ROUND", of_round_rows)
    report.node_performances = len(performance_rows)
    report.edge_competed_in = len(competed_in_rows)
    report.edge_of_round = len(of_round_rows)

    # --- Rating nodes + HAS_RATING -----------------------------------------
    rating_rows: list[dict[str, Any]] = []
    has_rating_rows: list[dict[str, Any]] = []
    for rt in ratings:
        assert isinstance(rt, pg.Rating)
        rating_id = vocab.rat(vocab.ath(rt.athlete_id), rt.discipline)
        rating_rows.append(
            {
                "id": rating_id,
                "props": {
                    "discipline": rt.discipline,
                    "mu": rt.mu,
                    "sigma": rt.sigma,
                    "n_events": rt.n_events,
                    "last_event_at": _iso(rt.last_event_at),
                    "provisional": bool(rt.provisional),
                },
            }
        )
        has_rating_rows.append({"src_id": vocab.ath(rt.athlete_id), "tgt_id": rating_id})
    writer.merge_nodes("Rating", rating_rows)
    writer.merge_rels("HAS_RATING", has_rating_rows)
    report.node_ratings = len(rating_rows)
    report.edge_has_rating = len(has_rating_rows)

    # --- FACED (final/semi only), aggregated per ordered athlete pair ------
    _emit_faced(writer, report, results_by_round, rounds_by_id, round_event, event_date)

    return report


def _emit_faced(
    writer: GraphWriter,
    report: SyncReport,
    results_by_round: dict[int, list[pg.Result]],
    rounds_by_id: dict[int, pg.Round],
    round_event: dict[int, int],
    event_date: dict[int, date | None],
) -> None:
    """MERGE aggregated FACED edges for final/semi rounds (two directed edges/pair)."""

    @dataclass
    class _Agg:
        count: int = 0
        round_ids: list[int] = field(default_factory=list)
        first_date: date | None = None
        last_date: date | None = None

    # ordered pair (a, b) → aggregate; we store both (a,b) and (b,a).
    agg: dict[tuple[int, int], _Agg] = defaultdict(_Agg)

    for round_id, round_results in results_by_round.items():
        rnd_row = rounds_by_id.get(round_id)
        if rnd_row is None or rnd_row.round_type not in FACED_ROUND_TYPES:
            continue
        evt_date = event_date.get(round_event.get(round_id, -1))
        athlete_ids = sorted({r.athlete_id for r in round_results})
        # Every ordered pair within the round faced each other once.
        for a_id, b_id in itertools.permutations(athlete_ids, 2):
            entry = agg[(a_id, b_id)]
            entry.count += 1
            entry.round_ids.append(round_id)
            if evt_date is not None:
                if entry.first_date is None or evt_date < entry.first_date:
                    entry.first_date = evt_date
                if entry.last_date is None or evt_date > entry.last_date:
                    entry.last_date = evt_date

    faced_rows: list[dict[str, Any]] = []
    for (a_id, b_id), entry in agg.items():
        faced_rows.append(
            {
                "src_id": vocab.ath(a_id),
                "tgt_id": vocab.ath(b_id),
                "props": {
                    "count": entry.count,
                    "round_ids": sorted(entry.round_ids),
                    "first_date": _iso(entry.first_date),
                    "last_date": _iso(entry.last_date),
                },
            }
        )
    writer.merge_rels("FACED", faced_rows)
    report.edge_faced = len(faced_rows)


# ---------------------------------------------------------------------------
# Count validation — Neo4j counts == source counts minus documented filters.
# ---------------------------------------------------------------------------


def validate_counts(report: SyncReport) -> None:
    """Assert graph counts equal source counts minus documented filters.

    Raises:
        CountValidationError: on any unexplained drift between source and graph.
    """
    problems: list[str] = []

    if report.node_athletes != report.src_athletes:
        problems.append(f"athletes: graph={report.node_athletes} != source={report.src_athletes}")
    if report.node_events != report.src_events:
        problems.append(f"events: graph={report.node_events} != source={report.src_events}")
    if report.node_rounds != report.src_rounds:
        problems.append(f"rounds: graph={report.node_rounds} != source={report.src_rounds}")
    if report.node_ratings != report.src_ratings:
        problems.append(f"ratings: graph={report.node_ratings} != source={report.src_ratings}")

    # Performances == results minus DNS rows (the only documented results filter).
    dns = report.filtered.get("performance_skipped_dns", 0)
    expected_perf = report.src_results - dns
    if report.node_performances != expected_perf:
        problems.append(
            f"performances: graph={report.node_performances} != "
            f"source_results({report.src_results}) - dns({dns}) = {expected_perf}"
        )

    # COMPETED_IN and OF_ROUND are 1:1 with Performance nodes.
    if report.edge_competed_in != report.node_performances:
        problems.append(
            f"COMPETED_IN: {report.edge_competed_in} != performances {report.node_performances}"
        )
    if report.edge_of_round != report.node_performances:
        problems.append(
            f"OF_ROUND: {report.edge_of_round} != performances {report.node_performances}"
        )
    # OF_EVENT is 1:1 with Round nodes; HAS_RATING with Rating nodes;
    # IN_DISCIPLINE with Event nodes.
    if report.edge_of_event != report.node_rounds:
        problems.append(f"OF_EVENT: {report.edge_of_event} != rounds {report.node_rounds}")
    if report.edge_has_rating != report.node_ratings:
        problems.append(f"HAS_RATING: {report.edge_has_rating} != ratings {report.node_ratings}")
    if report.edge_in_discipline != report.node_events:
        problems.append(
            f"IN_DISCIPLINE: {report.edge_in_discipline} != events {report.node_events}"
        )

    if problems:
        raise CountValidationError(
            "Count validation failed (unexplained drift):\n  " + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------


@app.command()
def run(
    database_url: str | None = typer.Option(
        None,
        "--database-url",
        help="Override the source connection URL (default: config.DATABASE_URL).",
    ),
) -> None:
    """Run the L1 competition mirror against the configured source database."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()
    engine = pg.make_engine(database_url)
    writer = get_client()
    with pg.read_session(engine) as session:
        report = sync_graph(writer, session)
    report.log(console)
    validate_counts(report)
    console.print("[green]Count validation passed.[/green]")


if __name__ == "__main__":
    app()
