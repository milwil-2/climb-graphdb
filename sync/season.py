"""sync.season — Phase 4: season-level aggregates (graph read -> MERGE SeasonSummary).

Reads the representative-per-(athlete, event) outcome props already stamped on
``Performance`` nodes by the earlier outcome-variable phases, rolls them up into
one ``SeasonAggregate`` per (athlete, season, discipline) via
:mod:`climber_network.elo.season`, and MERGEs a ``SeasonSummary`` node (plus a
``HAD_SEASON`` edge from the athlete) for each. It also emits the season-driver
correlation (``over_under`` vs ``mean_rested_index``).

Read query
    One row per representative (athlete, event) — the representative being the
    ``Performance`` that carries ``elo_residual`` — joined optionally to the
    athlete's ``RestednessState`` for that event so the rested-index rollup is
    available. Only events with a known ``season`` participate.

Isolation / safety / idempotency
    Reads the graph READ-ONLY through the injected client (never imports
    ``climbing_elo`` / ``knowledge_graph``); the rollup math is self-contained
    stdlib. All writes go through the vocab-gated, batched ``merge_nodes`` /
    ``merge_rels`` (``SeasonSummary`` / ``HAD_SEASON``), keyed on the
    deterministic ``vocab.seas`` id, so re-running is a logical no-op. Nodes are
    written before edges (the ``Athlete`` already exists, so the edge MATCHes by
    :Entity id).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import typer

import climber_network.elo.season as season_mod
from climber_network import vocab

#: Read query: one row per representative (athlete, event). The representative is
#: the ``Performance`` carrying ``elo_residual``; only events with a known
#: ``season`` participate. The athlete's ``RestednessState`` for that event is
#: joined optionally so the season rested-index rollup is available when present.
#: Static string — no interpolation.
SEASON_QUERY = (
    "MATCH (a:Athlete)-[:COMPETED_IN]->(p:Performance)-[:OF_ROUND]->(:Round)"
    "-[:OF_EVENT]->(e:Event) "
    "WHERE p.elo_residual IS NOT NULL AND e.season IS NOT NULL "
    "OPTIONAL MATCH (a)-[:HAD_STATE]->(rs:RestednessState)-[:AT_EVENT]->(e) "
    "RETURN a.id AS athlete_id, e.season AS season, e.discipline AS discipline, "
    "p.elo_residual AS elo_residual, p.result_percentile AS result_percentile, "
    "p.surprisal AS surprisal, p.p_win AS p_win, p.rank_std AS rank_std, "
    "rs.rested_index AS rested_index"
)

#: Outcome props rolled up onto a SeasonSummary node (None-valued ones dropped).
_ROLLUP_PROPS = (
    "mean_elo_residual",
    "mean_result_percentile",
    "mean_surprisal",
    "season_skill",
    "season_consistency",
    "mean_rested_index",
)

app = typer.Typer(
    add_completion=False,
    help="Phase 4: season-level aggregates (SeasonSummary + HAD_SEASON).",
)


# ---------------------------------------------------------------------------
# Structural type for the graph client — lets tests inject a fake recorder.
# ---------------------------------------------------------------------------


class GraphClientLike(Protocol):
    """Subset of GraphClient used by this build (structural typing)."""

    def merge_nodes(self, label: str, rows: list[dict[str, Any]]) -> None: ...

    def merge_rels(self, rel_type: str, rows: list[dict[str, Any]]) -> None: ...

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


@dataclass
class SeasonReport:
    """Counts + the season-driver correlation emitted during a build, for logging."""

    records_read: int = 0
    season_summaries_written: int = 0
    drivers: dict[str, Any] = field(default_factory=dict)

    def log(self, console: Any) -> None:
        """Print a human-readable summary of counts + the season-driver block."""
        console.print("[bold]Phase 4 — season-level aggregates[/bold]")
        console.print(f"  records read: {self.records_read}")
        console.print(f"  season summaries written: {self.season_summaries_written}")
        drivers = self.drivers
        if drivers:
            overall = drivers.get("overall", {})
            r = overall.get("pearson_r")
            shown = "n/a" if r is None else f"{r:+.4f}"
            console.print(
                f"  drivers (over_under vs mean_rested_index): r={shown} n={overall.get('n', 0)}"
            )
            by_discipline = drivers.get("by_discipline", {})
            if by_discipline:
                console.print("    by_discipline:")
                for code, block in sorted(by_discipline.items()):
                    rk = block.get("pearson_r")
                    sk = "n/a" if rk is None else f"{rk:+.4f}"
                    console.print(f"      - {code}: r={sk} n={block.get('n', 0)}")


# ---------------------------------------------------------------------------
# Read -> records.
# ---------------------------------------------------------------------------


def _read_records(client: GraphClientLike) -> list[season_mod.PerformanceRecord]:
    """Run :data:`SEASON_QUERY` and build :class:`PerformanceRecord`s.

    Rows missing an athlete id / season / discipline are skipped (those are the
    grouping keys and cannot be defaulted). Outcome fields are passed through
    as-is (``None`` survives the rollup means).
    """
    records: list[season_mod.PerformanceRecord] = []
    for row in client.run_read(SEASON_QUERY):
        athlete_id = row.get("athlete_id")
        season = row.get("season")
        discipline = row.get("discipline")
        if athlete_id is None or season is None or discipline is None:
            continue
        records.append(
            season_mod.PerformanceRecord(
                athlete_id=str(athlete_id),
                season=int(season),
                discipline=str(discipline),
                elo_residual=_as_float(row.get("elo_residual")),
                result_percentile=_as_float(row.get("result_percentile")),
                surprisal=_as_float(row.get("surprisal")),
                p_win=_as_float(row.get("p_win")),
                rank_std=_as_float(row.get("rank_std")),
                rested_index=_as_float(row.get("rested_index")),
            )
        )
    return records


def _as_float(value: Any) -> float | None:
    """Coerce a graph value to float, preserving ``None``."""
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Aggregates -> graph writes.
# ---------------------------------------------------------------------------


def _write_season_summaries(
    client: GraphClientLike,
    aggregates: list[season_mod.SeasonAggregate],
    report: SeasonReport,
) -> None:
    """MERGE a SeasonSummary node + HAD_SEASON edge for each aggregate.

    Nodes are written first (batched ``merge_nodes``) then edges (batched
    ``merge_rels``); the ``Athlete`` already exists so the edge MATCHes by
    :Entity id. ``None``-valued rollup props are dropped. Keyed on the
    deterministic ``vocab.seas`` id, so re-running is idempotent.
    """
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for agg in aggregates:
        seas_id = vocab.seas(agg.athlete_id, agg.season, agg.discipline)
        props: dict[str, Any] = {
            "athlete_id": agg.athlete_id,
            "season": agg.season,
            "discipline": agg.discipline,
            "n_events": agg.n_events,
            "n_upsets": agg.n_upsets,
            "over_under": agg.over_under,
        }
        for name in _ROLLUP_PROPS:
            val = getattr(agg, name)
            if val is not None:
                props[name] = val
        node_rows.append({"id": seas_id, "props": props})
        edge_rows.append({"src_id": agg.athlete_id, "tgt_id": seas_id})

    client.merge_nodes("SeasonSummary", node_rows)
    client.merge_rels("HAD_SEASON", edge_rows)
    report.season_summaries_written = len(node_rows)


# ---------------------------------------------------------------------------
# Orchestration — pure with respect to the (injected) client.
# ---------------------------------------------------------------------------


def season(
    client: GraphClientLike,
    *,
    upset_threshold: float = 2.0,
) -> SeasonReport:
    """Read representative outcomes, roll them up per season, MERGE summaries.

    1. Read one row per representative (athlete, event) via :data:`SEASON_QUERY`.
    2. Aggregate into one :class:`SeasonAggregate` per (athlete, season, discipline).
    3. MERGE a ``SeasonSummary`` node + ``HAD_SEASON`` edge for each (idempotent).
    4. Compute the season-driver correlation (over_under vs mean_rested_index).
    """
    report = SeasonReport()

    records = _read_records(client)
    report.records_read = len(records)

    aggregates = season_mod.aggregate_seasons(records, upset_threshold=upset_threshold)
    _write_season_summaries(client, aggregates, report)
    report.drivers = season_mod.season_drivers_report(aggregates)
    return report


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------

_OUT_OPT = typer.Option(
    None,
    "--out",
    help="Optional path to write the structured season-driver report as JSON.",
)


@app.command()
def run(out: Path | None = _OUT_OPT) -> None:
    """Run the Phase-4 season aggregation against the configured Neo4j."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()
    client = get_client()
    report = season(client)
    report.log(console)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.drivers, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"[green]Season-driver report written to {out}.[/green]")


if __name__ == "__main__":
    app()
