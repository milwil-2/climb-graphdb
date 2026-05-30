"""sync.advancement — L3c: Multi-round event-progression projection (Phase 3, #48).

For each event in the source data, simulate the full qualification → semifinal →
final progression (pre-event projection) and stamp per-athlete advancement
probabilities onto each athlete's *representative* ``Performance`` node.

The five additive props written are:
* ``p_make_final``       — probability the athlete reaches the final.
* ``p_podium_event``     — probability of a top-3 finish in the final.
* ``p_win_event``        — probability of winning the event.
* ``advancement_surprise`` — ``-log P(reached rep's actual deepest round)``; how
  surprising the athlete's actual advancement was given the pre-event projection.
* ``mc_model_version``   — opaque version string from :data:`MC_PARAMS`.

Isolation / safety / idempotency
    Reads the upstream store READ-ONLY (never imports ``climbing_elo`` /
    ``knowledge_graph``); all graph writes use the vocab-gated, batched
    ``merge_nodes("Performance", rows)`` — additive, never touches ``elo_residual``
    / the closed-form MC props / L1 fields. All randomness is seeded per-event
    via a deterministic helper, so re-running is a logical no-op.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import typer

import climber_network.elo.advancement as adv
from climber_network import vocab
from climber_network.config import MC_PARAMS, MonteCarloParams
from climber_network.elo.reps import (
    ROUND_DEPTH,
    RepRound,
    mu_before_lookup,
    select_representative_rounds,
    sigma_before_lookup,
)
from climber_network.source import pg

app = typer.Typer(
    add_completion=False,
    help="L3c: Multi-round event-progression projection + advancement-surprise stamping.",
)


# ---------------------------------------------------------------------------
# Structural type for the graph client — lets tests inject a fake recorder.
# ---------------------------------------------------------------------------


class GraphClientLike(Protocol):
    """Subset of GraphClient used by this build (structural typing)."""

    def merge_nodes(self, label: str, rows: list[dict[str, Any]]) -> None: ...

    def run_read(self, cypher: str, **params: Any) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Report dataclass.
# ---------------------------------------------------------------------------


@dataclass
class AdvReport:
    """Counts emitted during an advancement-progression build, for logging."""

    events_simulated: int = 0
    performances_written: int = 0
    skipped: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def log(self, console: Any) -> None:
        """Print a human-readable summary of the advancement-sync build."""
        console.print("[bold]L3c — Multi-round event-progression projection[/bold]")
        console.print(
            f"  events simulated: {self.events_simulated}  "
            f"(Performance nodes written: {self.performances_written})"
        )
        if self.skipped:
            console.print("  [yellow]skipped:[/yellow]")
            for reason, n in sorted(self.skipped.items()):
                console.print(f"    - {reason}: {n}")


def _round_seed(base_seed: int, round_id: int) -> int:
    """Deterministic per-round seed so distinct entry rounds vary but each is reproducible."""
    return (base_seed * 1_000_003 + round_id) & 0x7FFF_FFFF


# ---------------------------------------------------------------------------
# Core computation.
# ---------------------------------------------------------------------------


def _round_depth(round_type: str) -> int:
    """Return the selection depth of *round_type* (unknown types sort lowest)."""
    return ROUND_DEPTH.get(round_type.lower(), -1)


def advancement(
    client: GraphClientLike,
    session: pg.Session,
    *,
    params: MonteCarloParams = MC_PARAMS,
) -> AdvReport:
    """Simulate multi-round event progressions and stamp advancement props. Idempotent.

    For each event:

    1. Order rounds by ``(ROUND_DEPTH[round_type], round_id)`` (qual → semi → final).
    2. Build the starting field from non-DNS athletes in the entry round who have
       a ``mu_before`` in the rating history.
    3. Derive :class:`~climber_network.elo.advancement.RoundSpec` objects with
       ``advance_count`` = next round's ``athlete_count`` (last round gets 0).
    4. Call :func:`~climber_network.elo.advancement.simulate_event_progression`
       with a deterministic per-event seed.
    5. Match results to representative rounds and stamp the five additive props.

    Parameters
    ----------
    client:
        Graph client (real or fake); only :meth:`merge_nodes` is called.
    session:
        Read-only SQLAlchemy session bound to the source (climbing-elo) database.
    params:
        :class:`~climber_network.config.MonteCarloParams` instance; defaults to
        the module-level :data:`~climber_network.config.MC_PARAMS` singleton.

    Returns
    -------
    AdvReport
        Counts and skip reasons accumulated during the build.
    """
    report = AdvReport()

    # ---- Load source data ---------------------------------------------------
    rounds_all = list(pg.iter_rows(session, pg.Round))
    results_all = list(pg.iter_rows(session, pg.Result))

    mu_before = mu_before_lookup(session)
    sigma_before = sigma_before_lookup(session)

    # ---- Select representative rounds (for stamping) ------------------------
    src_rounds_out = [0]
    src_results_out = [0]
    reps = select_representative_rounds(
        session,
        report.skipped,
        src_rounds_out=src_rounds_out,
        src_results_out=src_results_out,
    )

    # Index reps by (athlete_id, event_id) for fast lookup.
    rep_by_athlete_event: dict[tuple[int, int], RepRound] = {}
    for rep in reps:
        rep_by_athlete_event[(rep.athlete_id, rep.event_id)] = rep

    # ---- Group rounds and results by event -----------------------------------
    rounds_by_event: dict[int, list[pg.Round]] = defaultdict(list)
    for rnd in rounds_all:
        assert isinstance(rnd, pg.Round)
        rounds_by_event[rnd.event_id].append(rnd)

    # Non-DNS results indexed by round_id.
    results_by_round: dict[int, list[pg.Result]] = defaultdict(list)
    for res in results_all:
        assert isinstance(res, pg.Result)
        if not res.dns and res.rank is not None:
            results_by_round[res.round_id].append(res)

    # ---- Simulate progression per event -------------------------------------
    perf_rows: list[dict[str, Any]] = []

    for event_id, event_rounds in rounds_by_event.items():
        # Order rounds: lowest depth first, then by round_id as tie-break.
        ordered = sorted(event_rounds, key=lambda r: (_round_depth(r.round_type), r.id))
        if not ordered:
            continue

        entry_round = ordered[0]
        entry_round_id = entry_round.id

        # Starting field: distinct non-DNS athletes in the entry round with mu_before.
        entry_results = results_by_round[entry_round_id]
        athletes: list[tuple[str, float]] = []
        sigmas: dict[str, float] = {}
        seen_aids: set[int] = set()
        for res in entry_results:
            assert isinstance(res, pg.Result)
            if res.athlete_id in seen_aids:
                continue
            seen_aids.add(res.athlete_id)
            mu = mu_before.get((res.athlete_id, entry_round_id))
            if mu is None:
                report.skipped["no_mu_before_entry_round"] += 1
                continue
            aid_str = str(res.athlete_id)
            athletes.append((aid_str, mu))
            sig = sigma_before.get((res.athlete_id, entry_round_id))
            if sig is not None:
                sigmas[aid_str] = sig

        if not athletes:
            report.skipped["event_empty_field"] += 1
            continue

        # Build RoundSpecs: advance_count = next round's athlete_count (last → 0).
        round_specs: list[adv.RoundSpec] = []
        for i, rnd in enumerate(ordered):
            if i + 1 < len(ordered):
                advance_count = ordered[i + 1].athlete_count
            else:
                advance_count = 0  # last round — ignored by simulator
            round_specs.append(
                adv.RoundSpec(round_type=rnd.round_type, advance_count=advance_count)
            )

        # Simulate.
        sim_results = adv.simulate_event_progression(
            athletes,
            round_specs,
            sigmas,
            n_sims=params.n_sims,
            seed=_round_seed(params.seed, entry_round_id),
            scale=params.scale,
            sample_sigma=params.sample_sigma,
            model=params.model,
            default_sigma=params.default_sigma,
        )
        report.events_simulated += 1

        # Stamp advancement props onto each representative Performance.
        for aid_str, pr in sim_results.items():
            athlete_id = int(aid_str)
            maybe_rep: RepRound | None = rep_by_athlete_event.get((athlete_id, event_id))
            if maybe_rep is None:
                # Athlete had mu in entry round but was never chosen as a rep
                # (e.g. DNS in later rounds or no deeper round reached).
                report.skipped["no_rep_round"] += 1
                continue

            # P(reached rep's actual deepest round) — from advance_probs.
            # advance_probs maps round_type → P(reached); the first round is always 1.0.
            p_reached = pr.advance_probs.get(maybe_rep.round_type)
            if p_reached is None:
                # The rep's round_type isn't in the sim results (type mismatch
                # or a round not in the ordered list); fall back to entry-round = 1.0.
                p_reached = 1.0
            surprise = -math.log(max(p_reached, 1e-12))

            perf_id = vocab.perf(vocab.rnd(maybe_rep.round_id), vocab.ath(maybe_rep.athlete_id))
            perf_rows.append(
                {
                    "id": perf_id,
                    "props": {
                        "p_make_final": pr.p_make_final,
                        "p_podium_event": pr.p_podium,
                        "p_win_event": pr.p_win,
                        "advancement_surprise": surprise,
                        "mc_model_version": params.model_version,
                    },
                }
            )

    # ---- Single batched MERGE write -----------------------------------------
    client.merge_nodes("Performance", perf_rows)
    report.performances_written = len(perf_rows)

    return report


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------

_OUT_OPT = typer.Option(
    None,
    "--out",
    help="Optional path to write a structured JSON summary of the advancement build.",
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
    """Run the L3c advancement-projection build against the configured source DB + Neo4j."""
    from rich.console import Console

    from climber_network.graph.client import get_client

    console = Console()
    engine = pg.make_engine(database_url)
    client = get_client()
    with pg.read_session(engine) as session:
        report = advancement(client, session)
    report.log(console)

    if out is not None:
        summary = {
            "events_simulated": report.events_simulated,
            "performances_written": report.performances_written,
            "skipped": dict(report.skipped),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        console.print(f"[green]Advancement build summary written to {out}.[/green]")


if __name__ == "__main__":
    app()
