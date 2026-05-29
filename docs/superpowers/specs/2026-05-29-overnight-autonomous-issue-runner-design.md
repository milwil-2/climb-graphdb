# Overnight Autonomous Issue Runner — Design

**Date:** 2026-05-29
**Status:** Approved (design); pending implementation plan
**Goal:** Run Claude Code unattended for ~8 hours, making well-planned, robustly
coded progress through the open issue list — landing low-risk work and queuing
reviewed PRs for morning human approval, without ever compromising `main`.

---

## 1. Problem & constraints

We want an overnight run that clears agent-ready issues while respecting this
repo's hard rules (see `CLAUDE.md` / `CONTRIBUTING.md`):

- **Never push to `main`** (branch-protected); never force-push.
- **Merge gate is structural**: `.claude/hooks/require-pr-review.sh` blocks
  `gh pr merge` for any PR lacking the `claude-reviewed` label. This fires on
  subagent Bash calls too, and is **not** disabled by `--dangerously-skip-permissions`
  (that flag skips *prompts* only — not hooks, not branch protection, not the
  `deny` list).
- **Isolation**: no `import climbing_elo` / `import knowledge_graph` in
  `src|api|sync`; climbing-elo is read-only via DB connection.
- **Vocab gate**: every Cypher label/rel via `assert_label` / `assert_rel`.
- **Idempotent writes**: `MERGE`, never blind `CREATE`.
- **Env access only in `config.py`**; secrets only in the gitignored `.env`.
- **CI quality gate**: ruff (format+lint), mypy, bandit, pip-audit, pytest
  ≥85% coverage, isolation + tracked-secrets guards, gitleaks.

The #1 determinant of an unattended run's success is **spec quality**: a vague
issue produces vague code with no one to course-correct at 3am. So issues are
pre-planned with explicit acceptance criteria before the run starts.

## 2. Tonight's scope (decided)

Agent-ready set, **lowest-# first**:

| # | Title | Treatment |
|---|---|---|
| #35 | chore: gitignore `.claude/worktrees/` | **Warm-up**, low-risk → tagged `auto-merge-ok`, auto-merged |
| #36 | perf(sync): batch Neo4j writes with UNWIND | Code; touches `sync/` → PR **queued** for review |
| #11 → #12 → #13 | P5 chain: news ingestion → LLM extraction → staged ingest | Code; touch `api/`+`sync/` → **stacked PRs**, queued |
| #6, #7 | L4 research: training sources / training-state inference | **Research draft doc** (no code) → PR queued |

**Excluded tonight:** #14 (needs live infra + ops judgment), #15 (vague
umbrella — must be decomposed by a human into concrete issues first).

**Resulting execution order:** the trivial #35 runs **first** as a pipeline
warm-up (a deliberate exception to strict lowest-#, to prove the auto-merge path
end-to-end before larger work), then strict lowest-# over the rest:

```
#35  →  #6  →  #7  →  #11  →  #12  →  #13  →  #36
```

## 3. Architecture

Two layers, deliberately separated:

- **Orchestrator (thin `/loop` driver).** Never writes product code, so its
  context stays small across the whole night. Selects the next issue, sets up
  the worktree, dispatches exactly one subagent, records the result, applies the
  merge policy, and moves on. Stops at queue-empty or the 8h wall-clock budget.
- **Per-issue subagent (fresh context, own git worktree).** Does *all* the
  actual work for one issue and returns a structured status. Fresh context +
  worktree isolation per issue is what prevents 8-hour compaction drift and
  cross-issue contamination.

## 4. Components

### 4.1 Issue queue + dependency gate
The agent-ready set in §2, lowest-# first (after the #35 warm-up). The P5 chain
has hard dependencies (#12 needs #11, #13 needs #12). Because hybrid policy
*queues* P5 work (no auto-merge), #11 won't be merged overnight — so the
orchestrator **stacks** #12's worktree/branch on #11's branch (and #13 on #12)
rather than blocking. Result: an ordered stack of dependent PRs to review in the
morning.

### 4.2 Orchestrator responsibilities
1. Pick next issue per ordering + dependency gate.
2. Create worktree: fresh from `main`, or stacked on predecessor for P5.
3. Dispatch one subagent with the issue's plan doc as the task; await status.
4. Comment the result on the issue; append to the run log.
5. Apply merge policy (§4.4).
6. On blocked/failed: comment + continue (§6).
7. Stop at queue-empty or 8h budget. The budget is a **soft cap checked
   between issues** — an in-flight subagent is never interrupted mid-issue; the
   orchestrator simply stops dispatching once elapsed time exceeds 8h.

### 4.3 Per-issue subagent workflow (rigid)
1. Read the plan doc + issue + `CLAUDE.md` / `CONTRIBUTING.md`.
2. **TDD**: write failing test(s) first (matching `tests/conftest.py`
   conventions via the `gen-test` skill), then implement.
3. Honor repo invariants: vocab-gated Cypher, MERGE-idempotency, config-only
   env, no sibling imports.
4. **Full local gate** (mirrors CI), with evidence:
   `ruff format --check . && ruff check . && mypy src sync api &&`
   `bandit -r src api sync -s B101 && pip-audit &&`
   `pytest --cov=src --cov-fail-under=85`
5. `verification-before-completion`: no "done" claim without passing output.
6. `/security-review` + `/code-review` (+ graph-safety invariants). If **no
   high/critical findings** → `gh pr edit <n> --add-label claude-reviewed`.
7. Push branch; open PR (or stacked PR).
8. Apply merge policy; return
   `{issue, branch, pr, gate, review, action, notes}`.
9. **Budget**: cap turns/time. If blocked or the gate fails after 2 honest
   attempts → abort, comment, return `blocked`.

**Research-doc tasks (#6, #7) carve-out:** these produce a markdown research
draft under `docs/research/`, not code. They skip TDD (step 2) and the
implementation-specific gate checks; the subagent instead writes the doc, runs
`ruff format --check`/`mypy`/`pytest` only to confirm it changed nothing in the
code tree, then opens a queued PR. Review is `/code-review` for doc quality (no
`/security-review` needed for a pure doc).

### 4.4 Hybrid merge policy
- A PR is auto-merged (`gh pr merge --auto --squash`) **only if** its issue is
  tagged `auto-merge-ok`. The `require-pr-review.sh` hook still requires the
  `claude-reviewed` label, so auto-merge always rides on a real review pass.
- Everything touching `src|api|sync` is **not** tagged `auto-merge-ok` → PR is
  left open for the user's morning review.
- Tonight: only #35 is tagged `auto-merge-ok`.

## 5. Data flow

```
queue → orchestrator picks next → load plan → dispatch subagent (worktree)
      → TDD + gate + dual review + PR → return status
      → orchestrator logs + applies merge policy → next …
morning → user reviews queued PRs (P5 stack reviewed in order)
```

## 6. Error handling / stuck protocol

- Gate fails after **2 honest attempts** → abort issue, comment failing output,
  mark blocked, continue. One bad issue never burns the night.
- Subagent crash/timeout → orchestrator catches, logs, continues.
- Dependency not ready → stack on predecessor (P5) or skip + log.
- **Structural guardrails (cannot be bypassed):** branch protection on `main`,
  `deny` rules (no force-push / no push-to-main), and the `require-pr-review.sh`
  merge gate. `--dangerously-skip-permissions` skips prompts only.
- Forbidden always: force-push, push to `main`, merge unreviewed, touch `.env`,
  import sibling projects.

## 7. Artifacts to build

1. **Plan docs** — one per agent-ready issue (acceptance criteria, files, test
   plan, risk→merge policy). Produced in a plan pass with the user.
2. **Orchestrator prompt** — committed at `.claude/overnight/orchestrator.md`.
3. **Subagent task template** — the rigid per-issue workflow (§4.3),
   at `.claude/overnight/subagent-task.md`.
4. **`auto-merge-ok` label** — created and applied to #35.
5. **Launch command/script** — `claude --dangerously-skip-permissions`, then
   `/loop` the orchestrator; output teed to `~/overnight-$(date +%F).log`.

## 8. Pre-sleep smoke test

Run on **#35 only** with the user watching ~5 minutes: confirm worktree creation
→ TDD/gate → review → label → auto-merge → cleanup → "next issue." If that full
loop works once, the unattended run is trustworthy; then let it run.

## 9. Realistic expectation

8 hours will **not** clear all 9 issues. A realistic night: #35 merged, a
reviewed PR for #36, real reviewed progress on the #11→#13 stack, and research
draft docs for #6/#7. The exact depth depends on P5 complexity. Cost: ~8h of
Opus across subagents is significant — budget accordingly.

## 10. Out of scope

- Issues #14, #15 (excluded tonight).
- Any change to branch protection, the merge gate, or the `deny` list.
- Parallel multi-issue execution (sequential only, for review clarity and to
  keep the dependency stack coherent).
