# Overnight Autonomous Issue Runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the artifacts that let Claude Code run unattended overnight — a thin orchestrator that dispatches one fresh worktree subagent per issue (TDD → full gate → dual review → ship/queue), gated by your existing merge hook and branch protection.

**Architecture:** A single headless `claude -p` orchestrator session iterates the agent-ready issue queue in order. For each issue it creates a git worktree under `.claude/worktrees/<issue>`, dispatches one subagent with that issue's plan doc, records the returned status, applies the hybrid merge policy, and moves on — stopping at queue-empty or an 8h soft budget. The orchestrator writes no product code itself, so its context stays lean all night.

**Tech Stack:** Claude Code CLI (`--dangerously-skip-permissions`, headless `-p`, Agent subagents), `gh` CLI, `git worktree`, `uv`, the repo's existing review-gate hook.

**Source spec:** `docs/superpowers/specs/2026-05-29-overnight-autonomous-issue-runner-design.md`

---

## File Structure

| Path | Responsibility |
|---|---|
| `.claude/overnight/subagent-task.md` | Rigid per-issue worker prompt (code variant + research-doc variant), with `{PLACEHOLDERS}` the orchestrator substitutes per dispatch |
| `.claude/overnight/orchestrator.md` | The driver prompt: queue, selection, worktree setup, dispatch, status logging, merge policy, stop condition |
| `.claude/overnight/launch.sh` | Unattended launcher: starts headless `claude`, tees output to a timestamped log |
| `.claude/overnight/plan-template.md` | Template for the per-issue plan docs |
| `docs/agent-plans/issue-<n>.md` | One plan doc per agent-ready issue (produced in the interactive plan pass) |
| `.claude/overnight/run-state.md` | Created at runtime by the orchestrator; the night's status ledger (not committed) |

Worktrees live under `.claude/worktrees/` — which issue **#35** will gitignore, so the runner's scratch space stays out of `git status`.

---

## Task 1: Create the `auto-merge-ok` label and tag #35

**Files:** none (GitHub state only)

- [ ] **Step 1: Create the label**

```bash
gh label create auto-merge-ok \
  --color 0E8A16 \
  --description "Low-risk issue: PR may be auto-merged after Claude review (no human gate)"
```

- [ ] **Step 2: Tag the only auto-merge issue for tonight**

```bash
gh issue edit 35 --add-label auto-merge-ok
```

- [ ] **Step 3: Verify**

Run: `gh issue view 35 --json labels --jq '.labels[].name'`
Expected: output includes `auto-merge-ok`.

- [ ] **Step 4: Confirm nothing else carries it**

Run: `gh issue list --label auto-merge-ok --json number --jq '.[].number'`
Expected: `35` only.

---

## Task 2: Write the subagent task template

**Files:**
- Create: `.claude/overnight/subagent-task.md`

- [ ] **Step 1: Write the file with both variants**

```markdown
# Overnight worker — issue #{ISSUE}

You are implementing GitHub issue **#{ISSUE}** for the `climb-graphdb` repo,
working ALONE and UNATTENDED. There is no human to ask. Follow the plan at
`{PLAN_PATH}` and the rules in `CLAUDE.md` + `CONTRIBUTING.md` exactly.

**Your workspace:** `{WORKTREE}` (branch `{BRANCH}`, based on `{BASE_BRANCH}`).
Treat it as your working directory — run every git/uv command there. Do ALL
work in this worktree.

**Hard prohibitions (never, even though prompts are bypassed):**
push to `main`; force-push; edit `.env`; `import climbing_elo` /
`import knowledge_graph` anywhere under `src|api|sync`; merge a PR that lacks a
clean review.

## Workflow — CODE issue (do every step, in order)

1. Read `{PLAN_PATH}`, then `gh issue view {ISSUE}`, then `CLAUDE.md` and
   `CONTRIBUTING.md`.
2. **TDD:** write the failing test(s) first, matching `tests/conftest.py`
   conventions (reuse `source_session` / `seeded_session` / `FakeGraphClient` /
   `FakeNeo4jDriver`; the `gen-test` skill scaffolds the shape). Run them and
   confirm they FAIL for the intended reason.
3. Implement the minimal code to pass. Honor the invariants: every Cypher
   label/rel through `assert_label` / `assert_rel`; `MERGE` (never blind
   `CREATE`); env access only via `config.py`.
4. Run the FULL local gate and paste its output. ALL must pass:
   ```
   uv run ruff format --check . && uv run ruff check . \
     && uv run mypy src sync api \
     && uv run bandit -r src api sync -s B101 \
     && uv run pip-audit \
     && uv run pytest --cov=src --cov-fail-under=85
   ```
5. If the gate fails: fix and retry **once**. If it still fails, STOP — do NOT
   open a PR. Run `gh issue comment {ISSUE}` with the failing output, then
   report `status=blocked`.
6. Run `/security-review`, then `/code-review` on your diff. If there is ANY
   high/critical finding, fix and re-run. If it cannot be cleared, STOP and
   report `status=blocked` with the finding.
7. Only when the gate is green AND reviews are clean: push the branch and
   `gh pr create` (title `<type>: <summary> (closes #{ISSUE})`, body linking the
   issue). Then `gh pr edit <pr> --add-label claude-reviewed`.
8. Merge policy:
   - If `{AUTO_MERGE}` is `true`: `gh pr merge <pr> --auto --squash`.
   - Else: `gh pr comment <pr> --body "Queued for human review."` and leave it
     open.
9. Report EXACTLY this single status line and nothing after it:
   ```
   STATUS {issue:{ISSUE}, branch:{BRANCH}, pr:<#|none>, gate:<pass|fail>, review:<clean|findings|skipped>, action:<merged|queued|blocked>, notes:<one line>}
   ```

**Budget:** stay within ~{MAX_CALLS} tool calls. If you approach it, wrap up at
the nearest safe point: commit WIP to `{BRANCH}`, comment the state on the
issue, report `status=blocked`.

## Workflow — RESEARCH-DOC issue (#6, #7 only)

Same prohibitions and reporting, but:
1. Read `{PLAN_PATH}` + `gh issue view {ISSUE}` + `CLAUDE.md`.
2. Write a research draft to `docs/research/issue-{ISSUE}-<slug>.md` answering the
   issue's questions with cited sources. NO code, NO tests.
3. Confirm you changed nothing in the code tree:
   `uv run ruff format --check . && uv run mypy src sync api && uv run pytest`
   (all should pass unchanged).
4. Run `/code-review` for doc quality (skip `/security-review` — pure doc).
5. Push, `gh pr create`, `gh pr edit <pr> --add-label claude-reviewed`,
   `gh pr comment <pr> --body "Queued for human review."` (research docs are
   never auto-merged).
6. Report the same `STATUS {...}` line.
```

- [ ] **Step 2: Verify content completeness**

Run: `grep -c '{ISSUE}' .claude/overnight/subagent-task.md`
Expected: ≥ 4 (placeholders present in both variants).
Checklist: file contains the gate command, the `STATUS {...}` line, both the
CODE and RESEARCH-DOC workflows, and all five `{PLACEHOLDERS}` (`{ISSUE}`,
`{PLAN_PATH}`, `{WORKTREE}`, `{BRANCH}`, `{BASE_BRANCH}`, `{AUTO_MERGE}`,
`{MAX_CALLS}`).

---

## Task 3: Write the orchestrator prompt

**Files:**
- Create: `.claude/overnight/orchestrator.md`

- [ ] **Step 1: Write the file**

```markdown
# Overnight Orchestrator — climb-graphdb

You are the OVERNIGHT ORCHESTRATOR. You DO NOT write product code. Your only job:
drive the issue queue, dispatch ONE subagent per issue, record results, apply the
merge policy, and stop on time. Keep your own context lean — record one-line
statuses, never the subagents' work.

## Config
- QUEUE (execution order): `35, 6, 7, 11, 12, 13, 36`
- AUTO_MERGE_OK: `{35}` (only these pass `{AUTO_MERGE}=true` to the subagent)
- RESEARCH_DOC issues: `{6, 7}` (use the research-doc variant)
- DEPENDENCY CHAIN: `11 → 12 → 13` (stack each on its predecessor's branch)
- Per-issue type: `35=chore`, `6/7=docs`, `11/12/13=feat`, `36=perf`
- MAX_CALLS per subagent: `120`
- BUDGET: 8 hours, SOFT (checked only between issues)
- Subagent template: `.claude/overnight/subagent-task.md`
- Plan docs: `docs/agent-plans/issue-<n>.md`

## Setup (once, at start)
1. `start=$(date +%s)` — remember it.
2. Create `.claude/overnight/run-state.md` with a table: issue | status | pr | notes.
3. `git switch main && git pull --ff-only`.

## Loop — one iteration per issue
1. If `($(date +%s) - start) >= 28800` OR QUEUE is exhausted → go to **Finish**.
2. Pick the next not-yet-done issue in QUEUE order. Skip any marked
   `skipped-blocked`.
3. Resolve base + branch:
   - Default base = `main`; branch = `<type>/<issue>-<slug>`.
   - P5 stacking: `#12` base = the `#11` branch; `#13` base = the `#12` branch.
   - If a P5 predecessor came back `blocked`, mark this issue `skipped-blocked`,
     `gh issue comment <issue>` explaining the dependency is broken, and `continue`.
4. Create the worktree:
   `git worktree add -b <branch> .claude/worktrees/<issue> <base>`.
5. Fill the subagent template (substitute every placeholder; set
   `{AUTO_MERGE}=true` iff the issue ∈ AUTO_MERGE_OK; use the RESEARCH-DOC
   workflow iff the issue ∈ RESEARCH_DOC; `{MAX_CALLS}=120`). Dispatch ONE
   `general-purpose` subagent with that filled prompt as its task. Wait for it.
6. Parse the returned `STATUS {...}` line. Append a row to `run-state.md` and
   `gh issue comment <issue>` with the one-line result.
7. Cleanup:
   - `action=merged` → `git worktree remove .claude/worktrees/<issue> --force`.
   - `queued` / `blocked` → leave the worktree+branch in place for morning review.
8. `continue` the loop.

## Finish
- Write a final summary table to `run-state.md`.
- Post one comment to the lowest-numbered touched issue (or open a tracking
  comment) summarizing: merged / queued-for-review / blocked, with PR links.
- Print the summary and exit.

## Absolute rules
- Never push to `main`; never force-push; never merge a PR yourself (the
  subagent merges, gated by `require-pr-review.sh`); never edit `.env`.
- Never block the whole run waiting on anything. Ambiguous or stuck → comment,
  mark the issue, and move to the next.
```

- [ ] **Step 2: Verify content completeness**

Checklist: the file states the QUEUE order `35, 6, 7, 11, 12, 13, 36`, the
`28800`-second budget check, the `git worktree add` command, the stacking rule
for #12/#13, the "dispatch ONE subagent" instruction, and the absolute rules.
Run: `grep -E '28800|git worktree add|QUEUE' .claude/overnight/orchestrator.md`
Expected: all three match.

---

## Task 4: Write the unattended launcher

**Files:**
- Create: `.claude/overnight/launch.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Launch the overnight autonomous issue runner, fully unattended.
# Output is teed to a timestamped log in $HOME so you have a morning trail.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

log="$HOME/overnight-$(date +%F-%H%M).log"
echo "=== Overnight run starting $(date) on branch $(git branch --show-current) ===" | tee "$log"

# Headless, single long-running orchestrator session. --dangerously-skip-permissions
# avoids stalls at 3am; the review-gate hook + branch protection + deny list are the
# real guardrails (they are NOT disabled by this flag).
claude --dangerously-skip-permissions \
  --append-system-prompt "You are running UNATTENDED overnight. Never wait for input; if anything blocks, log it and move on to the next issue." \
  -p "$(cat .claude/overnight/orchestrator.md)" \
  --output-format stream-json 2>&1 | tee -a "$log"

echo "=== Overnight run finished $(date) ===" | tee -a "$log"
echo "Log: $log"
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x .claude/overnight/launch.sh
```

- [ ] **Step 3: Lint the script**

Run: `shellcheck .claude/overnight/launch.sh` (if installed) — else
`bash -n .claude/overnight/launch.sh`.
Expected: no errors (a clean parse / no shellcheck warnings).

---

## Task 5: Write the plan-doc template, then run the plan pass (interactive checkpoint)

**Files:**
- Create: `.claude/overnight/plan-template.md`
- Create: `docs/agent-plans/issue-35.md`, `issue-36.md`, `issue-11.md`,
  `issue-12.md`, `issue-13.md`, `issue-6.md`, `issue-7.md`

- [ ] **Step 1: Write the template**

```markdown
# Plan — issue #<n>: <title>

**Type / risk:** <chore|docs|feat|perf> / <low → auto-merge-ok | touches src|api|sync → queue>

**Acceptance criteria** (what "done" means, testable):
- [ ] <criterion 1>
- [ ] <criterion 2>

**Files to touch:**
- <path> — <why>

**Test plan:**
- <exact test(s) to write first, and the fixtures to reuse>

**Out of scope / do not touch:**
- <explicit exclusions>

**Notes for the worker:**
- <gotchas, links, dependency on a prior issue's branch>
```

- [ ] **Step 2: CHECKPOINT — conduct the plan pass with the user**

This step is interactive and MUST happen before launch. For each issue in
`35, 36, 11, 12, 13, 6, 7`, in order:
1. `gh issue view <n>` and read it aloud-in-summary to the user.
2. Fill the template: pin down acceptance criteria, files, and the first test.
3. For #11/#12/#13, record the stacking dependency in **Notes**.
4. For #6/#7, frame as a research-doc deliverable (questions to answer + where
   to look), not code.
5. Save to `docs/agent-plans/issue-<n>.md`. Confirm with the user before moving on.

Expected exit: seven plan docs exist, each user-approved, zero `<placeholder>`
angle-bracket tokens remaining.
Run: `! grep -rl '<[a-z].*>' docs/agent-plans/` (should find nothing → exit 0
means a leftover was found; invert mentally: ensure no template tokens remain).

---

## Task 6: Commit the runner artifacts

**Files:** all of `.claude/overnight/*` and `docs/agent-plans/*`

- [ ] **Step 1: Stage and commit**

```bash
git add .claude/overnight/ docs/agent-plans/
git commit -m "chore: add overnight autonomous issue runner (orchestrator, subagent template, launcher, plans)"
```

- [ ] **Step 2: Verify the tree is clean and the commit landed**

Run: `git status --porcelain && git log --oneline -1`
Expected: no unexpected untracked files; the commit is HEAD.
(Note: `.claude/worktrees/` and `.claude/overnight/run-state.md` should NOT be
committed — they are runtime scratch. If they appear, they belong in
`.gitignore` — which is exactly what issue #35 adds.)

---

## Task 7: Smoke test on #35 (interactive — watch ~5 minutes)

**Files:** none (validates the live pipeline)

- [ ] **Step 1: Dry-run the orchestrator against ONLY #35**

Temporarily run the orchestrator with QUEUE limited to `35` (either edit the
QUEUE line for the test or instruct it inline). Watch for the full chain:
worktree created under `.claude/worktrees/35/` → subagent does TDD/gate →
`/security-review` + `/code-review` → `claude-reviewed` label →
`gh pr merge --auto --squash` → worktree removed → `STATUS {... action:merged}`.

- [ ] **Step 2: Confirm the merge gate actually engaged**

Verify in the log that the subagent added the `claude-reviewed` label BEFORE the
merge, and that `require-pr-review.sh` did not block it. If the hook blocked,
the review step was skipped — fix the subagent prompt before trusting the run.
Expected: PR for #35 merged; `git worktree list` shows the #35 worktree gone.

- [ ] **Step 3: Restore the full QUEUE**

If you edited the QUEUE line for the test, restore it to
`35, 6, 7, 11, 12, 13, 36` (35 will be detected as already-done/closed and
skipped).

---

## Task 8: Hand off to the unattended run

- [ ] **Step 1: Launch and walk away**

```bash
./.claude/overnight/launch.sh
```

- [ ] **Step 2: Morning review**

Read `~/overnight-<date>.log` and `.claude/overnight/run-state.md`, then review
the queued PRs (the P5 stack in order: #11 → #12 → #13). Merge what's good;
comment on what's blocked.

---

## Self-Review

**Spec coverage (spec §-by-§):**
- §2 scope + execution order → Task 3 QUEUE (`35,6,7,11,12,13,36`); #14/#15 absent. ✓
- §3 two-layer architecture → orchestrator (Task 3) writes no code; subagent (Task 2) does the work. ✓
- §4.1 dependency stacking → Task 3 step 3 (P5 base = predecessor branch). ✓
- §4.3 rigid subagent workflow (TDD→gate→review→PR) → Task 2. ✓
- §4.4 hybrid merge policy + `auto-merge-ok` label → Task 1 + Task 2 step 8 + Task 3 config. ✓
- §5 data flow → orchestrator loop (Task 3) + launcher (Task 4). ✓
- §6 stuck protocol (2 attempts, comment+continue, structural guardrails) → Task 2 step 5, Task 3 absolute rules. ✓
- §7 artifacts (plans, orchestrator, subagent template, label, launcher) → Tasks 1–5. ✓
- §8 smoke test on #35 → Task 7. ✓
- Research-doc carve-out for #6/#7 → Task 2 RESEARCH-DOC variant + Task 5 step 2.4. ✓

**Placeholder scan:** The `{ISSUE}`/`<n>` tokens in Tasks 2/3/5 are *intentional*
template placeholders inside the artifacts, substituted at dispatch / during the
plan pass — not plan gaps. Task 5 is interactive by necessity (acceptance
criteria require user input); its template and process are fully specified.

**Consistency:** `STATUS {...}` shape, the gate command, the QUEUE order, and
the `auto-merge-ok` label name match across Tasks 1, 2, and 3.
