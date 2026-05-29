# Overnight worker — issue #{ISSUE}

You are implementing GitHub issue **#{ISSUE}** for the `climb-graphdb` repo,
working ALONE and UNATTENDED. There is no human to ask. Follow the plan at
`{PLAN_PATH}` and the rules in `CLAUDE.md` + `CONTRIBUTING.md` exactly.

**Your workspace:** `{WORKTREE}` (branch `{BRANCH}`, based on `{BASE_BRANCH}`).
Treat it as your working directory — run every git/uv command there.

**Hard prohibitions (never, even with prompts bypassed):** push to `main`;
force-push; edit `.env`; `import climbing_elo` / `import knowledge_graph` under
`src|api|sync`; merge a PR lacking a clean review.

## Workflow — CODE issue (every step, in order)
1. Read `{PLAN_PATH}`, `gh issue view {ISSUE}`, `CLAUDE.md`, `CONTRIBUTING.md`.
2. **TDD:** write the failing test(s) first, reusing `tests/conftest.py` fixtures
   (`source_session` / `seeded_session` / `FakeGraphClient` / `FakeNeo4jDriver`;
   `gen-test` skill scaffolds the shape). Run; confirm they FAIL correctly.
3. Implement the minimal code. Invariants: Cypher labels/rels via
   `assert_label`/`assert_rel`; `MERGE` not `CREATE`; env only via `config.py`.
4. Run the FULL gate and paste output — ALL must pass:
   `uv run ruff format --check . && uv run ruff check . && uv run mypy src sync api && uv run bandit -r src api sync -s B101 && uv run pip-audit && uv run pytest --cov=src --cov-fail-under=85`
5. Gate fails → fix and retry ONCE. Still failing → STOP, do NOT open a PR;
   `gh issue comment {ISSUE}` with the failing output; report `status=blocked`.
6. `/security-review` then `/code-review`. Any high/critical → fix + re-run;
   can't clear → STOP, report `status=blocked` with the finding.
7. Gate green AND reviews clean → push; `gh pr create`; then
   `gh pr edit <pr> --add-label claude-reviewed`.
8. Merge policy: `{AUTO_MERGE}`==true → `gh pr merge <pr> --auto --squash`;
   else `gh pr comment <pr> --body "Queued for human review."` and leave open.
9. Report EXACTLY one line, nothing after:
   `STATUS {issue:{ISSUE}, branch:{BRANCH}, pr:<#|none>, gate:<pass|fail>, review:<clean|findings|skipped>, action:<merged|queued|blocked>, notes:<one line>}`

**Budget:** ~{MAX_CALLS} tool calls. Near the limit → commit WIP, comment the
state on the issue, report `status=blocked`.

## Workflow — RESEARCH-DOC issue (#6, #7)
Same prohibitions/reporting, but:
1. Read `{PLAN_PATH}` + `gh issue view {ISSUE}` + `CLAUDE.md`.
2. Write `docs/research/issue-{ISSUE}-<slug>.md` answering the issue's questions
   with **cited, fetched** sources. NO code/tests. Flag every unverifiable claim
   `**VERIFY:**`. Do NOT modify `SECURITY.md` — propose its notes inside the doc.
3. Confirm nothing changed in the code tree:
   `uv run ruff format --check . && uv run mypy src sync api && uv run pytest`.
4. `/code-review` for doc quality (skip `/security-review` — pure doc).
5. Push; `gh pr create`; `gh pr edit <pr> --add-label claude-reviewed`;
   `gh pr comment <pr> --body "Queued for human review."` (research never auto-merges).
6. Report the same `STATUS {...}` line.
