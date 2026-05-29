# Contributing to climb-graphdb

This repo uses **GitHub Flow**: a protected, always-green `main`, with all work
landing through short-lived branches and reviewed pull requests. These rules
keep multiple people (and their Claude Code agents) from stepping on each other.

## Golden rules

1. **Never commit or push directly to `main`.** It is branch-protected and will
   reject it. Everything lands via PR.
2. **One issue → one short-lived branch → one PR.** Merge within a day or two;
   long-lived branches are the main cause of conflicts.
3. **One checkout = one branch = one Claude Code session.** Never point two
   people (or two agents) at the same working copy/branch. Each person clones
   their own copy; use `git worktree` for parallel branches locally.
4. **Commits are solely authored by you** — no `Co-Authored-By` trailer and no
   "Generated with Claude" text. (Also satisfies the deploy commit-author check.)
5. **Don't force-push shared branches.** Force-push only *your own* feature
   branch, and only with `--force-with-lease`.

## Workflow

```bash
# 1. Start from fresh main
git switch main && git pull

# 2. Branch off, named for the issue
git switch -c feat/12-llm-extraction      # feat/ | fix/ | chore/ | docs/  + issue number

# 3. Work in small commits. Keep main merged in if it moves:
git fetch origin && git merge origin/main # (or rebase your own branch)

# 4. Run the quality gate locally before pushing (see below)

# 5. Push and open a PR
git push -u origin feat/12-llm-extraction
gh pr create --base main                  # body: "Closes #12" + the checklist

# 6. CI runs. A Claude review verifies the diff; then it auto-merges on green CI.
gh pr merge --auto --squash      # lands the moment required checks pass; branch auto-deletes
```

Link the PR to its issue with **`Closes #<n>`** so the issue auto-closes on merge.

## Quality gate (must pass; CI enforces it)

```bash
uv sync --all-extras --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run bandit -r src api sync -s B101
uv run pip-audit
uv run pytest                 # network tests are deselected by default
```

Install the pre-commit hooks so issues are caught before they're committed:

```bash
uv run pre-commit install
```

Add/extend tests with every change — see `tests/conftest.py` and use the
`/gen-test` skill to scaffold a test that matches our conventions.

## Merge policy & branch protection on `main`

**The hard gate is CI**, enforced by GitHub — you cannot bypass these via a normal push:

- **PR required** (no direct pushes to `main`).
- Required checks green: **`quality`** and **`gitleaks`**.
- All **conversations resolved**.
- Branch **up to date** with `main` before merge.
- **No force-pushes, no deletions** of `main`.

**Review is by Claude, not a required human approval** (0 approvals are
*required* by the platform). The expectation: every PR is verified with a
**Claude review — `/security-review` + `/code-review`** — from a maintainer's
Claude Code session *before* it's merged. Those reviews have already caught real
bugs CI didn't (a DOM XSS, a silent `n=0` correlation). Once the diff is
verified and CI is green, land it with GitHub **auto-merge**:

```bash
gh pr merge <n> --auto --squash    # merges automatically when checks pass
```

> Because human approval isn't *enforced*, CI is the only thing the platform
> guarantees. **Always run the Claude review before enabling auto-merge** —
> especially for external/contributor PRs. A human can still approve normally,
> and admins can `gh pr merge --admin` in a pinch.

## Working with Claude Code

- Commit the shared **`.claude/settings.json`** (agreed permissions); keep
  personal overrides in **`.claude/settings.local.json`** (gitignored).
- Let Claude keep its defaults: **branch first, never push to `main`, commit/push
  only when asked.** Don't configure auto-push.
- A maintainer's Claude session **reviews each PR** (`/security-review` +
  `/code-review`) and then **completes the merge** via auto-merge — see
  *Merge policy* above. Don't enable auto-merge on a PR you haven't reviewed.

## Security & isolation

- **Secrets only in the gitignored `.env`** (copy from `.env.example`). Never
  commit `.env`, data payloads, or caches. `gitleaks` runs in CI + pre-commit.
- **Project isolation (hard rule):** no file under `src/`, `api/`, or `sync/`
  may `import climbing_elo` or `knowledge_graph`. The upstream climbing-elo data
  is consumed **read-only via a DB connection only**. CI fails on violation.

## Dependencies

Dependabot opens weekly update PRs. If one is behind `main`, comment
**`@dependabot rebase`** (its token can update workflow files; a personal token
needs the `workflow` OAuth scope — `gh auth refresh -h github.com -s workflow`).
