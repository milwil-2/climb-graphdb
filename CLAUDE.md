# CLAUDE.md — climb-graphdb

A self-contained **Neo4j knowledge graph** fusing geography, travel / circadian
load, imputed training, and competition results for **World-Cup climbing**.

> **Agents: start here.** This file auto-loads every session. Before opening or
> merging a PR, read **[`CONTRIBUTING.md`](./CONTRIBUTING.md)** (full workflow +
> merge policy). Security model is in **[`SECURITY.md`](./SECURITY.md)**.

## Working here — the merge gate (must follow)

- **Sync `main`, then branch** — always start from latest to avoid `BEHIND`/conflict churn:
  `git switch main && git pull`, then `git switch -c feat|fix|chore|docs/<issue#>-slug`.
  **Never push to `main`** (branch-protected).
- Open a PR; CI checks **`quality`** + **`gitleaks`** are the enforced hard gate.
- **Before merging, review the PR yourself**: run **`/security-review`** + **`/code-review`**, confirm no high/critical findings, then label it:
  `gh pr edit <n> --add-label claude-reviewed`.
  A committed `PreToolUse` hook (`.claude/hooks/require-pr-review.sh`) **blocks `gh pr merge` for any PR without that label** — so this isn't optional.
- Land it with `gh pr merge <n> --auto --squash`. Rationale + details in `CONTRIBUTING.md`.

## Isolation constraint (HARD RULE)

This repo is **fully isolated** from the sibling projects. No file under
`src/`, `api/`, or `sync/` may `import climbing_elo` or `import knowledge_graph`.
The climbing-elo data is consumed **READ-ONLY via a database connection only**
(`DATABASE_URL` → a read-only Supabase role) — never by importing that codebase.
CI enforces this:

```
! grep -rnE "^(from|import) (climbing_elo|knowledge_graph)" src api sync
```

## Commands

```bash
uv sync --all-extras --dev                  # install (toolchain in [dependency-groups] dev)
uv run pytest                               # tests (network-marked tests deselected by default)
uv run ruff format . && uv run ruff check . # format + lint
uv run mypy src                             # type check
uv run bandit -r src api sync -s B101       # security scan (B101 asserts skipped)
uv run pip-audit                            # dependency CVE audit
uv run uvicorn api.index:app --reload       # run the API locally
```

## Conventions

- **Closed vocabulary** lives in `src/climber_network/vocab.py`. Every Cypher
  label / relationship-type interpolation MUST pass through `assert_label` /
  `assert_rel` (the single injection-safety gate). Node ids come from the
  builder functions in the same module.
- **Idempotent writes**: always `MERGE` (never blind `CREATE`) so re-running an
  ingest is a no-op.
- **Constants & env access only in `config.py`** — nothing else reads
  `os.environ` directly. Import the getter functions or `TRAVEL_PARAMS`.
- **Tests with every change** — reuse `tests/conftest.py` fixtures; the
  `/gen-test` skill scaffolds a test matching repo conventions.

## Shared Claude Code config

- **`.claude/settings.json`** (committed) — shared permissions + the review-gate
  hook. Personal overrides go in the gitignored `.claude/settings.local.json`.
- **`.mcp.json`** (committed) — a **read-only** Neo4j MCP server
  (`mcp-neo4j-cypher` via `uvx`, `NEO4J_READ_ONLY=true`); creds resolve from env
  (`${NEO4J_URI}`/`${NEO4J_USER}`/`${NEO4J_PASSWORD}`) — **not committed**. Enabled
  plugins: `code-review`, `claude-md-management`.

## CRITICAL gotchas

1. **certifi / macOS TLS** — macOS framework Python ships without a CA bundle,
   so `neo4j+s://` (Aura) fails with `CERTIFICATE_VERIFY_FAILED`. Both
   `graph/client.py` and `api/db.py` set `SSL_CERT_FILE` to `certifi.where()`
   *before* the neo4j driver is used. Keep this pattern in any new DB module.
2. **Aura username is the INSTANCE ID**, not the literal string `"neo4j"`.
   Always read `NEO4J_USER` from the environment.
3. **Commits solely authored by Milan Willett** — no `Co-Authored-By` trailer,
   no "Generated with Claude" text anywhere (also satisfies `COMMIT_AUTHOR_REQUIRED`).
   Collaborators author their own commits under their own identity.
4. **Secrets only in the gitignored `.env`** (copy from `.env.example`). Never
   commit `.env`, data payloads, or caches.
5. **Proprietary planning docs are local-only** — `PRD.md`, `PLAN.md`, and
   `IMPLEMENTATION_PLAN.md` are gitignored and intentionally NOT in the repo.
   Don't expect to find them here, and never commit them.
