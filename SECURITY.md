# Security Policy

## Threat Model

**climb-graphdb** is a read-oriented knowledge-graph service for competitive
climbing data. The key trust boundaries are:

| Surface | Risk | Mitigation |
|---|---|---|
| Upstream Supabase DB | Credential leak / write access | Connection uses a **READ-ONLY** Postgres role with no DML or DDL privileges. The role is scoped to the `climbing-elo` DB and cannot modify, drop, or create tables. |
| Neo4j Aura | Credential leak | Credentials stored only in `.env` (gitignored) and Vercel encrypted env vars. Never committed. |
| Athlete & injury data | Privacy | Imputed training loads and injury risk scores are **private-by-default**. They are not exposed via public API endpoints and are excluded from all public exports. |
| Project isolation | Transitive dependency confusion | This project **never imports** `climbing_elo` or `knowledge_graph`. CI enforces this with a hard grep-based isolation guard on every push and pull request. |
| `/ingest` endpoint | Unauthenticated writes | Protected by `INGEST_API_KEY` bearer token. When the variable is **unset** in the cloud environment the route is disabled entirely (fail-closed). |

## Secrets Policy

- `.env` and all `.env.*` variants (except `.env.example`) are **gitignored**.
- `.env.example` contains only placeholder values — no real credentials.
- **gitleaks** runs as both a pre-commit hook and a dedicated CI job to detect
  accidentally staged secrets before they reach the remote.
- Rotate any credential immediately if you suspect it has been exposed and
  notify milanwillett@gmail.com.

## Security Scans

| Tool | When | Scope |
|---|---|---|
| `bandit -r src sync` | Pre-commit + CI | Python AST-level security issues |
| `pip-audit` | CI | Known CVEs in locked dependencies |
| `gitleaks` | Pre-commit + CI (separate job) | Secret / token patterns in git history and working tree |
| GitHub Dependabot | Weekly (automated PRs) | `pip` packages and GitHub Actions versions |

## Supported Versions

Only the latest commit on the `main` branch receives security fixes. No
backport policy exists for older commits.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email: **milanwillett@gmail.com**

Include:
- A description of the vulnerability and potential impact
- Steps to reproduce or a proof-of-concept (if safe to share)
- Any suggested remediation

You can expect an initial response within **72 hours** and a resolution
timeline within **14 days** for confirmed issues.
