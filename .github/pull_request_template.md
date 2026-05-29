<!-- Keep PRs small and focused on a single issue. See CONTRIBUTING.md. -->

Closes #

## Summary

<!-- What does this change and why? One or two sentences. -->

## Changes

<!-- Bullet the notable changes. -->
-

## Testing

<!-- How was this verified? Note new/updated tests. -->
-

## Checklist

- [ ] Linked to an issue (`Closes #…`)
- [ ] Tests added/updated; `uv run pytest` green (network deselected)
- [ ] `ruff format --check` · `ruff check` · `mypy src` · `bandit -r src api sync -s B101` · `pip-audit` all green
- [ ] `/security-review` run on the diff (note any findings + how addressed)
- [ ] No secrets / `.env` / data payloads committed
- [ ] Commit solely authored (no `Co-Authored-By` / Claude attribution)
- [ ] No `climbing_elo` / `knowledge_graph` imports (isolation rule)
- [ ] Docs updated if behavior/setup changed
