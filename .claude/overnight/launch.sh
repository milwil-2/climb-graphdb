#!/usr/bin/env bash
# Launch the overnight autonomous issue runner.
#
# SAFETY: this starts a headless Claude session with --dangerously-skip-permissions
# so it never stalls on a prompt at 3am. That flag skips PROMPTS only — it does
# NOT disable the require-pr-review.sh merge gate, branch protection on main, or
# the deny list in .claude/settings.json. Those structural guardrails still hold.
#
# Recommended: watch the first issue (the warm-up) before walking away. Only
# issues labeled `auto-merge-ok` can land in main unattended; everything else is
# left as a queued PR for human review.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

log="$HOME/overnight-$(date +%F-%H%M).log"
echo "=== Overnight run starting $(date) on branch $(git branch --show-current) ===" | tee "$log"

claude --dangerously-skip-permissions \
  --append-system-prompt "You are running UNATTENDED overnight. Never wait for input; if anything blocks, log it and move on to the next issue." \
  -p "$(cat .claude/overnight/orchestrator.md)" \
  --output-format stream-json 2>&1 | tee -a "$log"

echo "=== Overnight run finished $(date) ===" | tee -a "$log"
echo "Log: $log"
