# Contributing Guidelines

We use a lightweight Trunk‑Based Development model optimized for rapid agent iteration.

## Branching Strategy

- `main` is protected and represents the latest stable, production‑ready agent.
- Create a short‑lived feature branch per experiment:
  - `git checkout -b feature/<concise-experiment-name>`
  - Example: `feature/improved-critical-heuristic`
- Merge to `main` only via Pull Request after review and passing checks.

## Commit Hygiene

- Keep commits scoped and descriptive (what/why, not just what).
- Reference related experiments or issues in commit messages and PRs.
- Avoid committing large artifacts (datasets, weights, logs). See below.

## Data and Model Artifacts

- Do NOT commit `data/`, model checkpoints, or large binary artifacts.
- Use external artifact stores and/or one of:
  - Git‑LFS for large binary files (weights, checkpoints).
  - DVC for dataset/weight versioning with cloud remotes.
- Include minimal metadata (config, seeds, model IDs) in the PR so runs are reproducible.

## Pre-Commit Checks

- Run linters/formatters if configured.
- Ensure `python -m pytest` (or targeted tests) pass locally when applicable.
- Sanity check README updates if you touch user‑facing flows.

## PR Reviews

- Keep PRs small and focused; describe experiment intent and expected outcomes.
- Link relevant dashboards (e.g., W&B) and logs instead of attaching large files.
- Address review feedback promptly; prefer follow‑ups over gigantic PRs.

## Contact

For anything, contact Satvik Golechha (7vik).
