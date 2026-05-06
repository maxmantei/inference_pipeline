# inference-pipeline

Utilities for ML/DL threaded pipelines in inference services.

## Development

Install project dependencies:

```bash
uv sync --dev
```

Install git hooks:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

Run hooks across the repository:

```bash
uv run pre-commit run --all-files
```

Run static type checking:

```bash
uv run mypy
```

## Branching and releases

This repository follows a lightweight git-flow model:

- `main` stays stable and is the default branch.
- `dev` is the integration branch for ongoing work.
- Feature/fix work happens in short-lived branches created from `dev`.

Recommended branch name prefixes for PRs into `dev`:

- `feat/`
- `feature/`
- `fix/`
- `bugfix/`
- `hotfix/`
- `chore/`
- `docs/`
- `refactor/`
- `perf/`
- `test/`
- `ci/`
- `build/`
- `style/`
- `release/`

Release flow:

1. Merge completed work into `dev` via PR.
2. Open a PR from `dev` (or `release/*`) into `main`.
3. Bump `[project].version` in `pyproject.toml` as part of that PR.

GitHub Actions enforces CI on PRs to `dev` and `main`, validates branch naming
for PRs into `dev`, and validates version bumps for PRs into `main`.
