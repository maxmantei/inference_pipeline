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
