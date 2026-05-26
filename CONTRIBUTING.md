# Contributing

Thanks for contributing. This guide keeps the codebase consistent and reviewable.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The package uses a `src/` layout; the importable package is `recommender_systems`.

## Code style

- Formatting and linting are handled by [Ruff](https://docs.astral.sh/ruff/). Run
  `ruff format` and `ruff check` before committing; CI enforces both.
- Line length is 100 characters. Target Python 3.10+.
- Type-hint all public functions and methods.
- Public functions and classes get NumPy-style docstrings (Parameters / Returns).
  Don't document the obvious — comment *why*, not *what*.
- Prefer small, pure functions and explicit arguments over hidden state. No code should
  run at import time.

## Tests

- Every algorithm and bug fix ships with tests under `tests/`.
- Use small, in-memory fixtures (a handful of users/items) so tests stay fast and
  deterministic.
- Run the suite with `pytest`.

## Commits and pull requests

- Write [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`,
  `refactor:`, `test:`, `docs:`, `chore:`. Imperative mood, no trailing period.
- Keep each PR to one logical change. Smaller PRs review faster and merge sooner.
- Branch names: `feat/<slug>`, `fix/<slug>`, `refactor/<slug>`, `chore/<slug>`.
- PR descriptions state the motivation and link the issue they close.
- A PR is mergeable when CI is green, the diff is focused, and review threads are resolved.
