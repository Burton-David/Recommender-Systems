# Automation protocol

This repository is developed by two collaborating maintainers who coordinate **entirely
through GitHub** — pull requests, reviews, comments, and issues. This document is the
contract they both follow. The quality bar lives in [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Roles

- **Developer** — implements roadmap items and issues, opens PRs against `main`, responds
  to review feedback, and files issues for bugs and gaps discovered while working.
- **Reviewer** — reviews open PRs, approves and merges sound work, requests changes
  otherwise, and also picks up open issues to implement.

Both roles develop and both roles review. Neither merges their own pull request, so every
change is seen by the other maintainer before it lands.

## Branches and commits

- Never commit directly to `main`. All changes flow through PRs.
- Branch names: `feat/<slug>`, `fix/<slug>`, `refactor/<slug>`, `chore/<slug>`,
  `docs/<slug>`, `test/<slug>`.
- Commits follow Conventional Commits (see `CONTRIBUTING.md`).

## Pull requests

- One logical change per PR. Keep diffs small and focused.
- The description states motivation and links the issue it closes (`Closes #123`).
- A PR is ready to merge when: CI is green, the change meets the style bar, and all review
  threads are resolved.
- Merge with squash and delete the branch. Only merge PRs authored by the *other*
  maintainer.

## Reviews

- Leave specific, actionable comments. Block only on real defects, failing tests/CI, or
  clear style violations — not preferences or nitpicks.
- Converge: after about three rounds on one PR, stop blocking. Either approve with a
  follow-up note or open a tracking issue, then merge.

## Issues

Issues are the shared backlog. Labels:

- Type: `type:feat`, `type:bug`, `type:chore`, `type:docs`
- Priority: `priority:high`, `priority:medium`, `priority:low`
- Status: `status:ready` (actionable now), `status:blocked` (waiting on something)

Workflow:

- Only work issues labeled `status:ready`.
- Self-assign an issue before starting so the other maintainer doesn't duplicate it.
- File new issues with a clear scope and acceptance criteria.

## Definition of done

The project is "done" when the roadmap milestone is complete, there are no open
`status:ready` issues, no open PRs, and CI is green on `main`. When that state holds, there
is nothing to do.

## Kill switch

If a file named `STOP` exists at the repository root on `main`, both maintainers halt
immediately and take no action until it is removed.
