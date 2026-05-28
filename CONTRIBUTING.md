# Contributing to Cli Modelarium

Thanks for your interest in Cli Modelarium - a terminal tool for
statistically rigorous LLM comparison across multiple providers.

## Quick Links

- GitHub: https://github.com/lavellehatcherjr/cli-modelarium
- Issues: https://github.com/lavellehatcherjr/cli-modelarium/issues/new
- Maintainer: Lavelle Hatcher Jr ([@lavellehatcherjr](https://github.com/lavellehatcherjr)) - Creator & Maintainer

## How to Contribute

Cli Modelarium is maintained by one person. I personally review and decide on
every contribution. Not everything will be merged - so for anything beyond a
small fix, please open an issue first so we can agree on the approach before
you spend time writing code. This saves your effort as much as mine.

1. **Typos and small bug fixes** -> open a pull request directly.
2. **New features or larger changes** -> open an issue first to discuss the
   idea. Please don't send a large PR without checking in - it may not fit the
   direction of the project, and I'd rather save you the work.
3. **Refactor-only or style-only PRs** -> please don't open these unless I've
   asked for them as part of a specific fix.
4. **Questions** -> open an issue and I'll respond when I can.

Because I'm a solo maintainer, reviews can take some time. Thank you for your
patience, and thank you for helping make the project better.

## Development Setup

Clone the repository and install it with the development extras:

```bash
git clone https://github.com/lavellehatcherjr/cli-modelarium.git
cd cli-modelarium
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,schema]"
```

This requires Python 3.11 or newer.

## Running Tests and Lint

Before opening a pull request, please run the test suite and the linter so
your change is easy to review:

```bash
# Run the full test suite
python -m pytest tests/

# Run the linter on your changes
ruff check .
```

All tests should pass. Please make sure your change does not add new lint
warnings. (A few pre-existing lint items are being cleaned up separately, so
don't worry about those.)

## What Makes a Good PR

- Keep it focused - one logical change per PR.
- Include or update tests for any behavior you change.
- Match the existing code style.
- Describe what the change does and why in the PR description.

## Security Issues

See [SECURITY.md](SECURITY.md) for reporting security vulnerabilities. Do
NOT open public issues for security problems.

## License

By contributing, you agree that your contributions will be licensed under the
project's Apache 2.0 license.
