#!/usr/bin/env bash
# Run a curated model group with a single flag (model groups).
#
# Static groups: all-premium (= all-flagship), all-budget, all-reasoning,
# all-fast, all-cheap, all-open-weight. A group is filtered to the providers
# you have keys for; `cli-modelarium list-models` shows configured models.
# all-budget = gpt-5.4-nano, claude-haiku-4-5, gemini-3.1-flash-lite,
# grok-4.1-fast, deepseek-v4-flash, mistral-small-latest.

set -euo pipefail

cli-modelarium "Explain the CAP theorem in 2 sentences." --models all-budget
