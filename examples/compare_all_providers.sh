#!/usr/bin/env bash
# Compare every cloud model you have a key for (--models all), cost-capped.
#
# `all` resolves to every cloud model with a configured API key (it excludes
# local models and OpenRouter). Because it fans out across all your providers,
# pair it with --max-cost as a safety ceiling.

set -euo pipefail

cli-modelarium "Summarize the CAP theorem in 3 sentences." --models all --max-cost 0.50
