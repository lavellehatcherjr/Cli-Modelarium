#!/usr/bin/env bash
# Reproducibility analysis - run the same prompt N times across models
# to see variance in latency, tokens, and outputs.
#
# Output includes mean/median/stdev of latency, coefficient of variation,
# mode answer, and output diversity per model.

set -euo pipefail

cli-modelarium "What is quantum computing in one paragraph?" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro-preview \
  --runs 10 \
  --output reproducibility_results.csv \
  --output-format csv
