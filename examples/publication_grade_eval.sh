#!/usr/bin/env bash
# Publication-grade evaluation with reproducible bootstrap CIs.
#
# Demonstrates the 2026 industry-standard recipe:
#   - Bootstrap confidence intervals (BCa method)
#   - Paired t-test (more statistical power for same-prompt comparisons)
#   - Reproducibility via --bootstrap-seed
#
# Use this pattern when writing papers, reports, or benchmarks where
# others need to verify your numbers.

set -euo pipefail

cli-modelarium "Explain the theory of relativity in one paragraph." \
  --models gpt-5.5,claude-opus-4-7 \
  --runs 30 \
  --judge gemini-3.1-pro \
  --significance-test paired-t \
  --confidence-intervals \
  --ci-level 0.95 \
  --ci-method bca \
  --bootstrap-resamples 5000 \
  --bootstrap-seed 42 \
  --output publication_eval.json
