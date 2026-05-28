#!/usr/bin/env bash
# Statistical significance testing - compare models with proper tests.
#
# When --runs > 1 with 2+ models, cli-modelarium automatically computes
# pairwise significance tests with Cohen's d effect sizes. Customize the
# test (welch/mann-whitney/paired-t/wilcoxon-signed) and correction method
# (bonferroni/holm) to match your data.

set -euo pipefail

cli-modelarium "Solve this step by step: What is 247 multiplied by 389?" \
  --models gpt-5.5,claude-opus-4-7,gemini-3.1-pro \
  --runs 30 \
  --judge mistral-large \
  --significance-test welch \
  --correction holm \
  --significance-threshold 0.01 \
  --output significance_results.json
