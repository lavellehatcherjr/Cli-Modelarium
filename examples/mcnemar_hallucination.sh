#!/usr/bin/env bash
# McNemar's test for hallucination rate comparison.
#
# McNemar's test auto-triggers when:
#   --check-hallucination is set
#   --runs > 1
#   2 or more models being compared
#
# Tests whether the difference in pass/fail rates between models is
# statistically significant. Uses exact binomial test for small samples
# and Edwards-corrected chi-square for larger ones - not chi2_contingency
# (which would compute a test of independence, not McNemar).

set -euo pipefail

cli-modelarium batch examples/hallucination_test.json \
  --models gpt-5.5,claude-opus-4-7 \
  --runs 20 \
  --judge gemini-3.1-pro \
  --check-hallucination \
  --expected-facts-file examples/expected_facts_example.txt \
  --bootstrap-seed 42 \
  --output mcnemar_hallucination_results.json
