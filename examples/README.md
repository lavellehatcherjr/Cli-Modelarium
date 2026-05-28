# Cli Modelarium Examples

## Basic examples

- `basic_comparison.sh` - Compare a single prompt across 3 providers
- `batch_evaluation.json` - Multi-prompt batch with assertions (`contains`,
  `json_valid`, `json_schema`, `regex`)
- `hallucination_test.json` - Hallucination detection input format
- `expected_facts_example.txt` - Reference facts file format (one fact
  per line, `#` for comments)
- `ci_eval_suite.json` - CI/CD-ready evaluation suite with assertions for
  refusals, performance, and cost
- `github_actions_workflow.yml` - Example GitHub Actions workflow for
  prompt regression testing on pull requests

## Statistical evaluation examples

Demonstrate the statistical features shipped in v0.1.1, v0.1.2, and v0.1.3:

- `reproducibility_analysis.sh` (v0.1.1) - Run prompts N times to measure
  variance in latency, tokens, and outputs across runs
- `statistical_significance.sh` (v0.1.2) - Pairwise significance testing
  (Welch/Mann-Whitney) with Bonferroni/Holm correction and Cohen's d
  effect sizes
- `publication_grade_eval.sh` (v0.1.3) - Bootstrap confidence intervals
  + paired t-test with `--bootstrap-seed` for reproducibility
- `mcnemar_hallucination.sh` (v0.1.3) - McNemar's test on hallucination
  rates (auto-triggers with `--check-hallucination` + `--runs > 1`)

See the main [README.md](../README.md) for full usage documentation.

## Try them

```bash
# Single-prompt comparison
./examples/basic_comparison.sh

# Batch with assertions; CSV output
cli-modelarium batch examples/batch_evaluation.json \
  --models gpt-5.5,claude-opus-4-7 \
  --output results.csv

# Hallucination check against reference facts
cli-modelarium batch examples/hallucination_test.json \
  --models gpt-5.5 \
  --judge claude-opus-4-7 \
  --check-hallucination \
  --expected-facts-file examples/expected_facts_example.txt

# Reproducibility analysis across 3 models
./examples/reproducibility_analysis.sh

# Statistical significance testing
./examples/statistical_significance.sh

# Publication-grade evaluation with reproducible CIs
./examples/publication_grade_eval.sh

# McNemar's test on hallucination rates
./examples/mcnemar_hallucination.sh
```
