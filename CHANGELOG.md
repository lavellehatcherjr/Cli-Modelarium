# Changelog

All notable changes to Cli Modelarium will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2026-05-28

### Added

- Bootstrap confidence intervals on per-cell means via `scipy.stats.bootstrap`. Auto-enabled when `--runs > 1`.
- New CLI flags on `compare`:
  - `--confidence-intervals` / `--no-confidence-intervals` (auto-enabled with `--runs > 1`)
  - `--ci-level FLOAT` (default `0.95`)
  - `--ci-method {bca,percentile,basic}` (default `bca` - publication-grade)
  - `--bootstrap-resamples INT` (default `5000`, min `100`)
  - `--bootstrap-seed INT` (required for reproducible CIs)
- New test choices on `--significance-test`:
  - `paired-t` - paired t-test via `scipy.stats.ttest_rel` (more statistical power for same-prompt comparisons)
  - `wilcoxon-signed` - Wilcoxon signed-rank via `scipy.stats.wilcoxon` (non-parametric paired)
- McNemar's test for paired binary outcomes. Auto-triggers when `--check-hallucination` is set with `--runs > 1` and 2+ models. Uses exact binomial test (`scipy.stats.binomtest`) for small discordant counts or Edwards continuity-corrected chi-square (`scipy.stats.chi2.sf`) for larger samples - NOT `scipy.stats.chi2_contingency` on the full 2×2 table (which would compute a test of independence, not McNemar).
- Bootstrap CIs on Cohen's d effect sizes via paired/independent bootstrap.
- `mcnemar_tests` array in JSON output when applicable.
- `methodology` block in JSON output recording bootstrap parameters, scipy version, and seed for reproducibility.
- Additive CI columns in CSV output (`latency_ms_ci_low`, `latency_ms_ci_high`, etc.) when CIs are enabled.
- Additive Markdown sections: "Bootstrap confidence intervals", "Statistical significance tests", "Binary outcome significance (McNemar)", "Statistical methodology".
- New public functions in `cli_modelarium.run_statistics`:
  - `ConfidenceInterval`, `McNemarResult` dataclasses
  - `bootstrap_ci()` - thin scipy wrapper with degenerate-data handling
  - `paired_t_test()`, `wilcoxon_signed_rank()`
  - `mcnemar_test()` - Edwards-corrected or exact binomial McNemar
  - `compute_significance_with_ci()` - like `compute_pairwise_significance` plus CIs on Cohen's d
  - `compute_stats_with_cis()` - CIs on per-model metric means
  - `compute_mcnemar_pairwise()` - pairwise McNemar over hallucination pass/fail
- New private helpers ensuring paired tests align by `run_index` even when failures are asymmetric:
  - `_extract_paired_metric_samples()`
  - `_align_paired_samples()`

### Changed

- `SignificanceResult` dataclass extended with seven optional fields (all default `None`): `bootstrap_ci_low`, `bootstrap_ci_high`, `bootstrap_method`, `bootstrap_resamples`, `bootstrap_seed`, `effect_size_ci_low`, `effect_size_ci_high`. v0.1.2-style positional instantiation continues to work unchanged.
- Output formatters extended so CSV, Markdown, and JSON all receive significance, CI, McNemar, and methodology data (previously only JSON received significance results - a v0.1.2 wiring gap).
- `_emit_batch_results` threads the new parameters into every formatter branch.

### Dependencies

- No new runtime dependencies (uses scipy 1.17 already in v0.1.2).
- `NOTICE` unchanged - scipy is already attributed.
- Python version unchanged (still `>=3.11` from v0.1.2).

## [0.1.2] - 2026-05-28

### ⚠️ Breaking Changes

- **Minimum Python version is now 3.11** (was 3.10).
  - Reason: scipy 1.17+ is a new runtime dependency for statistical significance testing, and scipy 1.17 requires Python 3.11+.
  - Python 3.10 users can continue using cli-modelarium v0.1.1, which remains available on PyPI.
  - Python 3.10 reaches end-of-life in October 2026.

### Added

- Pairwise statistical significance testing on the `compare` command. Auto-enabled when `--runs > 1` with 2+ models.
- New CLI flags on `compare`:
  - `--significance` / `--no-significance` (auto-enabled with `--runs > 1` and 2+ models)
  - `--significance-threshold FLOAT` (default: `0.05`)
  - `--significance-test {welch,mann-whitney}` (default: `welch`)
  - `--correction {none,bonferroni,holm}` (default: `bonferroni`)
  - `--significance-metric {score,latency_ms,output_tokens,cost_usd}` (default: `score` when judging, else `latency_ms`)
- Welch's t-test via `scipy.stats.ttest_ind(equal_var=False)`.
- Mann-Whitney U test via `scipy.stats.mannwhitneyu` with continuity correction.
- Cohen's d effect size with conventional interpretation bands (`negligible` / `small` / `medium` / `large`), implemented in pure stdlib.
- Bonferroni and Holm-Bonferroni multiple-comparison corrections, implemented in pure stdlib with monotone enforcement.
- JSON output now includes a `significance_tests` array when significance testing was performed. When significance is disabled or trivially absent, the JSON schema is unchanged (additive only).
- Display strategy: single-line summary for 2 models, matrix table for 3-5 models, top-K significant pairs for 6+ models (full matrix available in JSON).
- New functions in `cli_modelarium.run_statistics`:
  - `SignificanceResult` dataclass
  - `compute_pairwise_significance()`
  - `welch_t_test()`, `mann_whitney_u_test()`
  - `cohens_d()`, `cohens_d_interpretation()`
  - `bonferroni_correct()`, `holm_correct()`

### Changed

- `pyproject.toml`: `requires-python` bumped to `>=3.11`.
- `pyproject.toml`: classifiers updated - removed Python 3.10, added 3.13 and 3.14.
- `pyproject.toml`: `[tool.ruff].target-version` bumped to `py311`.
- `NOTICE`: added attributions for scipy, numpy, and the bundled native libraries (OpenBLAS, LAPACK, libquadmath).
- README: new "System Requirements" section and statistical-significance documentation.

### Fixed

- Resolved a latent inconsistency: `tests/test_jsonschema_optional.py` already imported `tomllib` (Python 3.11+ stdlib only) while the project declared 3.10 support. The Python bump retroactively fixes this.

### Dependencies

- Added: `scipy>=1.17,<2.0` (pulls `numpy>=1.26.4` as a transitive dependency).

## [0.1.1] - 2026-05-27

### Added

- `--runs N` flag on the `compare` command for statistical reproducibility analysis. Runs each (model, temperature, system_prompt) combination N times (1-100) and displays mean/median/stdev/CV of timing and tokens, cost totals, output frequency analysis, mode output, and output diversity.
- `--show-all-runs` flag to override the auto-collapse heuristic when many concurrent display panels would be created.
- New module `src/cli_modelarium/run_statistics.py` with `RunStats` dataclass and `compute_run_stats()` function for pure-stdlib statistical analysis.
- Hallucination rate calculation when `--check-hallucination` is combined with `--runs N`. Reports "N of M runs flagged as High risk" with the aggregate hallucination rate.
- Cost warning when `--runs N` is used without `--max-cost` (helps prevent unexpected spend).

### Changed

- `compare` command's display path branches when `runs > 1` to show statistical summary instead of per-run details. When `runs == 1` (default), behavior is byte-identical to v0.1.0.
- `run_streaming_comparison()` accepts new keyword parameters `runs: int = 1` and `show_all_runs: bool = False`. Default values preserve existing behavior.
- `StreamState` dataclass has new field `run_index: int = 0`. Default value preserves all existing test expectations.
- `BatchResult` dataclass has new field `run_index: int = 0`. Only emitted in CSV/JSON/Markdown output when the surrounding `runs` parameter > 1.
- Live streaming display auto-collapses when total concurrent tasks exceed 12 (configurable via `--show-all-runs`).
- LLM-as-judge with `--runs N`:
  - Default (with `--judge` or `--judges`): mode-only judging (one judge call per cell, expanded to every run in the cell)
  - With `--check-hallucination`: per-run judging (computes hallucination rate)
- JSON output schema additive when `runs > 1`: new `total_runs` and `stats_by_cell` top-level fields, plus `run_index` per result. When `runs == 1`, schema is byte-identical to v0.1.0.
- CSV output adds `run_index` column when `runs > 1`. When `runs == 1`, columns are unchanged.
- Markdown output adds a "Per-cell statistical summary" section when `runs > 1`. When `runs == 1`, output is unchanged.

### Fixed

- N/A (no bug fixes in this release; only additions)

## [0.1.0] - 2026-05-25

### Added

- Initial v0.1.0 release
- 8 cloud provider integrations: OpenAI, Anthropic, Google, xAI, DeepSeek, Mistral, Groq, OpenRouter
- Local model support: Ollama, LM Studio, vLLM, llama.cpp via OpenAI-compatible API
- Parallel streaming with TTFT (Time To First Token) tracking
- Multi-prompt batch mode with CSV, JSON, and Markdown output formats
- System prompt support: single, multiple (comparison), and file-based
- LLM-as-a-judge scoring with panel mode and self-evaluation skip
- Deterministic assertions with 10 types (`contains`, `not_contains`, `regex`, `equals`, `json_valid`, `json_schema`, `min_length_chars`, `max_length_chars`, `latency_under`, `cost_under`)
- CI/CD exit codes: 0 = success, 1 = assertion failure, 2 = call failure (call failures dominate)
- Hallucination detection preset with optional reference facts and worst-wins panel aggregation
- OS-native keychain integration via `keyring`
- API key format validation for 8 providers
- Error message redaction prevents key leakage
- Localhost-only validation for local model URLs
- Cross-platform support: macOS, Windows 10+/ARM, Linux
- Rate limit handling: 429 retry with exponential backoff, 529 (Anthropic overloaded) with longer backoff
- `retry-after` header honored when present
- Per-provider semaphores for concurrent request management
- Atomic file writes for output integrity
- Apache 2.0 License with proper NOTICE attribution
