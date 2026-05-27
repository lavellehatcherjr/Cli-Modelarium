"""Statistical analysis of multi-run comparisons.

For the --runs N feature on the compare command. Computes mean, median,
standard deviation, coefficient of variation, frequency analysis, mode
output, and output diversity from a list of StreamState results.

The run-aggregation half uses pure stdlib (statistics, collections.Counter).
The pairwise significance half (v0.1.2) delegates the math to scipy.stats
(Welch's t-test, Mann-Whitney U) and implements Cohen's d plus
Bonferroni/Holm corrections in pure stdlib.

The contract: pass a list of StreamState objects representing N runs of
the SAME (model, temperature, system_prompt) cell. Returns a RunStats
dataclass with all metrics computed only on successful runs.

Failed runs (state.error is not None) are counted in n_failed but excluded
from all numerical statistics. n_succeeded < 2 means stdev is undefined
and returned as None.

Mode tie-breaking: when all outputs are unique, mode_output is None and
mode_count is 0. The user sees this as "no mode" rather than an arbitrary
pick - honest about high model variability.
"""

from __future__ import annotations

import math
import statistics as stdlib_statistics
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import scipy.stats as _scipy_stats

from cli_modelarium.streaming import StreamState


@dataclass
class RunStats:
    """Aggregate statistics for N runs of the same cell.

    All numerical stats are computed only on successful runs.
    Values are None when undefined (n_succeeded < 1 for means,
    n_succeeded < 2 for stdev).
    """

    n_runs: int
    n_succeeded: int
    n_failed: int

    # Timing statistics
    latency_mean_ms: float | None
    latency_median_ms: float | None
    latency_stdev_ms: float | None
    latency_cv: float | None  # Coefficient of variation = stdev/mean
    ttft_mean_ms: float | None

    # Token statistics
    output_tokens_mean: float | None
    output_tokens_stdev: float | None

    # Cost statistics
    cost_total_usd: float  # Sum across all successful runs
    cost_mean_usd: float | None  # None if no successful runs

    # Output analysis
    unique_outputs: int
    mode_output: str | None  # None when all unique (no mode)
    mode_count: int  # 0 when no mode
    output_diversity: float  # unique_outputs / n_succeeded (1.0 means all unique)


def compute_run_stats(states: list[StreamState]) -> RunStats:
    """Compute statistics from a group of N runs.

    Statistics are computed only on successful runs. Failed runs are
    counted in n_failed but don't contribute to means/stdevs.
    """
    n_runs = len(states)
    successful = [s for s in states if s.error is None]
    failed = [s for s in states if s.error is not None]
    n_succeeded = len(successful)
    n_failed = len(failed)

    latencies = [s.latency_ms for s in successful if s.latency_ms is not None]
    ttfts = [s.ttft_ms for s in successful if s.ttft_ms is not None]
    output_token_counts = [s.output_tokens for s in successful]
    costs = [s.cost_usd for s in successful]

    latency_mean_ms = stdlib_statistics.mean(latencies) if latencies else None
    latency_median_ms = stdlib_statistics.median(latencies) if latencies else None
    latency_stdev_ms = stdlib_statistics.stdev(latencies) if len(latencies) >= 2 else None

    if latency_mean_ms and latency_stdev_ms and latency_mean_ms > 0:
        latency_cv = latency_stdev_ms / latency_mean_ms
    else:
        latency_cv = None

    ttft_mean_ms = stdlib_statistics.mean(ttfts) if ttfts else None

    output_tokens_mean = (
        stdlib_statistics.mean(output_token_counts) if output_token_counts else None
    )
    output_tokens_stdev = (
        stdlib_statistics.stdev(output_token_counts)
        if len(output_token_counts) >= 2
        else None
    )

    cost_total_usd = sum(costs)
    cost_mean_usd = stdlib_statistics.mean(costs) if costs else None

    # Output frequency analysis (exact string matching, no normalization).
    outputs = [s.text for s in successful]
    output_counter = Counter(outputs)
    unique_outputs = len(output_counter)

    # Mode: only when at least one output appears more than once.
    # All-unique returns no mode (honest about high variability).
    if unique_outputs < n_succeeded and output_counter:
        most_common = output_counter.most_common(1)[0]
        mode_output = most_common[0]
        mode_count = most_common[1]
    else:
        mode_output = None
        mode_count = 0

    output_diversity = unique_outputs / n_succeeded if n_succeeded > 0 else 0.0

    return RunStats(
        n_runs=n_runs,
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        latency_mean_ms=latency_mean_ms,
        latency_median_ms=latency_median_ms,
        latency_stdev_ms=latency_stdev_ms,
        latency_cv=latency_cv,
        ttft_mean_ms=ttft_mean_ms,
        output_tokens_mean=output_tokens_mean,
        output_tokens_stdev=output_tokens_stdev,
        cost_total_usd=cost_total_usd,
        cost_mean_usd=cost_mean_usd,
        unique_outputs=unique_outputs,
        mode_output=mode_output,
        mode_count=mode_count,
        output_diversity=output_diversity,
    )


def group_states_by_cell(
    states: list[StreamState],
) -> dict[tuple[str, float, str | None], list[StreamState]]:
    """Group N x M x T x S states by (model, temperature, system_prompt) cell.

    Returns a dict keyed by cell tuple, values are lists of states sorted
    by run_index for deterministic ordering.
    """
    groups: dict[tuple[str, float, str | None], list[StreamState]] = {}
    for state in states:
        key = (state.model, state.temperature, state.system_prompt)
        groups.setdefault(key, []).append(state)

    for key in groups:
        groups[key].sort(key=lambda s: s.run_index)

    return groups


# ===========================================================================
# v0.1.2: pairwise statistical significance testing
# ===========================================================================


@dataclass
class SignificanceResult:
    """Result of a pairwise statistical significance test.

    Produced for each pair of models when --runs N runs the same prompt
    multiple times on 2+ models. Math is delegated to scipy.stats for
    the test statistic; Cohen's d and corrections are computed locally.
    """

    model_a: str
    model_b: str
    metric: str  # "score", "latency_ms", "output_tokens", "cost_usd"
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    stdev_a: float | None
    stdev_b: float | None
    # Values: "welch_t_test", "mann_whitney_u", "trivial",
    # "zero_variance", or "insufficient_samples".
    test_used: str
    test_statistic: float | None
    degrees_of_freedom: float | None  # None for Mann-Whitney / non-applicable
    p_value: float | None  # None when no test could be run
    p_value_corrected: float | None  # None when raw p_value is None
    correction_method: str  # "bonferroni", "holm", or "none"
    n_comparisons: int  # Total pairwise comparisons (drives correction)
    effect_size: float | None  # Cohen's d (None when undefined)
    effect_size_interpretation: str  # "negligible", "small", "medium", "large", "undefined"
    threshold: float  # User-specified significance threshold
    significant_at_threshold: bool  # corrected p < threshold

    # v0.1.3 additions - all optional to preserve v0.1.2 backward compat (F2).
    # When None, the field was not requested or could not be computed.
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    bootstrap_method: str | None = None  # "bca", "percentile", "basic"
    bootstrap_resamples: int | None = None
    bootstrap_seed: int | None = None
    effect_size_ci_low: float | None = None
    effect_size_ci_high: float | None = None


def cohens_d(sample_a: list[float], sample_b: list[float]) -> float | None:
    """Cohen's d effect size for two independent samples.

    Pooled standard deviation with (n_a + n_b - 2) denominator (Cohen 1988).
    Returns None when undefined (n < 2 in either group, or both groups
    have zero variance with different means).

    d = (mean_a - mean_b) / s_pooled
    s_pooled = sqrt(((n_a-1)*var_a + (n_b-1)*var_b) / (n_a + n_b - 2))
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    if n_a < 2 or n_b < 2:
        return None

    mean_a = stdlib_statistics.mean(sample_a)
    mean_b = stdlib_statistics.mean(sample_b)
    var_a = stdlib_statistics.variance(sample_a)  # Bessel's correction (n-1)
    var_b = stdlib_statistics.variance(sample_b)

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var == 0:
        # Both samples constant. Equal means => d=0; different => undefined.
        return 0.0 if mean_a == mean_b else None

    return (mean_a - mean_b) / math.sqrt(pooled_var)


def cohens_d_interpretation(d: float | None) -> str:
    """Map Cohen's d magnitude to conventional bands (Cohen 1988).

    |d| < 0.2: negligible
    0.2 <= |d| < 0.5: small
    0.5 <= |d| < 0.8: medium
    0.8 <= |d|: large
    """
    if d is None:
        return "undefined"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def bonferroni_correct(
    p_values: list[float], n_comparisons: int | None = None
) -> list[float]:
    """Bonferroni: multiply each p-value by n, cap at 1.0.

    When n_comparisons is None, uses len(p_values).
    """
    if not p_values:
        return []
    n = n_comparisons if n_comparisons is not None else len(p_values)
    return [min(1.0, p * n) for p in p_values]


def holm_correct(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction.

    1. Sort p-values ascending (track original indices).
    2. Raw adjusted p[i] = p_sorted[i] * (n - i)  for rank i (0-indexed).
    3. Apply monotone enforcement: running max so adjusted p-values can
       never decrease as raw p-values increase.
    4. Cap at 1.0.
    5. Return in original order.
    """
    if not p_values:
        return []
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    monotone: list[tuple[int, float]] = []
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        raw_adj = p * (n - rank)
        running_max = max(running_max, raw_adj)
        monotone.append((orig_idx, min(1.0, running_max)))

    result = [0.0] * n
    for orig_idx, adj_p in monotone:
        result[orig_idx] = adj_p
    return result


def welch_t_test(
    sample_a: list[float], sample_b: list[float]
) -> tuple[float, float, float]:
    """Welch's t-test wrapper around scipy.stats.ttest_ind(equal_var=False).

    Returns (t_statistic, degrees_of_freedom, two_tailed_p_value).
    Raises ValueError if either sample has fewer than 2 elements.

    scipy 1.17+ exposes the Welch-Satterthwaite df via result.df; older
    scipy releases don't, so we compute it manually as a fallback.
    """
    if len(sample_a) < 2 or len(sample_b) < 2:
        raise ValueError("Welch's t-test requires at least 2 samples per group")

    result = _scipy_stats.ttest_ind(sample_a, sample_b, equal_var=False)
    df = getattr(result, "df", None)
    if df is None:
        var_a = stdlib_statistics.variance(sample_a)
        var_b = stdlib_statistics.variance(sample_b)
        n_a, n_b = len(sample_a), len(sample_b)
        num = (var_a / n_a + var_b / n_b) ** 2
        den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / den if den > 0 else 0.0

    return float(result.statistic), float(df), float(result.pvalue)


def mann_whitney_u_test(
    sample_a: list[float], sample_b: list[float]
) -> tuple[float, float]:
    """Mann-Whitney U two-sided test via scipy.stats.mannwhitneyu.

    Uses the default continuity correction and method='auto' (scipy picks
    exact for small samples, asymptotic for large). Returns (U_statistic
    for sample_a, two_tailed_p_value).

    Raises ValueError if either sample is empty.
    """
    if len(sample_a) < 1 or len(sample_b) < 1:
        raise ValueError("Mann-Whitney U test requires at least 1 sample per group")
    result = _scipy_stats.mannwhitneyu(
        sample_a,
        sample_b,
        alternative="two-sided",
        use_continuity=True,
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def _extract_metric_samples(
    states_by_model: dict[str, list[Any]],
    judge_results: list[Any] | None,
    metric: str,
) -> dict[str, list[float]]:
    """Pull per-model metric values from states (+ optional judge results).

    For metric="score" the values come from JudgeResult.average_score
    via judge_results. For all other metrics the values come from the
    StreamState fields directly. Failed states (state.error not None)
    are excluded.
    """
    samples: dict[str, list[float]] = {}

    if metric == "score":
        # Map each judge_result back to its source state by position.
        # Both lists are kept in parallel by the caller (run_judging).
        scores_by_id: dict[int, float] = {}
        if judge_results:
            for jr in judge_results:
                avg = getattr(jr, "average_score", None)
                if avg is None:
                    continue
                state_id = getattr(jr, "_state_id", None)
                if state_id is not None:
                    scores_by_id[state_id] = float(avg)
        for model, states in states_by_model.items():
            samples[model] = [
                scores_by_id[id(s)]
                for s in states
                if s.error is None and id(s) in scores_by_id
            ]
        return samples

    for model, states in states_by_model.items():
        successful = [s for s in states if s.error is None]
        if metric == "latency_ms":
            samples[model] = [
                float(s.latency_ms) for s in successful if s.latency_ms is not None
            ]
        elif metric == "output_tokens":
            samples[model] = [float(s.output_tokens) for s in successful]
        elif metric == "cost_usd":
            samples[model] = [float(s.cost_usd) for s in successful]
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return samples


def compute_pairwise_significance(
    states_by_model: dict[str, list[Any]],
    judge_results: list[Any] | None,
    *,
    metric: str = "latency_ms",
    test: Literal[
        "welch", "mann-whitney", "paired-t", "wilcoxon-signed"
    ] = "welch",
    correction: Literal["none", "bonferroni", "holm"] = "bonferroni",
    threshold: float = 0.05,
) -> list[SignificanceResult]:
    """All pairwise significance tests between models, with correction.

    Returns one SignificanceResult per unordered (model_a, model_b) pair.
    Pair order follows insertion order in states_by_model so output is
    deterministic.

    Tests:
      - "welch": Welch's t-test (independent samples, unequal variances)
      - "mann-whitney": Mann-Whitney U (non-parametric, independent)
      - "paired-t": paired t-test (same prompts, requires run_index alignment)
      - "wilcoxon-signed": Wilcoxon signed-rank (non-parametric, paired)

    For paired tests, samples are aligned by run_index so position i on
    model A matches position i on model B by the same original run, even
    when failures are asymmetric.

    Edge cases:
      - n < 3 in either group: test_used="insufficient_samples", p=None
      - both groups zero-variance, same mean: test_used="trivial", p=1.0
      - both groups zero-variance, different means: test_used="zero_variance", p=None
      - judge_results is None and metric="score": empty samples per model
    """
    models = list(states_by_model.keys())
    if len(models) < 2:
        return []

    is_paired = test in ("paired-t", "wilcoxon-signed")
    if is_paired:
        # Per-model dict keyed by run_index; align per-pair below.
        paired_by_model = _extract_paired_metric_samples(
            states_by_model, judge_results, metric
        )
        samples_by_model: dict[str, list[float]] = {
            m: list(d.values()) for m, d in paired_by_model.items()
        }
    else:
        paired_by_model = None
        samples_by_model = _extract_metric_samples(
            states_by_model, judge_results, metric
        )

    pairs: list[tuple[str, str]] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            pairs.append((model_a, model_b))
    n_comparisons = len(pairs)

    raw_results: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    for model_a, model_b in pairs:
        if is_paired and paired_by_model is not None:
            sample_a, sample_b = _align_paired_samples(
                paired_by_model.get(model_a, {}),
                paired_by_model.get(model_b, {}),
            )
        else:
            sample_a = samples_by_model.get(model_a, [])
            sample_b = samples_by_model.get(model_b, [])
        n_a, n_b = len(sample_a), len(sample_b)

        mean_a = stdlib_statistics.mean(sample_a) if sample_a else 0.0
        mean_b = stdlib_statistics.mean(sample_b) if sample_b else 0.0
        stdev_a = stdlib_statistics.stdev(sample_a) if n_a >= 2 else None
        stdev_b = stdlib_statistics.stdev(sample_b) if n_b >= 2 else None

        if n_a < 3 or n_b < 3:
            raw_results.append(
                {
                    "model_a": model_a, "model_b": model_b,
                    "n_a": n_a, "n_b": n_b,
                    "mean_a": mean_a, "mean_b": mean_b,
                    "stdev_a": stdev_a, "stdev_b": stdev_b,
                    "test_used": "insufficient_samples",
                    "test_statistic": None, "degrees_of_freedom": None,
                    "p_value": None, "effect_size": None,
                }
            )
            raw_p_values.append(1.0)
            continue

        if stdev_a == 0 and stdev_b == 0:
            if mean_a == mean_b:
                raw_results.append(
                    {
                        "model_a": model_a, "model_b": model_b,
                        "n_a": n_a, "n_b": n_b,
                        "mean_a": mean_a, "mean_b": mean_b,
                        "stdev_a": 0.0, "stdev_b": 0.0,
                        "test_used": "trivial",
                        "test_statistic": 0.0, "degrees_of_freedom": None,
                        "p_value": 1.0, "effect_size": 0.0,
                    }
                )
                raw_p_values.append(1.0)
            else:
                raw_results.append(
                    {
                        "model_a": model_a, "model_b": model_b,
                        "n_a": n_a, "n_b": n_b,
                        "mean_a": mean_a, "mean_b": mean_b,
                        "stdev_a": 0.0, "stdev_b": 0.0,
                        "test_used": "zero_variance",
                        "test_statistic": None, "degrees_of_freedom": None,
                        "p_value": None, "effect_size": None,
                    }
                )
                raw_p_values.append(1.0)
            continue

        if test == "welch":
            t_stat, df, p_value = welch_t_test(sample_a, sample_b)
            test_used = "welch_t_test"
        elif test == "mann-whitney":
            t_stat, p_value = mann_whitney_u_test(sample_a, sample_b)
            df = None
            test_used = "mann_whitney_u"
        elif test == "paired-t":
            t_stat, df, p_value = paired_t_test(sample_a, sample_b)
            test_used = "paired_t_test"
        elif test == "wilcoxon-signed":
            t_stat, p_value = wilcoxon_signed_rank(sample_a, sample_b)
            df = None
            test_used = "wilcoxon_signed_rank"
        else:
            raise ValueError(f"Unknown test: {test}")

        d = cohens_d(sample_a, sample_b)
        raw_results.append(
            {
                "model_a": model_a, "model_b": model_b,
                "n_a": n_a, "n_b": n_b,
                "mean_a": mean_a, "mean_b": mean_b,
                "stdev_a": stdev_a, "stdev_b": stdev_b,
                "test_used": test_used,
                "test_statistic": t_stat,
                "degrees_of_freedom": df,
                "p_value": p_value,
                "effect_size": d,
            }
        )
        raw_p_values.append(p_value)

    if correction == "bonferroni":
        corrected_p_values = bonferroni_correct(raw_p_values, n_comparisons)
    elif correction == "holm":
        corrected_p_values = holm_correct(raw_p_values)
    else:
        corrected_p_values = list(raw_p_values)

    final_results: list[SignificanceResult] = []
    for r, corrected_p in zip(raw_results, corrected_p_values, strict=True):
        if r["p_value"] is None:
            corrected: float | None = None
            significant = False
        else:
            corrected = corrected_p
            significant = corrected_p < threshold

        final_results.append(
            SignificanceResult(
                model_a=r["model_a"],
                model_b=r["model_b"],
                metric=metric,
                n_a=r["n_a"],
                n_b=r["n_b"],
                mean_a=r["mean_a"],
                mean_b=r["mean_b"],
                stdev_a=r["stdev_a"],
                stdev_b=r["stdev_b"],
                test_used=r["test_used"],
                test_statistic=r["test_statistic"],
                degrees_of_freedom=r["degrees_of_freedom"],
                p_value=r["p_value"],
                p_value_corrected=corrected,
                correction_method=correction,
                n_comparisons=n_comparisons,
                effect_size=r["effect_size"],
                effect_size_interpretation=cohens_d_interpretation(r["effect_size"]),
                threshold=threshold,
                significant_at_threshold=significant,
            )
        )

    return final_results


# ===========================================================================
# v0.1.3: bootstrap confidence intervals + paired tests + McNemar's test
# ===========================================================================


@dataclass
class ConfidenceInterval:
    """Bootstrap confidence interval for a scalar statistic.

    Used in per-cell statistics to show uncertainty on means alongside
    the point estimate. Recorded with the parameters that produced it
    so output is reproducible given the same seed.
    """

    point_estimate: float
    ci_low: float
    ci_high: float
    ci_level: float  # e.g. 0.95
    method: str  # "bca", "percentile", "basic"
    n_resamples: int
    seed: int | None  # None = non-deterministic (warned in CLI)
    n_samples: int  # Sample size used


@dataclass
class McNemarResult:
    """McNemar's test result for a paired binary comparison.

    Used for hallucination pass/fail comparisons between models. The
    discordant counts (b, c) are the inputs that matter to the test;
    both_pass / both_fail are recorded for the 2x2 table summary.

    Implementation uses Edwards-corrected chi-square or exact binomial
    test (NOT scipy.stats.chi2_contingency, which tests independence).
    """

    model_a: str
    model_b: str
    metric: str  # e.g. "hallucination_rate"
    both_pass: int
    a_pass_b_fail: int  # discordant: b
    a_fail_b_pass: int  # discordant: c
    both_fail: int
    n_discordant: int
    a_pass_rate: float
    b_pass_rate: float
    chi2_statistic: float | None  # None when exact binomial used
    p_value: float | None  # None when n_discordant == 0
    p_value_corrected: float | None
    correction_method: str  # "bonferroni", "holm", "none"
    n_comparisons: int
    threshold: float
    significant_at_threshold: bool
    method: str  # "exact_binomial" or "edwards_chi2"


def bootstrap_ci(
    samples: list[float],
    statistic: Callable[[Any], float] | None = None,
    *,
    ci_level: float = 0.95,
    method: str = "bca",
    n_resamples: int = 5000,
    seed: int | None = None,
) -> ConfidenceInterval | None:
    """Bootstrap confidence interval for a scalar statistic.

    Thin wrapper around scipy.stats.bootstrap. Default statistic is
    np.mean; default method is "bca" (bias-corrected and accelerated,
    industry standard for publication-grade CIs).

    Returns None when:
      - sample has fewer than 2 observations
      - scipy raises (degenerate data, etc.)
      - the resulting CI bounds are non-finite (zero-variance edge case)
    """
    n = len(samples)
    if n < 2:
        return None

    stat_fn: Callable[[Any], float] = statistic if statistic is not None else np.mean

    data = (samples,)
    try:
        with warnings.catch_warnings():
            # BCa on degenerate data emits a DegenerateDataWarning then
            # returns NaN bounds; we filter the NaN below.
            warnings.simplefilter("ignore")
            result = _scipy_stats.bootstrap(
                data,
                stat_fn,
                n_resamples=n_resamples,
                confidence_level=ci_level,
                method=method,
                random_state=seed,
            )
    except Exception:
        return None

    ci_low = float(result.confidence_interval.low)
    ci_high = float(result.confidence_interval.high)
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        return None

    point = float(stat_fn(samples))
    return ConfidenceInterval(
        point_estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
        method=method,
        n_resamples=n_resamples,
        seed=seed,
        n_samples=n,
    )


def paired_t_test(
    sample_a: list[float], sample_b: list[float]
) -> tuple[float, float, float]:
    """Paired t-test via scipy.stats.ttest_rel.

    Inputs MUST be same-length and aligned by observation. Use
    `_extract_paired_metric_samples` + `_align_paired_samples` to build
    aligned inputs from raw states_by_model.

    Returns (t_statistic, degrees_of_freedom, two_tailed_p_value).
    Raises ValueError on unequal lengths or n < 2.
    """
    if len(sample_a) != len(sample_b):
        raise ValueError(
            f"Paired t-test requires equal-length samples, "
            f"got {len(sample_a)} and {len(sample_b)}"
        )
    if len(sample_a) < 2:
        raise ValueError("Paired t-test requires at least 2 paired samples")

    result = _scipy_stats.ttest_rel(sample_a, sample_b)
    df = float(len(sample_a) - 1)
    return float(result.statistic), df, float(result.pvalue)


def wilcoxon_signed_rank(
    sample_a: list[float], sample_b: list[float]
) -> tuple[float, float]:
    """Wilcoxon signed-rank paired test via scipy.stats.wilcoxon.

    Non-parametric paired test - more robust than paired_t for non-normal
    or ordinal data. Inputs MUST be same-length and aligned by observation.

    Defaults: zero_method="wilcox" (drops zero differences),
    correction=False, alternative="two-sided".

    Returns (W_statistic, two_tailed_p_value).
    Raises ValueError on unequal lengths or n < 2.
    """
    if len(sample_a) != len(sample_b):
        raise ValueError(
            f"Wilcoxon requires equal-length samples, "
            f"got {len(sample_a)} and {len(sample_b)}"
        )
    if len(sample_a) < 2:
        raise ValueError("Wilcoxon requires at least 2 paired samples")

    with warnings.catch_warnings():
        # All-zero-differences emits a harmless RuntimeWarning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = _scipy_stats.wilcoxon(
            sample_a,
            sample_b,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
        )
    return float(result.statistic), float(result.pvalue)


def mcnemar_test(
    b_only_a_pass: int,
    c_only_b_pass: int,
    *,
    exact: bool = False,
) -> tuple[float | None, float]:
    """McNemar's test for paired binary outcomes.

    Critical: do NOT implement via scipy.stats.chi2_contingency on the
    full 2x2 table - that computes a test of independence, not McNemar's.
    McNemar only uses the discordant pairs (off-diagonal entries).

    Args:
        b_only_a_pass: count where model A passes, B fails
        c_only_b_pass: count where model A fails, B passes
        exact: force the exact binomial test (recommended for n < 25)

    Returns:
        (chi2_statistic_or_None, p_value)
        chi2 is None when the exact binomial test is used.

    Algorithm:
      - n_discordant = b + c
      - n_discordant == 0  -> no test possible, returns (None, 1.0)
      - exact OR n_discordant < 25 -> exact binomial test via binomtest
      - otherwise           -> Edwards continuity-corrected chi-square

    References:
      McNemar (1947); Edwards (1948) for continuity correction.
    """
    n_discordant = b_only_a_pass + c_only_b_pass
    if n_discordant == 0:
        return None, 1.0

    if exact or n_discordant < 25:
        result = _scipy_stats.binomtest(b_only_a_pass, n_discordant, p=0.5)
        return None, float(result.pvalue)

    chi2 = (abs(b_only_a_pass - c_only_b_pass) - 1) ** 2 / n_discordant
    p = float(_scipy_stats.chi2.sf(chi2, df=1))
    return float(chi2), p


def _extract_paired_metric_samples(
    states_by_model: dict[str, list[Any]],
    judge_results: list[Any] | None,
    metric: str,
) -> dict[str, dict[int, float]]:
    """Pull per-model metric values indexed by run_index for paired tests.

    Returns {model: {run_index: value}}. Failed states (state.error not
    None) and missing values are excluded so the caller can intersect
    indices to get genuine pairs.

    For metric="score", values come from JudgeResult.average_score via
    the same `_state_id` linkage as `_extract_metric_samples`.
    """
    samples_by_model: dict[str, dict[int, float]] = {}

    if metric == "score":
        scores_by_state_id: dict[int, float] = {}
        if judge_results:
            for jr in judge_results:
                avg = getattr(jr, "average_score", None)
                state_id = getattr(jr, "_state_id", None)
                if avg is not None and state_id is not None:
                    scores_by_state_id[state_id] = float(avg)
        for model, states in states_by_model.items():
            samples_by_model[model] = {
                s.run_index: scores_by_state_id[id(s)]
                for s in states
                if s.error is None and id(s) in scores_by_state_id
            }
        return samples_by_model

    for model, states in states_by_model.items():
        successful = [s for s in states if s.error is None]
        if metric == "latency_ms":
            samples_by_model[model] = {
                s.run_index: float(s.latency_ms)
                for s in successful
                if s.latency_ms is not None
            }
        elif metric == "output_tokens":
            samples_by_model[model] = {
                s.run_index: float(s.output_tokens) for s in successful
            }
        elif metric == "cost_usd":
            samples_by_model[model] = {
                s.run_index: float(s.cost_usd) for s in successful
            }
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return samples_by_model


def _align_paired_samples(
    samples_a: dict[int, float],
    samples_b: dict[int, float],
) -> tuple[list[float], list[float]]:
    """Intersect on run_index, return aligned (a, b) lists sorted by index.

    Position i in the output pair corresponds to the same original
    run_index on both models, so paired_t_test / wilcoxon_signed_rank
    receive genuine pairs even when failures were asymmetric.
    """
    common_indices = sorted(set(samples_a.keys()) & set(samples_b.keys()))
    aligned_a = [samples_a[i] for i in common_indices]
    aligned_b = [samples_b[i] for i in common_indices]
    return aligned_a, aligned_b


def _bootstrap_mean(
    samples: list[float],
    *,
    ci_level: float,
    method: str,
    n_resamples: int,
    seed: int | None,
) -> tuple[float | None, float | None]:
    """Convenience for bootstrap CI on a mean - returns (low, high) or (None, None)."""
    ci = bootstrap_ci(
        samples,
        statistic=np.mean,
        ci_level=ci_level,
        method=method,
        n_resamples=n_resamples,
        seed=seed,
    )
    if ci is None:
        return None, None
    return ci.ci_low, ci.ci_high


def compute_stats_with_cis(
    states_by_model: dict[str, list[Any]],
    judge_results: list[Any] | None,
    *,
    ci_level: float = 0.95,
    ci_method: str = "bca",
    n_resamples: int = 5000,
    seed: int | None = None,
) -> dict[str, dict[str, ConfidenceInterval | None]]:
    """Compute bootstrap CIs on per-model metric means.

    Returns {model: {metric_name: ConfidenceInterval | None}} for the
    four base metrics (latency_ms, output_tokens, cost_usd, plus score
    when judge_results provided).
    """
    out: dict[str, dict[str, ConfidenceInterval | None]] = {}
    metrics = ["latency_ms", "output_tokens", "cost_usd"]
    if judge_results:
        metrics.append("score")

    for metric in metrics:
        samples_by_model = _extract_metric_samples(
            states_by_model, judge_results, metric
        )
        for model, samples in samples_by_model.items():
            ci = bootstrap_ci(
                samples,
                statistic=np.mean,
                ci_level=ci_level,
                method=ci_method,
                n_resamples=n_resamples,
                seed=seed,
            )
            out.setdefault(model, {})[metric] = ci
    return out


def compute_mcnemar_pairwise(
    states_by_model: dict[str, list[Any]],
    judge_by_state_id: dict[int, Any],
    *,
    correction: Literal["none", "bonferroni", "holm"] = "bonferroni",
    threshold: float = 0.05,
    metric: str = "hallucination_rate",
) -> list[McNemarResult]:
    """Pairwise McNemar's tests on hallucination pass/fail outcomes.

    "Pass" is defined as judge.aggregated_risk_level != "High". Runs
    where either model's judge result is missing (or the model failed)
    are skipped for that pair. Intersection on run_index gives the
    set of paired runs used for the 2x2 table.

    Returns one McNemarResult per unordered (model_a, model_b) pair.
    """
    models = list(states_by_model.keys())
    if len(models) < 2:
        return []

    # Build per-model {run_index: pass_bool} using state_id -> judge lookup.
    outcomes_by_model: dict[str, dict[int, bool]] = {}
    for model, states in states_by_model.items():
        m: dict[int, bool] = {}
        for s in states:
            if s.error is not None:
                continue
            jr = judge_by_state_id.get(id(s))
            if jr is None or not getattr(jr, "judges", None):
                continue
            risk = getattr(jr, "aggregated_risk_level", None)
            if risk is None:
                continue
            m[s.run_index] = risk != "High"
        outcomes_by_model[model] = m

    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pairs.append((a, b))
    n_comparisons = len(pairs)

    raw: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for a, b in pairs:
        outcomes_a = outcomes_by_model.get(a, {})
        outcomes_b = outcomes_by_model.get(b, {})
        common = sorted(set(outcomes_a.keys()) & set(outcomes_b.keys()))

        both_pass = a_pass_b_fail = a_fail_b_pass = both_fail = 0
        for idx in common:
            pa, pb = outcomes_a[idx], outcomes_b[idx]
            if pa and pb:
                both_pass += 1
            elif pa and not pb:
                a_pass_b_fail += 1
            elif not pa and pb:
                a_fail_b_pass += 1
            else:
                both_fail += 1

        n_paired = len(common)
        a_pass_rate = (both_pass + a_pass_b_fail) / n_paired if n_paired else 0.0
        b_pass_rate = (both_pass + a_fail_b_pass) / n_paired if n_paired else 0.0
        n_discordant = a_pass_b_fail + a_fail_b_pass

        chi2_stat, p_value = mcnemar_test(a_pass_b_fail, a_fail_b_pass)
        # method label for downstream display
        if n_discordant == 0:
            method_label = "no_discordant"
        elif n_discordant < 25:
            method_label = "exact_binomial"
        else:
            method_label = "edwards_chi2"

        raw.append(
            {
                "model_a": a,
                "model_b": b,
                "metric": metric,
                "both_pass": both_pass,
                "a_pass_b_fail": a_pass_b_fail,
                "a_fail_b_pass": a_fail_b_pass,
                "both_fail": both_fail,
                "n_discordant": n_discordant,
                "a_pass_rate": a_pass_rate,
                "b_pass_rate": b_pass_rate,
                "chi2_statistic": chi2_stat,
                "p_value": p_value,
                "method": method_label,
            }
        )
        # If p_value is None (no discordant pairs), treat as p=1.0 for correction.
        raw_p_values.append(p_value if p_value is not None else 1.0)

    if correction == "bonferroni":
        corrected = bonferroni_correct(raw_p_values, n_comparisons)
    elif correction == "holm":
        corrected = holm_correct(raw_p_values)
    else:
        corrected = list(raw_p_values)

    results: list[McNemarResult] = []
    for r, c_p in zip(raw, corrected, strict=True):
        if r["p_value"] is None:
            corrected_p: float | None = None
            sig = False
        else:
            corrected_p = c_p
            sig = c_p < threshold
        results.append(
            McNemarResult(
                model_a=r["model_a"],
                model_b=r["model_b"],
                metric=r["metric"],
                both_pass=r["both_pass"],
                a_pass_b_fail=r["a_pass_b_fail"],
                a_fail_b_pass=r["a_fail_b_pass"],
                both_fail=r["both_fail"],
                n_discordant=r["n_discordant"],
                a_pass_rate=r["a_pass_rate"],
                b_pass_rate=r["b_pass_rate"],
                chi2_statistic=r["chi2_statistic"],
                p_value=r["p_value"],
                p_value_corrected=corrected_p,
                correction_method=correction,
                n_comparisons=n_comparisons,
                threshold=threshold,
                significant_at_threshold=sig,
                method=r["method"],
            )
        )
    return results


def compute_significance_with_ci(
    states_by_model: dict[str, list[Any]],
    judge_results: list[Any] | None,
    *,
    metric: str = "latency_ms",
    test: Literal[
        "welch", "mann-whitney", "paired-t", "wilcoxon-signed"
    ] = "welch",
    correction: Literal["none", "bonferroni", "holm"] = "bonferroni",
    threshold: float = 0.05,
    compute_ci: bool = True,
    ci_level: float = 0.95,
    ci_method: str = "bca",
    n_resamples: int = 5000,
    seed: int | None = None,
) -> list[SignificanceResult]:
    """compute_pairwise_significance + optional bootstrap CIs on Cohen's d.

    When compute_ci is True, attaches a bootstrap CI on the effect size
    (Cohen's d) to each result via paired bootstrap of the d statistic.
    For paired tests the bootstrap samples paired observations; for
    independent tests it bootstraps both samples independently and
    recomputes d each resample.
    """
    results = compute_pairwise_significance(
        states_by_model,
        judge_results,
        metric=metric,
        test=test,
        correction=correction,
        threshold=threshold,
    )

    if not compute_ci or not results:
        return results

    is_paired = test in ("paired-t", "wilcoxon-signed")
    if is_paired:
        paired_by_model = _extract_paired_metric_samples(
            states_by_model, judge_results, metric
        )
    else:
        samples_by_model = _extract_metric_samples(
            states_by_model, judge_results, metric
        )

    rng = np.random.default_rng(seed)

    for sr in results:
        if sr.effect_size is None or sr.test_used in (
            "insufficient_samples",
            "zero_variance",
            "trivial",
        ):
            continue

        if is_paired:
            aligned_a, aligned_b = _align_paired_samples(
                paired_by_model.get(sr.model_a, {}),
                paired_by_model.get(sr.model_b, {}),
            )
            n_pairs = len(aligned_a)
            if n_pairs < 2:
                continue
            arr_a = np.asarray(aligned_a, dtype=float)
            arr_b = np.asarray(aligned_b, dtype=float)
            ds: list[float] = []
            for _ in range(n_resamples):
                idx = rng.integers(0, n_pairs, n_pairs)
                d = _cohens_d_numpy(arr_a[idx], arr_b[idx])
                if d is not None:
                    ds.append(d)
        else:
            arr_a = np.asarray(samples_by_model.get(sr.model_a, []), dtype=float)
            arr_b = np.asarray(samples_by_model.get(sr.model_b, []), dtype=float)
            if len(arr_a) < 2 or len(arr_b) < 2:
                continue
            ds = []
            for _ in range(n_resamples):
                ia = rng.integers(0, len(arr_a), len(arr_a))
                ib = rng.integers(0, len(arr_b), len(arr_b))
                d = _cohens_d_numpy(arr_a[ia], arr_b[ib])
                if d is not None:
                    ds.append(d)

        if len(ds) < 2:
            continue
        alpha = 1.0 - ci_level
        lower_pct = 100.0 * alpha / 2.0
        upper_pct = 100.0 * (1.0 - alpha / 2.0)
        lo = float(np.percentile(ds, lower_pct))
        hi = float(np.percentile(ds, upper_pct))
        if math.isfinite(lo) and math.isfinite(hi):
            sr.effect_size_ci_low = lo
            sr.effect_size_ci_high = hi
            sr.bootstrap_method = ci_method
            sr.bootstrap_resamples = n_resamples
            sr.bootstrap_seed = seed

    return results


def _cohens_d_numpy(a: np.ndarray, b: np.ndarray) -> float | None:
    """Cohen's d on numpy arrays (faster bootstrap inner loop)."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return None
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var == 0:
        return 0.0 if float(np.mean(a)) == float(np.mean(b)) else None
    return (float(np.mean(a)) - float(np.mean(b))) / math.sqrt(pooled_var)
