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
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

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
    test: Literal["welch", "mann-whitney"] = "welch",
    correction: Literal["none", "bonferroni", "holm"] = "bonferroni",
    threshold: float = 0.05,
) -> list[SignificanceResult]:
    """All pairwise significance tests between models, with correction.

    Returns one SignificanceResult per unordered (model_a, model_b) pair.
    Pair order follows insertion order in states_by_model so output is
    deterministic.

    Edge cases:
      - n < 3 in either group: test_used="insufficient_samples", p=None
      - both groups zero-variance, same mean: test_used="trivial", p=1.0
      - both groups zero-variance, different means: test_used="zero_variance", p=None
      - judge_results is None and metric="score": empty samples per model
    """
    models = list(states_by_model.keys())
    if len(models) < 2:
        return []

    samples_by_model = _extract_metric_samples(states_by_model, judge_results, metric)

    pairs: list[tuple[str, str]] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            pairs.append((model_a, model_b))
    n_comparisons = len(pairs)

    raw_results: list[dict[str, Any]] = []
    raw_p_values: list[float] = []

    for model_a, model_b in pairs:
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
