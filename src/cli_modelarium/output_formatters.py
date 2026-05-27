"""Output writers for batch mode: CSV, JSON, Markdown.

All writers are atomic: they write to `<path>.tmp` and `os.replace` into
place. A SIGINT during write leaves the original file (or no file) intact
rather than a half-written one.

`BatchResult` is the unified data shape - built from a `StreamState` plus
its source `BatchPrompt`. The CSV column order is the spec'd canonical
order and tests pin it explicitly.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from cli_modelarium import __version__
from cli_modelarium.assertions import (
    ERROR_MARK,
    FAIL_MARK,
    PASS_MARK,
    AssertionResult,
    count_passed,
    failed_types,
    format_assertion_message,
)
from cli_modelarium.batch import BatchPrompt
from cli_modelarium.judging import JudgeResult
from cli_modelarium.pricing import PRICING_AS_OF, is_local_model
from cli_modelarium.streaming import StreamState

# Canonical CSV column order. Pinned by tests so downstream pipelines can
# rely on this layout. Judge and assertion columns are appended at the end
# so existing integrations that ignore unknown columns keep working.
CSV_COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "prompt",
    "system",
    "model",
    "temperature",
    "latency_ms",
    "ttft_ms",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cost_usd",
    "output",
    "error",
    "retries",
    "judge_score_avg",
    "judge_score_std",
    "judge_count",
    "hallucination_risk",
    "assertions_passed",
    "assertions_total",
    "assertions_failed_types",
)


@dataclass
class BatchResult:
    """A single batch row - the union of a StreamState and a BatchPrompt,
    optionally enriched with judge scores (Phase 8) and assertion results (Phase 9).
    """

    prompt_id: str
    prompt: str
    system: str | None
    model: str
    temperature: float
    latency_ms: float | None
    ttft_ms: float | None
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    output: str
    error: str | None
    retries: int
    # The raw assertion configs from the user's input file. Always preserved
    # so JSON round-trips the user's intent even when assertions were skipped.
    assertions_raw: list[dict[str, Any]] = field(default_factory=list)
    # Phase 8 judging: None when judging wasn't requested for this batch.
    judge_result: JudgeResult | None = None
    # Phase 9 assertions: None when assertion execution was skipped
    # (--no-assertions, or failed main call); empty list when the prompt
    # simply had no assertions configured.
    assertion_results: list[AssertionResult] | None = None
    # v0.1.1: 0 for single-run/batch flows (the default); 0..N-1 for the
    # compare command's --runs N flag. Only emitted in CSV/JSON/Markdown
    # when the surrounding `runs` parameter > 1.
    run_index: int = 0


def state_to_result(
    state: StreamState,
    bp: BatchPrompt,
    judge_result: JudgeResult | None = None,
    assertion_results: list[AssertionResult] | None = None,
) -> BatchResult:
    """Convert a StreamState + its source BatchPrompt (and optional Judge/Assertion
    results) into a BatchResult.
    """
    return BatchResult(
        prompt_id=bp.id,
        prompt=bp.prompt,
        system=state.system_prompt,
        model=state.model,
        temperature=state.temperature,
        latency_ms=state.latency_ms,
        ttft_ms=state.ttft_ms,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        cached_tokens=state.cached_tokens,
        cost_usd=state.cost_usd,
        output=state.text,
        error=state.error,
        retries=state.attempts,
        assertions_raw=list(bp.assertions),
        judge_result=judge_result,
        assertion_results=assertion_results,
        run_index=state.run_index,
    )


def _judge_cell_avg(r: BatchResult) -> Any:
    """Render the judge_score_avg field for tabular output. Empty if no judging."""
    if r.judge_result is None or r.judge_result.average_score is None:
        return ""
    return round(r.judge_result.average_score, 2)


def _judge_cell_std(r: BatchResult) -> Any:
    if r.judge_result is None or r.judge_result.std_dev is None:
        return ""
    return round(r.judge_result.std_dev, 3)


def _judge_count(r: BatchResult) -> int:
    if r.judge_result is None:
        return 0
    return sum(1 for j in r.judge_result.judges if j.score is not None)


def _hallucination_risk_cell(r: BatchResult) -> str:
    """Render the hallucination_risk column for tabular output. Empty when not in
    hallucination mode (i.e., no judge had a risk_level).
    """
    if r.judge_result is None:
        return ""
    if r.judge_result.aggregated_risk_level is None:
        return ""
    return r.judge_result.aggregated_risk_level


def _assertion_passed_cell(r: BatchResult) -> Any:
    """Render assertions_passed for tabular output. Empty when assertions not run."""
    if r.assertion_results is None:
        return ""
    passed, _ = count_passed(r.assertion_results)
    return passed


def _assertion_total_cell(r: BatchResult) -> Any:
    """Render assertions_total. Empty when not run.

    Uses count_passed's denominator (excludes 'error' rows) so the
    pass/total ratio is interpretable.
    """
    if r.assertion_results is None:
        return ""
    _, total = count_passed(r.assertion_results)
    return total


def _assertion_failed_types_cell(r: BatchResult) -> str:
    """Semicolon-separated list of failed assertion types for CI grep."""
    if r.assertion_results is None:
        return ""
    return ";".join(failed_types(r.assertion_results))


# ===== CSV =====


def _format_csv(results: list[BatchResult], runs: int = 1) -> str:
    """Build the full CSV text. Output field newlines are escaped to literal \\n
    so cells don't break spreadsheet imports.

    When `runs > 1` an additional `run_index` column is appended to every
    row. When `runs == 1` (the default) the column set is byte-identical
    to v0.1.0.
    """
    fieldnames = list(CSV_COLUMNS)
    if runs > 1:
        fieldnames.append("run_index")
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        row = {
            "prompt_id": r.prompt_id,
            "prompt": _csv_escape(r.prompt),
            "system": _csv_escape(r.system or ""),
            "model": r.model,
            "temperature": r.temperature,
            "latency_ms": _none_or(r.latency_ms),
            "ttft_ms": _none_or(r.ttft_ms),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cached_tokens": r.cached_tokens,
            "cost_usd": r.cost_usd,
            "output": _csv_escape(r.output),
            "error": _csv_escape(r.error or ""),
            "retries": r.retries,
            "judge_score_avg": _judge_cell_avg(r),
            "judge_score_std": _judge_cell_std(r),
            "judge_count": _judge_count(r),
            "hallucination_risk": _hallucination_risk_cell(r),
            "assertions_passed": _assertion_passed_cell(r),
            "assertions_total": _assertion_total_cell(r),
            "assertions_failed_types": _assertion_failed_types_cell(r),
        }
        if runs > 1:
            row["run_index"] = r.run_index
        writer.writerow(row)
    return buf.getvalue()


def write_csv(results: list[BatchResult], output_path: Path, runs: int = 1) -> None:
    """Atomic write of `results` to `output_path` as CSV."""
    atomic_write_bytes(output_path, _format_csv(results, runs=runs).encode("utf-8"))


def _csv_escape(text: str) -> str:
    """Escape newlines in a cell so the CSV stays one row per record."""
    if not text:
        return ""
    return text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\r")


def _none_or(value: float | None) -> Any:
    """Render None as the empty string in CSV (vs the literal 'None')."""
    return "" if value is None else value


# ===== JSON =====


def _format_json(
    results: list[BatchResult],
    runs: int = 1,
    significance_results: list | None = None,
) -> str:
    """Build the JSON payload string with metadata header + results array.

    When judging is enabled (any result has a judge_result), include
    `judge_cost_usd` and `total_cost_usd_with_judges`. When assertions ran
    (any result has assertion_results), include `total_assertions`,
    `total_assertions_passed`, `pass_rate`.

    When `runs > 1`, additive fields appear:
        * top-level `total_runs`
        * top-level `stats_by_cell` array (one entry per cell)
        * each result record gains a `run_index`
    When `runs == 1`, the schema is byte-identical to v0.1.0.

    When `significance_results` is provided and non-empty, a
    `significance_tests` array is added. When it's None/empty, no new
    key appears - so output stays byte-identical for callers who don't
    use significance.
    """
    total_cost = sum(r.cost_usd for r in results if r.error is None)
    judge_cost = sum(
        j.cost_usd for r in results if r.judge_result is not None for j in r.judge_result.judges
    )
    has_judges = any(r.judge_result is not None for r in results)
    has_assertions = any(r.assertion_results is not None for r in results)

    payload: dict[str, Any] = {
        "version": __version__,
        "pricing_as_of": PRICING_AS_OF,
        "total_cost_usd": total_cost,
        "total_results": len(results),
        "failed_results": sum(1 for r in results if r.error),
        "results": [_result_to_dict(r, include_run_index=runs > 1) for r in results],
    }
    if runs > 1:
        payload["total_runs"] = runs
        payload["stats_by_cell"] = _build_stats_by_cell(results)
    if has_judges:
        payload["judge_cost_usd"] = judge_cost
        payload["total_cost_usd_with_judges"] = total_cost + judge_cost
    if has_assertions:
        total_passed = 0
        total_definitive = 0
        for r in results:
            if r.assertion_results is None:
                continue
            p, d = count_passed(r.assertion_results)
            total_passed += p
            total_definitive += d
        payload["total_assertions"] = total_definitive
        payload["total_assertions_passed"] = total_passed
        payload["pass_rate"] = total_passed / total_definitive if total_definitive else 1.0
    if significance_results:
        payload["significance_tests"] = [
            {
                "model_a": r.model_a,
                "model_b": r.model_b,
                "metric": r.metric,
                "n_a": r.n_a,
                "n_b": r.n_b,
                "mean_a": r.mean_a,
                "mean_b": r.mean_b,
                "stdev_a": r.stdev_a,
                "stdev_b": r.stdev_b,
                "test_used": r.test_used,
                "test_statistic": r.test_statistic,
                "degrees_of_freedom": r.degrees_of_freedom,
                "p_value": r.p_value,
                "p_value_corrected": r.p_value_corrected,
                "correction_method": r.correction_method,
                "n_comparisons": r.n_comparisons,
                "effect_size": r.effect_size,
                "effect_size_metric": "cohens_d",
                "effect_size_interpretation": r.effect_size_interpretation,
                "threshold": r.threshold,
                "significant_at_threshold": r.significant_at_threshold,
            }
            for r in significance_results
        ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_json(
    results: list[BatchResult],
    output_path: Path,
    runs: int = 1,
    significance_results: list | None = None,
) -> None:
    """Atomic write of `results` to `output_path` as JSON."""
    atomic_write_bytes(
        output_path,
        _format_json(
            results, runs=runs, significance_results=significance_results
        ).encode("utf-8"),
    )


def _build_stats_by_cell(results: list[BatchResult]) -> list[dict[str, Any]]:
    """Aggregate per-cell stats for JSON output when runs > 1.

    A "cell" is the (model, temperature, system) tuple. Iteration order
    follows first-seen order in `results`, so the output is deterministic.
    """
    grouped: dict[tuple[str, float, str | None], list[BatchResult]] = {}
    insertion_order: list[tuple[str, float, str | None]] = []
    for r in results:
        key = (r.model, r.temperature, r.system)
        if key not in grouped:
            grouped[key] = []
            insertion_order.append(key)
        grouped[key].append(r)

    out: list[dict[str, Any]] = []
    for key in insertion_order:
        cell = grouped[key]
        model, temp, sp = key
        latencies = [r.latency_ms for r in cell if r.error is None and r.latency_ms is not None]
        token_counts = [r.output_tokens for r in cell if r.error is None]
        costs = [r.cost_usd for r in cell if r.error is None]
        outputs = [r.output for r in cell if r.error is None]
        n_succeeded = sum(1 for r in cell if r.error is None)
        n_failed = sum(1 for r in cell if r.error is not None)

        if len(latencies) >= 2:
            stdev = _safe_stdev(latencies)
            mean = _safe_mean(latencies)
            cv = stdev / mean if mean and stdev is not None else None
        else:
            stdev = None
            cv = None

        from collections import Counter as _Counter

        counter = _Counter(outputs)
        unique = len(counter)
        if unique < n_succeeded and counter:
            mode_output, mode_count = counter.most_common(1)[0]
        else:
            mode_output, mode_count = None, 0

        out.append(
            {
                "model": model,
                "temperature": temp,
                "system": sp,
                "n_runs": len(cell),
                "n_succeeded": n_succeeded,
                "n_failed": n_failed,
                "latency_mean_ms": _safe_mean(latencies),
                "latency_stdev_ms": stdev,
                "latency_cv": cv,
                "output_tokens_mean": _safe_mean(token_counts) if token_counts else None,
                "cost_total_usd": sum(costs),
                "unique_outputs": unique,
                "mode_output": mode_output,
                "mode_count": mode_count,
                "output_diversity": (unique / n_succeeded) if n_succeeded else 0.0,
            }
        )
    return out


def _safe_mean(values: list[float]) -> float | None:
    import statistics as _s

    return _s.mean(values) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    import statistics as _s

    return _s.stdev(values) if len(values) >= 2 else None


def _result_to_dict(r: BatchResult, include_run_index: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prompt_id": r.prompt_id,
        "prompt": r.prompt,
        "system": r.system,
        "model": r.model,
        "temperature": r.temperature,
        "latency_ms": r.latency_ms,
        "ttft_ms": r.ttft_ms,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cached_tokens": r.cached_tokens,
        "cost_usd": r.cost_usd,
        "output": r.output,
        "error": r.error,
        "retries": r.retries,
        "assertions": r.assertions_raw,
    }
    if include_run_index:
        out["run_index"] = r.run_index
    if r.judge_result is not None:
        out["judges"] = [
            {
                "model": j.model,
                "score": j.score,
                "reasoning": j.reasoning,
                "cost_usd": j.cost_usd,
                "latency_ms": j.latency_ms,
                "parse_error": j.parse_error,
                # Always include risk_level; will be None for non-hallucination
                # judging which keeps the schema predictable for downstream tools.
                "risk_level": j.risk_level,
            }
            for j in r.judge_result.judges
        ]
        out["judge_score_avg"] = r.judge_result.average_score
        out["judge_score_std"] = r.judge_result.std_dev
        out["judge_skipped"] = list(r.judge_result.skipped_models)
        if r.judge_result.aggregated_risk_level is not None:
            out["hallucination_risk"] = r.judge_result.aggregated_risk_level
    if r.assertion_results is not None:
        # Replace the raw config carryover with the executed results.
        passed, total = count_passed(r.assertion_results)
        out["assertions"] = [
            {
                "type": a.type,
                "passed": a.passed,
                "expected": a.expected,
                "actual": a.actual,
                "message": a.message,
                "error": a.error,
            }
            for a in r.assertion_results
        ]
        out["assertions_passed"] = passed
        out["assertions_total"] = total
    return out


# ===== Markdown =====


def _format_markdown(results: list[BatchResult], runs: int = 1) -> str:
    """Render results as Markdown grouped by prompt_id.

    Each prompt gets its own H2 section with a sub-table of
    (model, temperature, TTFT, latency, cost, status) rows; outputs follow
    the table in code-block form for readability.

    When `runs > 1`, a "Per-cell statistical summary" section is appended
    after the per-prompt sections. When `runs == 1`, the output is
    byte-identical to v0.1.0.
    """
    if not results:
        return (
            f"# Cli Modelarium batch results\n\n"
            f"_Pricing data as of {PRICING_AS_OF}._\n\n"
            f"No results - the batch was empty.\n"
        )

    total_cost = sum(r.cost_usd for r in results if r.error is None)
    failures = sum(1 for r in results if r.error)
    has_judges = any(r.judge_result is not None for r in results)
    has_assertions = any(r.assertion_results is not None for r in results)
    # Detect hallucination mode from the data: any judge has a risk_level
    # set, which means parse_hallucination_response was used.
    has_hallucination = any(
        r.judge_result is not None and any(j.risk_level for j in r.judge_result.judges)
        for r in results
    )
    judge_cost = sum(
        j.cost_usd for r in results if r.judge_result is not None for j in r.judge_result.judges
    )

    lines: list[str] = [
        "# Cli Modelarium batch results",
        "",
        f"- Version: {__version__}",
        f"- Pricing data as of: {PRICING_AS_OF}",
        f"- Total cost: ${total_cost:.6f}",
    ]
    if has_judges:
        lines.append(f"- Judge cost: ${judge_cost:.6f}")
        lines.append(f"- Combined cost: ${total_cost + judge_cost:.6f}")
    if has_assertions:
        total_passed = 0
        total_definitive = 0
        for r in results:
            if r.assertion_results is None:
                continue
            p, d = count_passed(r.assertion_results)
            total_passed += p
            total_definitive += d
        pass_rate = total_passed / total_definitive if total_definitive else 1.0
        lines.append(
            f"- Assertions: {total_passed}/{total_definitive} ({pass_rate * 100:.1f}% pass rate)"
        )
    lines.append(f"- Results: {len(results)} ({failures} failed)")
    lines.append("")

    # Group results by prompt_id, preserving submission order.
    grouped: dict[str, list[BatchResult]] = {}
    for r in results:
        grouped.setdefault(r.prompt_id, []).append(r)

    for prompt_id, items in grouped.items():
        first = items[0]
        lines.append(f"## {prompt_id}")
        lines.append("")
        lines.append(f"**Prompt:** {_md_escape(first.prompt)}")
        if first.system:
            lines.append("")
            lines.append(f"**System (default):** {_md_escape(first.system)}")
        lines.append("")

        header_cols = ["Model", "Temp", "TTFT (ms)", "Latency (ms)", "In", "Out", "Cost"]
        align_cols = [
            "-------",
            "-----:",
            "----------:",
            "-------------:",
            "---:",
            "----:",
            "-----:",
        ]
        if has_judges:
            header_cols.append("Hallucination Risk" if has_hallucination else "Score")
            align_cols.append(":------")
        if has_assertions:
            header_cols.append("Assertions")
            align_cols.append(":---------")
        header_cols.append("Status")
        align_cols.append(":-------")
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(align_cols) + "|")

        for r in items:
            ttft = f"{r.ttft_ms:.1f}" if r.ttft_ms is not None else "-"
            latency = f"{r.latency_ms:.1f}" if r.latency_ms is not None else "-"
            cost = "Free" if is_local_model(r.model) else f"${r.cost_usd:.6f}"
            status = "ok" if r.error is None else "error"
            cells = [
                f"`{r.model}`",
                f"{r.temperature:.1f}",
                ttft,
                latency,
                str(r.input_tokens),
                str(r.output_tokens),
                cost,
            ]
            if has_judges:
                if has_hallucination:
                    cells.append(_hallucination_summary_cell(r))
                else:
                    cells.append(_judge_summary_cell(r))
            if has_assertions:
                cells.append(_assertion_summary_cell(r))
            cells.append(status)
            lines.append("| " + " | ".join(cells) + " |")

            # Failed-assertion details below the row, so the user can see
            # WHICH assertion failed without opening the JSON.
            failure_lines = _assertion_failure_lines(r)
            if failure_lines:
                lines.append("")
                lines.extend(failure_lines)
        lines.append("")

        # Per-row outputs in fenced code blocks.
        for r in items:
            tag = f"{r.model} @ {r.temperature:.1f}"
            lines.append(f"**{tag}:**")
            lines.append("")
            if r.error:
                lines.append(f"> error: {r.error}")
            else:
                lines.append("```")
                lines.append(r.output if r.output else "(no output)")
                lines.append("```")
            lines.append("")

    if runs > 1:
        lines.append("## Per-cell statistical summary")
        lines.append("")
        lines.append(
            "| Model | Temp | OK/Fail | Latency mean ± stdev (ms) | CV | "
            "Tokens mean | Cost total | Diversity | Mode |"
        )
        lines.append("|-------|-----:|--------:|--------------------------:|---:|"
                     "------------:|-----------:|----------:|------|")
        for cell in _build_stats_by_cell(results):
            latency_cell = "-"
            if cell["latency_mean_ms"] is not None and cell["latency_stdev_ms"] is not None:
                latency_cell = (
                    f"{cell['latency_mean_ms']:.0f} ± {cell['latency_stdev_ms']:.0f}"
                )
            elif cell["latency_mean_ms"] is not None:
                latency_cell = f"{cell['latency_mean_ms']:.0f}"
            cv_cell = f"{cell['latency_cv']:.3f}" if cell["latency_cv"] is not None else "-"
            tokens_cell = (
                f"{cell['output_tokens_mean']:.0f}"
                if cell["output_tokens_mean"] is not None
                else "-"
            )
            if cell["mode_output"] is None:
                mode_cell = "(no mode)"
            else:
                preview = cell["mode_output"].replace("\n", " ").strip()
                if len(preview) > 40:
                    preview = preview[:37] + "..."
                mode_cell = f"`{preview}` ({cell['mode_count']}x)"
            row = (
                f"| `{cell['model']}` "
                f"| {cell['temperature']:.1f} "
                f"| {cell['n_succeeded']}/{cell['n_failed']} "
                f"| {latency_cell} "
                f"| {cv_cell} "
                f"| {tokens_cell} "
                f"| ${cell['cost_total_usd']:.6f} "
                f"| {cell['output_diversity']:.2f} "
                f"| {mode_cell} |"
            )
            lines.append(row)
        lines.append("")
        lines.append(
            "_Coefficient of variation (CV) < 0.05 indicates stable model behavior._"
        )
        lines.append("")

    return "\n".join(lines)


def write_markdown(results: list[BatchResult], output_path: Path, runs: int = 1) -> None:
    """Atomic write of `results` to `output_path` as Markdown."""
    atomic_write_bytes(output_path, _format_markdown(results, runs=runs).encode("utf-8"))


def render_markdown_to_console(
    results: list[BatchResult], console: Console, runs: int = 1
) -> None:
    """Render Markdown via Rich for stdout display."""
    console.print(Markdown(_format_markdown(results, runs=runs)))


def _assertion_summary_cell(r: BatchResult) -> str:
    """Render the markdown Assertions cell for one row.

    Examples:
        "5/5 ✓"  - all definitive results passed
        "3/5 ✗"  - some failed (caller can see details in the failure list below)
        "-"      - assertions didn't run (e.g., main call failed, or no assertions)
        "⚠ N/M"  - all rows errored (couldn't run, e.g., missing jsonschema)
    """
    if r.assertion_results is None or not r.assertion_results:
        return "-"
    passed, total = count_passed(r.assertion_results)
    if total == 0:
        # All assertions errored out (e.g., jsonschema not installed).
        return f"{ERROR_MARK} 0/{len(r.assertion_results)}"
    mark = PASS_MARK if passed == total else FAIL_MARK
    return f"{passed}/{total} {mark}"


def _assertion_failure_lines(r: BatchResult) -> list[str]:
    """Per-assertion bullets for the row's failures and errors.

    Always shows failing / errored assertions inline. Passing assertions
    are NOT listed here - they're summarized in the row's cell and the
    JSON has the full breakdown for downstream tools.
    """
    if r.assertion_results is None:
        return []
    interesting = [a for a in r.assertion_results if not a.passed]
    if not interesting:
        return []
    out: list[str] = []
    for a in interesting:
        out.append(f"- {format_assertion_message(a)}")
    return out


def _hallucination_summary_cell(r: BatchResult) -> str:
    """Render the markdown Hallucination Risk cell.

    Single judge: "Low (8)".
    Panel: "High [gpt-5.5: High (3), claude: Medium (5)]".
    No data: "-".
    """
    if r.judge_result is None or not r.judge_result.judges:
        return "-"
    successful = [j for j in r.judge_result.judges if j.score is not None and j.risk_level]
    if not successful:
        return "N/A"

    risk = r.judge_result.aggregated_risk_level or "?"

    if len(successful) == 1 and len(r.judge_result.judges) == 1:
        s = successful[0]
        return f"{s.risk_level} ({s.score})"

    breakdown = ", ".join(
        f"{j.model.split('/')[-1]}: {j.risk_level or '?'} "
        f"({j.score if j.score is not None else 'X'})"
        for j in r.judge_result.judges
    )
    return f"{risk} [{breakdown}]"


def _judge_summary_cell(r: BatchResult) -> str:
    """Render the markdown Score cell for one row.

    Single judge: "8"
    Panel:        "Avg 7.3 [gpt-5.5: 8, claude: 7]"
    No judge:     "-"  (e.g., main call failed)
    """
    if r.judge_result is None or not r.judge_result.judges:
        if r.judge_result is not None and r.judge_result.skipped_models:
            return f"- (skipped self-eval: {', '.join(r.judge_result.skipped_models)})"
        return "-"

    judges = r.judge_result.judges
    successful = [j for j in judges if j.score is not None]
    if not successful:
        return "N/A (judge parse failed)"

    if len(successful) == 1 and len(judges) == 1:
        return str(successful[0].score)

    avg = r.judge_result.average_score
    breakdown = ", ".join(
        f"{j.model.split('/')[-1]}: {j.score if j.score is not None else 'X'}" for j in judges
    )
    return f"Avg {avg:.1f} [{breakdown}]" if avg is not None else f"[{breakdown}]"


def _md_escape(text: str) -> str:
    """Escape Markdown special characters so user text doesn't accidentally format."""
    # Limit to characters that would break a single-line table cell.
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " / ")


# ===== Atomic write helper =====


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically.

    Writes to a sibling `<name>.tmp` first then `os.replace`s into place.
    `os.replace` is atomic on POSIX and on Windows (Python 3.3+). If a
    failure happens between write and replace, the `.tmp` file is cleaned up.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        # Including BaseException so KeyboardInterrupt cleans up too.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise
