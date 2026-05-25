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
from cli_modelarium.batch import BatchPrompt
from cli_modelarium.pricing import PRICING_AS_OF, is_local_model
from cli_modelarium.streaming import StreamState

# Canonical CSV column order. Pinned by tests so downstream pipelines can
# rely on this layout.
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
)


@dataclass
class BatchResult:
    """A single batch row - the union of a StreamState and a BatchPrompt."""

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
    # Phase 9 will execute these and surface pass/fail. For now we carry
    # them through so the JSON / CSV output round-trips the user's input.
    assertions_raw: list[dict[str, Any]] = field(default_factory=list)


def state_to_result(state: StreamState, bp: BatchPrompt) -> BatchResult:
    """Convert a StreamState + its source BatchPrompt into a BatchResult."""
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
    )


# ===== CSV =====


def _format_csv(results: list[BatchResult]) -> str:
    """Build the full CSV text. Output field newlines are escaped to literal \\n
    so cells don't break spreadsheet imports.
    """
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
    writer.writeheader()
    for r in results:
        writer.writerow(
            {
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
            }
        )
    return buf.getvalue()


def write_csv(results: list[BatchResult], output_path: Path) -> None:
    """Atomic write of `results` to `output_path` as CSV."""
    atomic_write_bytes(output_path, _format_csv(results).encode("utf-8"))


def _csv_escape(text: str) -> str:
    """Escape newlines in a cell so the CSV stays one row per record."""
    if not text:
        return ""
    return text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\r")


def _none_or(value: float | None) -> Any:
    """Render None as the empty string in CSV (vs the literal 'None')."""
    return "" if value is None else value


# ===== JSON =====


def _format_json(results: list[BatchResult]) -> str:
    """Build the JSON payload string with metadata header + results array."""
    total_cost = sum(r.cost_usd for r in results if r.error is None)
    payload = {
        "version": __version__,
        "pricing_as_of": PRICING_AS_OF,
        "total_cost_usd": total_cost,
        "total_results": len(results),
        "failed_results": sum(1 for r in results if r.error),
        "results": [_result_to_dict(r) for r in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_json(results: list[BatchResult], output_path: Path) -> None:
    """Atomic write of `results` to `output_path` as JSON."""
    atomic_write_bytes(output_path, _format_json(results).encode("utf-8"))


def _result_to_dict(r: BatchResult) -> dict[str, Any]:
    return {
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


# ===== Markdown =====


def _format_markdown(results: list[BatchResult]) -> str:
    """Render results as Markdown grouped by prompt_id.

    Each prompt gets its own H2 section with a sub-table of
    (model, temperature, TTFT, latency, cost, status) rows; outputs follow
    the table in code-block form for readability.
    """
    if not results:
        return (
            f"# Cli Modelarium batch results\n\n"
            f"_Pricing data as of {PRICING_AS_OF}._\n\n"
            f"No results - the batch was empty.\n"
        )

    total_cost = sum(r.cost_usd for r in results if r.error is None)
    failures = sum(1 for r in results if r.error)
    lines: list[str] = [
        "# Cli Modelarium batch results",
        "",
        f"- Version: {__version__}",
        f"- Pricing data as of: {PRICING_AS_OF}",
        f"- Total cost: ${total_cost:.6f}",
        f"- Results: {len(results)} ({failures} failed)",
        "",
    ]

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
        lines.append(
            "| Model | Temp | TTFT (ms) | Latency (ms) | In | Out | Cost | Status |"
        )
        lines.append(
            "|-------|-----:|----------:|-------------:|---:|----:|-----:|:-------|"
        )
        for r in items:
            ttft = f"{r.ttft_ms:.1f}" if r.ttft_ms is not None else "-"
            latency = f"{r.latency_ms:.1f}" if r.latency_ms is not None else "-"
            cost = "Free" if is_local_model(r.model) else f"${r.cost_usd:.6f}"
            status = "ok" if r.error is None else "error"
            lines.append(
                f"| `{r.model}` | {r.temperature:.1f} | {ttft} | {latency} | "
                f"{r.input_tokens} | {r.output_tokens} | {cost} | {status} |"
            )
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

    return "\n".join(lines)


def write_markdown(results: list[BatchResult], output_path: Path) -> None:
    """Atomic write of `results` to `output_path` as Markdown."""
    atomic_write_bytes(output_path, _format_markdown(results).encode("utf-8"))


def render_markdown_to_console(
    results: list[BatchResult], console: Console
) -> None:
    """Render Markdown via Rich for stdout display."""
    console.print(Markdown(_format_markdown(results)))


def _md_escape(text: str) -> str:
    """Escape Markdown special characters so user text doesn't accidentally format."""
    # Limit to characters that would break a single-line table cell.
    if not text:
        return ""
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " / ")
    )


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
