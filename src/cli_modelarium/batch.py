"""Multi-prompt batch mode.

Three concerns:

    1. Parsing & validation - load_batch_file() reads .txt or .json into
       BatchPrompt dataclasses with size-limit and structure checks.
    2. Cost-and-volume guards - check_batch_size_limits() rejects oversized
       batches before any API call is made.
    3. Orchestration - run_batch() builds StreamStates for every
       (prompt x system x model x temperature) tuple and runs them via the
       same `_call_with_retry` helper as the compare command, but with a
       Rich Progress bar instead of per-token Live panels.

Per-prompt system-prompt override (when a BatchPrompt has a `system` field)
takes precedence over command-line system prompts for that specific prompt
only - other prompts still use the command-line list.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from cli_modelarium.exceptions import (
    BatchSizeError,
    BatchValidationError,
    ModelariumError,
)
from cli_modelarium.io_safety import BATCH_INPUT_MAX_BYTES, safe_input_path
from cli_modelarium.models_registry import get_provider_for_model
from cli_modelarium.pricing import calculate_cost, is_local_model
from cli_modelarium.providers.base import BaseProvider
from cli_modelarium.streaming import (
    DEFAULT_MAX_RETRIES,
    StreamState,
    _call_with_retry,
)

# Safety caps. The build prompt sets these; --force-large bypasses both.
MAX_PROMPTS_PER_BATCH = 1000
MAX_TOTAL_CALLS = 10_000

# Cost estimation defaults. Assumed input/output tokens per call when we
# don't know the real shape yet. Deliberately on the high side - we'd rather
# refuse a borderline run than burn through somebody's quota.
ESTIMATE_INPUT_TOKENS = 500
ESTIMATE_OUTPUT_TOKENS = 500


@dataclass
class BatchPrompt:
    """One row of a batch input file."""

    id: str
    prompt: str
    system: str | None = None
    # Raw assertion dicts as-loaded; executed downstream by assertions.run_assertions.
    assertions: list[dict[str, Any]] = field(default_factory=list)


# ===== File parsing =====


def load_batch_file(file_path: str) -> list[BatchPrompt]:
    """Load a batch input file. Format is auto-detected from extension.

    Supported:
        .txt   - one prompt per non-blank, non-comment line
        .json  - top-level array of {"prompt": "...", "id"?: "...", ...}

    Files larger than `BATCH_INPUT_MAX_BYTES` are rejected by
    `safe_input_path` before any parsing happens.

    Raises:
        FileNotFoundError, ValueError: from safe_input_path.
        BatchValidationError: malformed content or unknown extension.
        json.JSONDecodeError: malformed JSON.
    """
    path = safe_input_path(file_path, max_size_bytes=BATCH_INPUT_MAX_BYTES)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _parse_txt(path)
    if suffix == ".json":
        return _parse_json(path)
    raise BatchValidationError(
        f"Cannot detect batch file format from extension {suffix!r}.\n"
        f"  Supported: .txt (one prompt per line), .json (array of objects).\n"
        f"  At: {path}"
    )


def _parse_txt(path: Path) -> list[BatchPrompt]:
    """One prompt per line. Lines starting with `#` are comments."""
    text = path.read_text(encoding="utf-8-sig")
    prompts: list[BatchPrompt] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # Comment line - inline `#` mid-line is NOT a comment marker
            # (would conflict with model output that legitimately contains `#`).
            continue
        prompts.append(BatchPrompt(id=f"p{len(prompts) + 1}", prompt=stripped))
    return prompts


def _parse_json(path: Path) -> list[BatchPrompt]:
    """Top-level JSON array of prompt objects."""
    raw = path.read_text(encoding="utf-8-sig")
    data = json.loads(raw)  # propagates JSONDecodeError verbatim
    if not isinstance(data, list):
        raise BatchValidationError(
            f"Batch JSON file must be an array at top level, "
            f"got {type(data).__name__!r}. At: {path}"
        )

    seen_ids: set[str] = set()
    prompts: list[BatchPrompt] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise BatchValidationError(
                f"Batch element #{i} is not an object (got {type(item).__name__!r}). At: {path}"
            )
        if "prompt" not in item:
            raise BatchValidationError(
                f"Batch element #{i} is missing required 'prompt' field. At: {path}"
            )
        prompt = item["prompt"]
        if not isinstance(prompt, str):
            raise BatchValidationError(f"Batch element #{i} 'prompt' must be a string. At: {path}")
        prompt_id = item.get("id") or f"p{i + 1}"
        if not isinstance(prompt_id, str):
            raise BatchValidationError(f"Batch element #{i} 'id' must be a string. At: {path}")
        if prompt_id in seen_ids:
            raise BatchValidationError(
                f"Batch contains duplicate prompt id {prompt_id!r}. At: {path}"
            )
        seen_ids.add(prompt_id)

        system = item.get("system")
        if system is not None and not isinstance(system, str):
            raise BatchValidationError(
                f"Batch element {prompt_id!r} 'system' must be a string. At: {path}"
            )

        assertions = item.get("assertions", [])
        if not isinstance(assertions, list):
            raise BatchValidationError(
                f"Batch element {prompt_id!r} 'assertions' must be a list. At: {path}"
            )

        prompts.append(
            BatchPrompt(
                id=prompt_id,
                prompt=prompt,
                system=system,
                assertions=list(assertions),
            )
        )
    return prompts


# ===== Size validation =====


def check_batch_size_limits(
    prompts: list[BatchPrompt],
    models: list[str],
    temperatures: list[float],
    command_system_prompts: list[str | None],
    *,
    force_large: bool = False,
) -> int:
    """Confirm the batch fits within the safety caps.

    Returns the total task count (useful for callers to display).

    Raises BatchSizeError when limits are exceeded and `force_large` is False.
    """
    if len(prompts) > MAX_PROMPTS_PER_BATCH and not force_large:
        raise BatchSizeError(
            f"Too many prompts: {len(prompts)} (max {MAX_PROMPTS_PER_BATCH}).\n"
            f"  Pass --force-large to bypass this safety cap."
        )

    total = _count_total_calls(prompts, models, temperatures, command_system_prompts)
    if total > MAX_TOTAL_CALLS and not force_large:
        raise BatchSizeError(
            f"Too many total API calls: {total} (max {MAX_TOTAL_CALLS}).\n"
            f"  Composition: {len(prompts)} prompts x {len(models)} models "
            f"x {len(temperatures)} temperatures (some prompts add their own system "
            f"prompts on top).\n"
            f"  Pass --force-large to bypass this safety cap, or reduce one "
            f"of the dimensions."
        )
    return total


def estimate_batch_cost(
    prompts: list[BatchPrompt],
    models: list[str],
    temperatures: list[float],
    command_system_prompts: list[str | None],
) -> float:
    """Return an upper-bound USD cost estimate for the batch.

    Assumes `ESTIMATE_INPUT_TOKENS` + `ESTIMATE_OUTPUT_TOKENS` per call.
    Unknown models are treated as $0 (we silently skip them in the estimate
    rather than failing - the actual run will fail more loudly when it
    reaches that model).
    """
    total = 0.0
    for bp in prompts:
        effective_sps = [bp.system] if bp.system else command_system_prompts
        for _sp in effective_sps:
            for model in models:
                for _temp in temperatures:
                    if is_local_model(model):
                        continue
                    try:
                        total += calculate_cost(
                            model,
                            input_tokens=ESTIMATE_INPUT_TOKENS,
                            output_tokens=ESTIMATE_OUTPUT_TOKENS,
                        )
                    except ModelariumError:
                        # Unknown model: skip in the estimate; real call surfaces it.
                        pass
    return total


def estimate_compare_cost(
    models: list[str],
    temperatures: list[float],
    system_prompts: list[str | None],
) -> float:
    """Upper-bound USD cost estimate for a compare run.

    Compare runs 1 prompt x M models x T temperatures x S system_prompts.
    Uses ESTIMATE_INPUT_TOKENS and ESTIMATE_OUTPUT_TOKENS as the per-call
    upper bound. Local models contribute $0. Unknown models are silently
    skipped (real call will fail at runtime if model truly invalid).

    Does NOT include judge cost (judge output length unknown until run).
    """
    total = 0.0
    for _sp in system_prompts:
        for model in models:
            if is_local_model(model):
                continue
            for _temp in temperatures:
                try:
                    total += calculate_cost(
                        model,
                        input_tokens=ESTIMATE_INPUT_TOKENS,
                        output_tokens=ESTIMATE_OUTPUT_TOKENS,
                    )
                except ModelariumError:
                    pass
    return total


def _count_total_calls(
    prompts: list[BatchPrompt],
    models: list[str],
    temperatures: list[float],
    command_system_prompts: list[str | None],
) -> int:
    """Count the total task count, accounting for per-prompt system overrides."""
    n_models = max(1, len(models))
    n_temps = max(1, len(temperatures))
    n_command_sps = max(1, len(command_system_prompts))
    total = 0
    for bp in prompts:
        n_sps = 1 if bp.system else n_command_sps
        total += n_sps * n_models * n_temps
    return total


# ===== Orchestration =====


def build_batch_states(
    prompts: list[BatchPrompt],
    models: list[str],
    temperatures: list[float],
    command_system_prompts: list[str | None],
) -> list[tuple[StreamState, BatchPrompt]]:
    """Build StreamStates for every (prompt x system x model x temperature) tuple.

    Each returned tuple pairs the state with its source BatchPrompt so the
    caller can look up the prompt text and assertions later.

    Per-prompt `system` overrides win for that prompt only; other prompts
    use the command-line `command_system_prompts` list.
    """
    pairs: list[tuple[StreamState, BatchPrompt]] = []
    for bp in prompts:
        effective_sps = [bp.system] if bp.system else command_system_prompts
        for sp in effective_sps:
            for model in models:
                provider_name = get_provider_for_model(model)
                for temperature in temperatures:
                    pairs.append(
                        (
                            StreamState(
                                model=model,
                                provider_name=provider_name,
                                temperature=temperature,
                                system_prompt=sp,
                            ),
                            bp,
                        )
                    )
    return pairs


async def run_batch(
    *,
    pairs: list[tuple[StreamState, BatchPrompt]],
    provider_factory: Callable[[str], BaseProvider],
    console: Console,
    concurrency: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
    show_progress: bool = True,
    sleep: Callable[[float], asyncio.Future[None]] = asyncio.sleep,
) -> list[tuple[StreamState, BatchPrompt]]:
    """Run every (state, prompt) pair in parallel under per-provider semaphores.

    Reuses `streaming._call_with_retry` so the 429/529 retry behavior matches
    the compare command exactly. The only difference is the display: a Rich
    Progress bar showing "Completed X/Y" instead of per-task Live panels.

    Returns the same `pairs` list (states mutated in place) for convenience.
    """
    provider_names = sorted({s.provider_name for s, _ in pairs})
    instances: dict[str, BaseProvider] = {name: provider_factory(name) for name in provider_names}
    semaphores: dict[str, asyncio.Semaphore] = {
        name: asyncio.Semaphore(concurrency) for name in provider_names
    }

    progress: Progress | None = None
    task_id = None
    if show_progress and pairs:
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        )
        task_id = progress.add_task(
            f"Running {len(pairs)} call{'s' if len(pairs) != 1 else ''}",
            total=len(pairs),
        )

    async def _run_one(state: StreamState, bp: BatchPrompt) -> None:
        provider = instances[state.provider_name]
        async with semaphores[state.provider_name]:
            state.mark_started()
            try:
                result = await _call_with_retry(
                    provider=provider,
                    state=state,
                    prompt=bp.prompt,
                    max_retries=max_retries,
                    sleep=sleep,
                )
                state.mark_complete(result)
            except ModelariumError as e:
                state.mark_error(str(e))
            except Exception as e:  # noqa: BLE001 - become an error row, not a crash
                state.mark_error(f"unexpected: {e}")
            finally:
                if progress is not None and task_id is not None:
                    progress.advance(task_id)

    coro = asyncio.gather(*[_run_one(s, bp) for s, bp in pairs])

    if progress is not None:
        with progress:
            await coro
    else:
        await coro

    return pairs


# ===== Filename safety =====

# Detect output formats from common extensions.
_EXTENSION_TO_FORMAT = {
    ".csv": "csv",
    ".json": "json",
    ".md": "markdown",
    ".markdown": "markdown",
}


def detect_output_format(path: Path) -> str | None:
    """Infer 'csv' / 'json' / 'markdown' from a file extension. None if unknown."""
    return _EXTENSION_TO_FORMAT.get(path.suffix.lower())


def output_overlaps_input(input_path: Path, output_path: Path) -> bool:
    """True if `output_path` would overwrite `input_path` (same resolved file)."""
    try:
        return input_path.resolve() == output_path.resolve()
    except OSError:
        return False
