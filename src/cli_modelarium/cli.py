"""Click CLI entry point for Cli Modelarium."""
from __future__ import annotations

import asyncio
import sys
from typing import Any

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from cli_modelarium import __version__
from cli_modelarium.batch import (
    ESTIMATE_INPUT_TOKENS,
    ESTIMATE_OUTPUT_TOKENS,
    build_batch_states,
    check_batch_size_limits,
    detect_output_format,
    estimate_batch_cost,
    load_batch_file,
    output_overlaps_input,
    run_batch,
)
from cli_modelarium.exceptions import (
    BatchSizeError,
    BatchValidationError,
    CostLimitExceededError,
    KeyNotConfiguredError,
    ModelariumError,
    OutputFormatError,
    ProviderError,
    UnknownModelError,
    UnknownProviderError,
)
from cli_modelarium.assertions import (
    AssertionResult,
    count_failed,
    count_passed,
    run_assertions,
)
from cli_modelarium.io_safety import load_system_prompt
from cli_modelarium.judging import (
    DEFAULT_CRITERIA,
    JUDGE_PROMPT_TEMPLATE,
    JudgeResult,
    print_tos_disclosure,
    run_judging,
    total_judge_calls,
    total_judge_cost,
)
from cli_modelarium.output_formatters import (
    render_markdown_to_console,
    state_to_result,
    write_csv,
    write_json,
    write_markdown,
)
from cli_modelarium.models_registry import (
    all_known_providers,
    list_models_for_provider,
    parse_models_arg,
)
from cli_modelarium.pricing import (
    PRICING,
    PRICING_AS_OF,
    is_local_model,
    pricing_freshness_note,
)
from cli_modelarium.providers.base import BaseProvider
from cli_modelarium.providers.local_provider import LocalProvider
from cli_modelarium.security import (
    KEY_PATTERNS,
    delete_key,
    delete_local_url,
    is_key_configured,
    load_local_url,
    redact_secrets,
    save_key,
    save_local_url,
)
from cli_modelarium.streaming import (
    DEFAULT_CONCURRENCY,
    StreamState,
    run_streaming_comparison,
)

# Lazy provider import map. Each value is `module_path:ClassName` and is
# resolved via importlib at call time so we don't pay for every SDK import
# on every CLI invocation (matters for fast `--help` and `list-models`).
PROVIDER_REGISTRY: dict[str, str] = {
    "openai": "cli_modelarium.providers.openai_provider:OpenAIProvider",
    "anthropic": "cli_modelarium.providers.anthropic_provider:AnthropicProvider",
    "google": "cli_modelarium.providers.google_provider:GoogleProvider",
    "xai": "cli_modelarium.providers.xai_provider:XAIProvider",
    "deepseek": "cli_modelarium.providers.deepseek_provider:DeepSeekProvider",
    "groq": "cli_modelarium.providers.groq_provider:GroqProvider",
    "openrouter": "cli_modelarium.providers.openrouter_provider:OpenRouterProvider",
    "mistral": "cli_modelarium.providers.mistral_provider:MistralProvider",
    "local": "cli_modelarium.providers.local_provider:LocalProvider",
}

# Exit codes used across the CLI (matches CI/CD conventions).
EXIT_OK = 0
EXIT_ASSERTION_FAILED = 1  # reserved for Phase 9
EXIT_CALL_FAILED = 2

console = Console()


class _DefaultCommandGroup(click.Group):
    """Routes unknown bare arguments to the `compare` subcommand.

    Lets users run `cli-modelarium "prompt" --models X` without typing the
    `compare` verb explicitly.
    """

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            if args and not args[0].startswith("-"):
                args.insert(0, "compare")
                return super().resolve_command(ctx, args)
            raise


@click.group(
    cls=_DefaultCommandGroup,
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version=__version__, prog_name="cli-modelarium")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Cli Modelarium - compare LLM outputs side-by-side from your terminal."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ===== compare =====


@main.command()
@click.argument("prompt", required=True)
@click.option("--models", required=True, help="Comma-separated model IDs or group names.")
@click.option("--temperatures", default="0.0", help="Comma-separated temperatures (default: 0.0).")
@click.option("--system-prompt", help="System prompt applied to every model.")
@click.option(
    "--system-prompts",
    help="Comma-separated system prompts; the comparison fans out across them. Use \\, for a literal comma.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(),
    help="Load a single system prompt from a UTF-8 file (max 1 MB).",
)
@click.option("--judge", help="Score outputs using this model as judge.")
@click.option(
    "--judges",
    help="Comma-separated panel of judges (scores averaged). Use \\, for a literal comma.",
)
@click.option(
    "--judge-criteria",
    help="Comma-separated custom scoring criteria. Use \\, for a literal comma.",
)
@click.option(
    "--judge-template",
    type=click.Path(),
    help="Load a custom judge prompt template from a UTF-8 file (max 1 MB).",
)
@click.option(
    "--include-reasoning", is_flag=True, help="Show each judge's reasoning in the output."
)
@click.option(
    "--no-judge-tos",
    is_flag=True,
    help="Suppress the judge-use ToS reminder (for CI/CD where it's been acknowledged).",
)
@click.option("--check-hallucination", is_flag=True, help="Apply hallucination preset (Phase 10).")
@click.option("--expected-facts", help="Comma-separated reference facts (Phase 10).")
@click.option("--expected-facts-file", type=click.Path(), help="Load expected facts from file (Phase 10).")
@click.option("--output", type=click.Path(), help="Output file path (Phase 7).")
@click.option(
    "--output-format",
    type=click.Choice(["csv", "json", "markdown"], case_sensitive=False),
    help="Output format (Phase 7).",
)
@click.option("--max-cost", type=float, help="Refuse to run if estimated cost exceeds this USD.")
@click.option(
    "--concurrency",
    type=int,
    default=DEFAULT_CONCURRENCY,
    help=f"Max concurrent calls per provider (default: {DEFAULT_CONCURRENCY}).",
)
@click.option("--local-url", help="Override default URL for local model server (Phase 5).")
@click.option("--no-stream", is_flag=True, help="Disable live streaming display.")
def compare(
    prompt: str,
    models: str,
    temperatures: str,
    system_prompt: str | None,
    system_prompts: str | None,
    system_prompt_file: str | None,
    judge: str | None,
    judges: str | None,
    judge_criteria: str | None,
    judge_template: str | None,
    include_reasoning: bool,
    no_judge_tos: bool,
    check_hallucination: bool,
    expected_facts: str | None,
    expected_facts_file: str | None,
    output: str | None,
    output_format: str | None,
    max_cost: float | None,
    concurrency: int,
    local_url: str | None,
    no_stream: bool,
) -> None:
    """Run a side-by-side comparison of LLMs on a single prompt."""
    try:
        model_list = parse_models_arg(models)
        if not model_list:
            raise click.UsageError("--models must include at least one model ID or group.")
        temp_list = _parse_temperatures(temperatures)
        system_prompt_list = _resolve_system_prompts(
            system_prompt=system_prompt,
            system_prompts=system_prompts,
            system_prompt_file=system_prompt_file,
        )
        judge_models = _resolve_judge_models(judge=judge, judges=judges)
        judge_criteria_list, judge_template_text = _resolve_judge_criteria_and_template(
            judge_criteria=judge_criteria,
            judge_template=judge_template,
        )
        # Validate judge models BEFORE the main comparison runs - misconfigured
        # judges should not fail late after burning money on the comparison.
        if judge_models:
            _validate_judge_models(judge_models, local_url=local_url)
    except click.UsageError:
        raise
    except (UnknownModelError, KeyNotConfiguredError, FileNotFoundError, ValueError) as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)

    if judge_models and not no_judge_tos:
        print_tos_disclosure(console)

    def provider_factory(name: str) -> BaseProvider:
        return _get_provider_instance(name, local_url=local_url)

    async def _run_all() -> tuple[list[StreamState], list[JudgeResult] | None]:
        states = await run_streaming_comparison(
            prompt=prompt,
            models=model_list,
            temperatures=temp_list,
            system_prompts=system_prompt_list,
            provider_factory=provider_factory,
            console=console,
            concurrency=concurrency,
            live_display=not no_stream,
        )
        jrs: list[JudgeResult] | None = None
        if judge_models:
            jrs = await run_judging(
                items=[(s, prompt) for s in states],
                judge_models=judge_models,
                criteria=judge_criteria_list,
                provider_factory=provider_factory,
                template=judge_template_text,
                skip_self_eval=True,
                concurrency=concurrency,
            )
        return states, jrs

    try:
        # Single asyncio.run keeps all httpx client cleanup on one event loop,
        # so we don't get "Event loop is closed" warnings on shutdown.
        states, judge_results = asyncio.run(_run_all())
    except KeyNotConfiguredError as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)
    except ModelariumError as e:
        _print_error(redact_secrets(str(e)))
        sys.exit(EXIT_CALL_FAILED)

    _display_results(states, judge_results=judge_results, include_reasoning=include_reasoning)

    if any(s.error for s in states):
        sys.exit(EXIT_CALL_FAILED)
    sys.exit(EXIT_OK)


# ===== batch =====


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--models", required=True, help="Comma-separated model IDs or group names.")
@click.option("--temperatures", default="0.0", help="Comma-separated temperatures (default: 0.0).")
@click.option("--system-prompt", help="System prompt applied to every prompt (per-prompt 'system' field in the input file wins for that prompt).")
@click.option(
    "--system-prompts",
    help="Comma-separated system prompts; the matrix fans out across them. Use \\, for a literal comma.",
)
@click.option(
    "--system-prompt-file",
    type=click.Path(),
    help="Load a single system prompt from a UTF-8 file (max 1 MB).",
)
@click.option("--judge", help="Score outputs using this model as judge.")
@click.option(
    "--judges",
    help="Comma-separated panel of judges (scores averaged). Use \\, for a literal comma.",
)
@click.option(
    "--judge-criteria",
    help="Comma-separated custom scoring criteria. Use \\, for a literal comma.",
)
@click.option(
    "--judge-template",
    type=click.Path(),
    help="Load a custom judge prompt template from a UTF-8 file (max 1 MB).",
)
@click.option(
    "--include-reasoning", is_flag=True, help="Show each judge's reasoning in the output."
)
@click.option(
    "--no-judge-tos",
    is_flag=True,
    help="Suppress the judge-use ToS reminder (for CI/CD where it's been acknowledged).",
)
@click.option("--check-hallucination", is_flag=True, help="Apply the hallucination detection preset (Phase 10).")
@click.option("--expected-facts", help="Comma-separated reference facts (Phase 10).")
@click.option("--expected-facts-file", type=click.Path(), help="Load expected facts from a file (Phase 10).")
@click.option("--output", type=click.Path(), help="Output file path. Format auto-detected from extension; omit to render Markdown on stdout.")
@click.option(
    "--output-format",
    type=click.Choice(["csv", "json", "markdown"], case_sensitive=False),
    help="Override the output format inferred from --output extension.",
)
@click.option("--max-cost", type=float, help="Refuse to run if the estimated cost exceeds this USD.")
@click.option(
    "--concurrency",
    type=int,
    default=DEFAULT_CONCURRENCY,
    help=f"Max concurrent calls per provider (default: {DEFAULT_CONCURRENCY}).",
)
@click.option("--local-url", help="Override the default URL for the local model server.")
@click.option(
    "--min-pass-rate",
    type=float,
    help="Exit 1 if assertion pass rate falls below this threshold (0.0-1.0). "
    "Default behaviour without this flag is strict: ANY assertion failure exits 1.",
)
@click.option(
    "--no-assertions",
    is_flag=True,
    help="Skip assertion checks entirely. Pass/fail counts are zeroed and exit code reflects only call status.",
)
@click.option(
    "--strict-assertions",
    is_flag=True,
    help="Make the default strict behaviour explicit (any assertion failure exits 1). Mutually exclusive with --min-pass-rate.",
)
@click.option("--no-judge", is_flag=True, help="Skip judge scoring (Phase 8).")
@click.option("--force", is_flag=True, help="Overwrite the output file if it exists.")
@click.option("--force-large", is_flag=True, help=f"Bypass safety caps (max 1000 prompts, max 10000 calls).")
def batch(
    file: str,
    models: str,
    temperatures: str,
    system_prompt: str | None,
    system_prompts: str | None,
    system_prompt_file: str | None,
    judge: str | None,
    judges: str | None,
    judge_criteria: str | None,
    judge_template: str | None,
    include_reasoning: bool,
    no_judge_tos: bool,
    check_hallucination: bool,
    expected_facts: str | None,
    expected_facts_file: str | None,
    output: str | None,
    output_format: str | None,
    max_cost: float | None,
    concurrency: int,
    local_url: str | None,
    min_pass_rate: float | None,
    no_assertions: bool,
    strict_assertions: bool,
    no_judge: bool,
    force: bool,
    force_large: bool,
) -> None:
    """Run a multi-prompt batch evaluation from a file.

    The input file is parsed by extension (.txt or .json). Output format is
    inferred from --output's extension (.csv / .json / .md), or pass
    --output-format to override. Omit --output to render Markdown to stdout.

    Per-prompt system prompts: include `"system": "..."` in a JSON prompt
    object to override the command-line system prompt for that one prompt.

    Assertions (JSON input only): include `"assertions": [...]` on a prompt
    object. Exit codes: 0 = all passed, 1 = assertion failure(s) or pass
    rate below --min-pass-rate, 2 = call failure or IO error. Call failures
    win over assertion failures (2 > 1).
    """
    # --strict-assertions and --min-pass-rate are alternatives; combining
    # them is ambiguous, so reject upfront.
    if strict_assertions and min_pass_rate is not None:
        raise click.UsageError(
            "--strict-assertions and --min-pass-rate are mutually exclusive."
        )
    if min_pass_rate is not None and not (0.0 <= min_pass_rate <= 1.0):
        raise click.UsageError(
            f"--min-pass-rate must be between 0.0 and 1.0 (got {min_pass_rate})."
        )

    try:
        prompts = load_batch_file(file)
        if not prompts:
            console.print(
                Panel(
                    f"No prompts to run - the file at {file} parsed as empty.",
                    title="Batch",
                    border_style="yellow",
                )
            )
            return

        model_list = parse_models_arg(models)
        if not model_list:
            raise click.UsageError("--models must include at least one model ID or group.")
        temp_list = _parse_temperatures(temperatures)
        command_sp_list = _resolve_system_prompts(
            system_prompt=system_prompt,
            system_prompts=system_prompts,
            system_prompt_file=system_prompt_file,
        )
        judge_models = _resolve_judge_models(judge=judge, judges=judges)
        judge_criteria_list, judge_template_text = _resolve_judge_criteria_and_template(
            judge_criteria=judge_criteria,
            judge_template=judge_template,
        )
        # Validate judge models BEFORE the batch starts - misconfigured
        # judges should not fail late after burning batch money.
        if judge_models and not no_judge:
            _validate_judge_models(judge_models, local_url=local_url)
    except click.UsageError:
        raise
    except (
        BatchValidationError,
        UnknownModelError,
        KeyNotConfiguredError,
        FileNotFoundError,
        ValueError,
    ) as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)

    # Resolve output target + format.
    try:
        output_path, output_fmt = _resolve_batch_output(
            input_path=file,
            output=output,
            output_format=output_format,
            force=force,
        )
    except OutputFormatError as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)

    # Size limits.
    try:
        total = check_batch_size_limits(
            prompts,
            model_list,
            temp_list,
            command_sp_list,
            force_large=force_large,
        )
    except BatchSizeError as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)

    # Cost ceiling.
    if max_cost is not None:
        est = estimate_batch_cost(prompts, model_list, temp_list, command_sp_list)
        if est > max_cost:
            _print_error(
                f"Estimated cost ${est:.4f} exceeds --max-cost ${max_cost:.4f}.\n"
                f"  Estimate assumes {ESTIMATE_INPUT_TOKENS} input + "
                f"{ESTIMATE_OUTPUT_TOKENS} output tokens per call across "
                f"{total} call{'s' if total != 1 else ''}.\n"
                f"  Reduce dimensions or raise --max-cost to proceed."
            )
            sys.exit(EXIT_CALL_FAILED)

    # Build states + run.
    def provider_factory(name: str) -> BaseProvider:
        return _get_provider_instance(name, local_url=local_url)

    pairs = build_batch_states(prompts, model_list, temp_list, command_sp_list)
    console.print(
        f"[dim]Running batch: {total} call{'s' if total != 1 else ''} "
        f"({len(prompts)} prompt{'s' if len(prompts) != 1 else ''} x "
        f"{len(model_list)} model{'s' if len(model_list) != 1 else ''} x "
        f"{len(temp_list)} temperature{'s' if len(temp_list) != 1 else ''})[/dim]"
    )

    if judge_models and not no_judge and not no_judge_tos:
        print_tos_disclosure(console)

    async def _run_batch_and_judge() -> list[JudgeResult] | None:
        await run_batch(
            pairs=pairs,
            provider_factory=provider_factory,
            console=console,
            concurrency=concurrency,
            show_progress=True,
        )
        if judge_models and not no_judge:
            return await run_judging(
                items=[(s, bp.prompt) for s, bp in pairs],
                judge_models=judge_models,
                criteria=judge_criteria_list,
                provider_factory=provider_factory,
                template=judge_template_text,
                skip_self_eval=True,
                concurrency=concurrency,
            )
        return None

    try:
        # Single asyncio.run keeps all httpx client cleanup on one event loop.
        judge_results = asyncio.run(_run_batch_and_judge())
    except KeyNotConfiguredError as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)
    except ModelariumError as e:
        _print_error(redact_secrets(str(e)))
        sys.exit(EXIT_CALL_FAILED)

    # Run assertions per-state. Failed-call states skip assertion execution
    # (no real output to check). Successful-call states with no configured
    # assertions get an empty list (not None) so they show as "0/0" - which
    # is vacuously fine.
    assertion_results_per_state: list[list[AssertionResult] | None] = []
    for state, bp in pairs:
        if no_assertions or state.error:
            assertion_results_per_state.append(None)
        elif not bp.assertions:
            assertion_results_per_state.append([])
        else:
            assertion_results_per_state.append(
                run_assertions(
                    output=state.text,
                    latency_ms=state.latency_ms,
                    cost_usd=state.cost_usd,
                    assertions=bp.assertions,
                )
            )

    # Convert StreamStates to BatchResults and emit.
    results = []
    for i, (state, bp) in enumerate(pairs):
        jr = judge_results[i] if judge_results is not None else None
        ar = assertion_results_per_state[i]
        results.append(state_to_result(state, bp, judge_result=jr, assertion_results=ar))
    _emit_batch_results(results, output_path=output_path, output_fmt=output_fmt)

    failed = sum(1 for r in results if r.error)
    success = len(results) - failed
    total_cost = sum(r.cost_usd for r in results if r.error is None)

    # Tally assertion outcomes. count_passed excludes `error` rows from
    # both numerator and denominator, so a missing-jsonschema doesn't
    # poison the pass rate or trigger exit 1.
    total_assertion_passed = 0
    total_assertion_definitive = 0
    total_assertion_failed = 0
    for ar in assertion_results_per_state:
        if ar is None:
            continue
        p, d = count_passed(ar)
        total_assertion_passed += p
        total_assertion_definitive += d
        total_assertion_failed += count_failed(ar)

    summary_parts = [
        f"[green]{success} succeeded[/green]",
        f"[red]{failed} failed[/red]",
        f"[dim]total cost ${total_cost:.6f}[/dim]",
    ]
    if judge_results is not None:
        j_cost = total_judge_cost(judge_results)
        j_calls = total_judge_calls(judge_results)
        summary_parts.append(
            f"[dim]judge cost ${j_cost:.6f} ({j_calls} call{'s' if j_calls != 1 else ''})[/dim]"
        )
    if total_assertion_definitive > 0 or total_assertion_failed > 0:
        pass_rate = (
            total_assertion_passed / total_assertion_definitive
            if total_assertion_definitive > 0
            else 1.0
        )
        rate_color = "green" if total_assertion_failed == 0 else "red"
        summary_parts.append(
            f"[{rate_color}]assertions {total_assertion_passed}/{total_assertion_definitive} "
            f"({pass_rate * 100:.0f}%)[/{rate_color}]"
        )
    console.print("  ".join(summary_parts))

    # Exit-code logic. Call failures dominate - usually they mean the user
    # needs to fix credentials/infra before they can even evaluate
    # assertions, so we surface that as 2 rather than the softer 1.
    if failed > 0:
        sys.exit(EXIT_CALL_FAILED)

    if not no_assertions:
        if min_pass_rate is not None:
            # --min-pass-rate threshold mode: tolerate some failures.
            if total_assertion_definitive > 0:
                pass_rate = total_assertion_passed / total_assertion_definitive
                if pass_rate < min_pass_rate:
                    sys.exit(EXIT_ASSERTION_FAILED)
        else:
            # Default / --strict-assertions: ANY failure exits 1.
            if total_assertion_failed > 0:
                sys.exit(EXIT_ASSERTION_FAILED)

    sys.exit(EXIT_OK)


def _resolve_batch_output(
    *,
    input_path: str,
    output: str | None,
    output_format: str | None,
    force: bool,
) -> tuple["Path | None", str]:
    """Decide where to write and which format to use.

    Returns (output_path_or_None, format_name).
        output_path is None when writing to stdout.
        format_name is one of: csv, json, markdown.

    Raises OutputFormatError for unknown extensions when --output-format
    isn't passed, and refuses to overwrite an existing file without --force.
    """
    from pathlib import Path

    if output is None:
        # Stdout default: markdown.
        return None, (output_format.lower() if output_format else "markdown")

    output_path = Path(output).expanduser().resolve()

    if output_overlaps_input(Path(input_path), output_path):
        raise OutputFormatError(
            f"Refusing to write output over the input file ({output_path}).\n"
            f"  Choose a different --output path."
        )

    if output_path.exists() and not force:
        raise OutputFormatError(
            f"Output file already exists: {output_path}\n"
            f"  Use --force to overwrite, or pick a different --output path."
        )

    if output_format:
        fmt = output_format.lower()
    else:
        detected = detect_output_format(output_path)
        if detected is None:
            raise OutputFormatError(
                f"Cannot infer output format from {output_path.suffix!r}.\n"
                f"  Pass --output-format csv|json|markdown explicitly, "
                f"or use a recognized extension (.csv .json .md)."
            )
        fmt = detected

    # Ensure the parent directory exists - users sometimes pass
    # `./results/today.csv` without creating ./results/ first.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path, fmt


def _emit_batch_results(
    results: list, *, output_path: "Path | None", output_fmt: str
) -> None:
    """Dispatch to the right writer/renderer based on resolved format."""
    if output_path is None:
        # Stdout: only markdown is rendered natively; csv/json get printed raw.
        if output_fmt == "markdown":
            render_markdown_to_console(results, console)
        elif output_fmt == "csv":
            from cli_modelarium.output_formatters import _format_csv

            console.print(_format_csv(results), end="")
        elif output_fmt == "json":
            from cli_modelarium.output_formatters import _format_json

            console.print(_format_json(results), end="")
        else:
            _print_error(f"Unsupported output format: {output_fmt!r}")
            sys.exit(EXIT_CALL_FAILED)
        return

    if output_fmt == "csv":
        write_csv(results, output_path)
    elif output_fmt == "json":
        write_json(results, output_path)
    elif output_fmt == "markdown":
        write_markdown(results, output_path)
    else:
        _print_error(f"Unsupported output format: {output_fmt!r}")
        sys.exit(EXIT_CALL_FAILED)
    console.print(f"[dim]Wrote {output_path}[/dim]")


# ===== configure =====


@main.command()
def configure() -> None:
    """Interactively set API keys for each provider."""
    providers = [p for p in all_known_providers() if p != "local"]

    console.print(
        Panel(
            "Configure API keys. Keys are stored in your OS-native keychain.\n"
            "Press Enter to skip any provider.",
            title="cli-modelarium setup",
            border_style="cyan",
        )
    )

    saved = 0
    for provider in providers:
        try:
            key = Prompt.ask(
                f"{provider.capitalize()} API key",
                password=True,
                default="",
                show_default=False,
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Setup cancelled.[/yellow]")
            sys.exit(EXIT_CALL_FAILED)

        if not key.strip():
            console.print(f"  [dim]Skipped {provider}[/dim]")
            continue

        try:
            save_key(provider, key)
        except ValueError as e:
            console.print(f"  [red]Invalid format - {e}[/red]")
            continue
        except Exception as e:
            console.print(f"  [red]Could not save: {redact_secrets(str(e))}[/red]")
            continue

        saved += 1
        console.print(f"  [green]Saved {provider} to keychain[/green]")

    console.print()
    console.print(
        Panel(
            f"{saved} of {len(providers)} providers configured.\n"
            f"Run: cli-modelarium list-models",
            title="Configuration complete",
            border_style="green",
        )
    )


# ===== keys =====


@main.group()
def keys() -> None:
    """Manage API keys (stored in OS-native keychain)."""


@keys.command("list")
def keys_list() -> None:
    """Show which providers have keys configured."""
    providers = [p for p in all_known_providers() if p != "local"]

    table = Table(title="API key status", border_style="dim")
    table.add_column("Provider", style="bold")
    table.add_column("Status")

    for provider in providers:
        if is_key_configured(provider):
            table.add_row(provider, "[green]configured[/green]")
        else:
            table.add_row(provider, "[dim]not configured[/dim]")

    saved_local = load_local_url()
    if saved_local:
        table.add_row("local", f"[green]{saved_local}[/green]")
    else:
        table.add_row("local", f"[dim]default ({LocalProvider.DEFAULT_URL})[/dim]")

    console.print(table)


@keys.command("set")
@click.argument("provider")
@click.option("--base-url", help="(local provider only) Override default base URL.")
def keys_set(provider: str, base_url: str | None) -> None:
    """Set or update the API key for a provider (prompts securely).

    For the local provider, pass --base-url to persist a default URL
    instead of prompting for an API key.
    """
    if provider == "local":
        if not base_url:
            _print_error(
                "Local provider takes --base-url, not an API key.\n"
                "  Example: cli-modelarium keys set local --base-url http://localhost:1234/v1"
            )
            sys.exit(EXIT_CALL_FAILED)
        try:
            LocalProvider._validate_local_url(base_url)
        except ModelariumError as e:
            _print_error(str(e))
            sys.exit(EXIT_CALL_FAILED)
        save_local_url(base_url)
        console.print(f"[green]Saved local provider URL: {base_url}[/green]")
        return

    if provider not in KEY_PATTERNS:
        _print_error(
            f"Unknown provider: {provider}.\n"
            f"Supported providers: {', '.join(sorted(KEY_PATTERNS))}, local"
        )
        sys.exit(EXIT_CALL_FAILED)

    try:
        key = Prompt.ask(f"{provider.capitalize()} API key", password=True)
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Cancelled.[/yellow]")
        sys.exit(EXIT_CALL_FAILED)

    try:
        save_key(provider, key)
    except ValueError as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)
    except Exception as e:
        _print_error(redact_secrets(str(e)))
        sys.exit(EXIT_CALL_FAILED)

    console.print(f"[green]Saved {provider} key to keychain.[/green]")


@keys.command("delete")
@click.argument("provider")
def keys_delete(provider: str) -> None:
    """Remove the API key for a provider from the keychain."""
    if provider == "local":
        delete_local_url()
        console.print("[green]Removed saved local provider URL.[/green]")
        return
    delete_key(provider)
    console.print(f"[green]Removed {provider} key from keychain.[/green]")


# ===== list-models =====


@main.command("list-models")
@click.option("--local", "local_only", is_flag=True, help="Show only local models (queries the local server).")
@click.option("--local-url", help="Override default URL for local-model discovery.")
def list_models(local_only: bool, local_url: str | None) -> None:
    """List supported models, grouped by provider."""
    if local_only:
        _list_local_models(local_url)
        return

    providers = all_known_providers()

    any_shown = False
    for provider in providers:
        models = list_models_for_provider(provider)
        if provider == "local":
            # Local models are dynamic - skip the static section; we show
            # discovered models when --local is passed.
            continue
        if not models:
            continue

        any_shown = True
        configured = "configured" if is_key_configured(provider) else "not configured"
        title = f"{provider} [dim]({configured})[/dim]"

        table = Table(title=title, border_style="dim", title_justify="left")
        table.add_column("Model", style="bold")
        table.add_column("Input $/MTok", justify="right")
        table.add_column("Output $/MTok", justify="right")
        table.add_column("Cached $/MTok", justify="right", style="dim")

        for model in models:
            entry = PRICING[model]
            cached = entry.get("cached_input")
            cached_text = f"${float(cached):.4f}" if cached is not None else "-"
            table.add_row(
                model,
                f"${float(entry['input']):.4f}",
                f"${float(entry['output']):.4f}",
                cached_text,
            )

        console.print(table)
        console.print()

    if not any_shown:
        _print_error("No models registered.")
        sys.exit(EXIT_CALL_FAILED)

    console.print(
        "[dim]Local models are routed by the `local/` prefix.[/dim] "
        "[dim]Run `cli-modelarium list-models --local` to discover what's running locally.[/dim]"
    )
    console.print(f"[dim]{pricing_freshness_note()}[/dim]")


def _list_local_models(local_url: str | None) -> None:
    """Query the local server's /models endpoint and render the result."""
    url = local_url or load_local_url() or LocalProvider.DEFAULT_URL

    try:
        # Validate the URL up front - we want LocalURLError BEFORE any I/O.
        LocalProvider._validate_local_url(url)
    except ModelariumError as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)

    try:
        models = asyncio.run(LocalProvider.discover_models(url))
    except httpx.ConnectError:
        console.print(
            Panel(
                f"Could not reach local server at {url}.\n\n"
                f"Possible causes:\n"
                f"  - Server not running (try: ollama serve)\n"
                f"  - Wrong URL (use --local-url to override)\n"
                f"  - Firewall blocking the connection",
                title="Local models",
                border_style="yellow",
            )
        )
        return
    except httpx.TimeoutException:
        console.print(
            Panel(
                f"Timed out connecting to {url} "
                f"(waited {LocalProvider.DISCOVERY_TIMEOUT_SECONDS:.0f}s).\n"
                f"The server may be starting up or under heavy load.",
                title="Local models",
                border_style="yellow",
            )
        )
        return
    except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
        console.print(
            Panel(
                f"Local server at {url} returned an unexpected response:\n"
                f"  {redact_secrets(str(e))}",
                title="Local models",
                border_style="yellow",
            )
        )
        return

    if not models:
        console.print(
            Panel(
                f"Local server at {url} responded but has no models installed.\n"
                f"For Ollama: ollama pull llama3.3",
                title="Local models",
                border_style="cyan",
            )
        )
        return

    table = Table(
        title=f"local [dim]({url})[/dim]",
        border_style="dim",
        title_justify="left",
    )
    table.add_column("Model ID for cli-modelarium", style="bold")
    table.add_column("Created", style="dim")
    table.add_column("Owned by", style="dim")
    table.add_column("Cost", justify="right")

    for entry in models:
        model_id = entry.get("id", "(unnamed)")
        created = entry.get("created")
        created_text = _format_unix_timestamp(created) if created else "-"
        owned_by = str(entry.get("owned_by", "-"))
        table.add_row(f"local/{model_id}", created_text, owned_by, "[dim]Free[/dim]")

    console.print(table)
    console.print(
        f"\n[dim]Use these via: cli-modelarium 'prompt' --models local/{models[0].get('id', '<name>')}[/dim]"
    )


def _format_unix_timestamp(ts: object) -> str:
    """Format a unix timestamp (int or float) as YYYY-MM-DD, or '-' if unparseable."""
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "-"


# ===== pricing =====


@main.command("pricing")
@click.argument("model", required=False)
@click.option("--all", "show_all", is_flag=True, help="Show pricing for every model.")
def pricing_cmd(model: str | None, show_all: bool) -> None:
    """Show pricing for a model or all models."""
    if show_all or model is None:
        table = Table(title="Pricing (per 1M tokens, USD)", border_style="dim")
        table.add_column("Model", style="bold")
        table.add_column("Provider")
        table.add_column("Input", justify="right")
        table.add_column("Output", justify="right")
        table.add_column("Cached", justify="right", style="dim")

        for name in sorted(PRICING):
            if name.endswith("/*"):
                continue
            entry = PRICING[name]
            if entry.get("is_local"):
                table.add_row(
                    name, str(entry["provider"]), "[dim]Free[/dim]", "[dim]Free[/dim]", "[dim]-[/dim]"
                )
                continue
            cached = entry.get("cached_input")
            cached_text = f"${float(cached):.4f}" if cached is not None else "-"
            table.add_row(
                name,
                str(entry["provider"]),
                f"${float(entry['input']):.4f}",
                f"${float(entry['output']):.4f}",
                cached_text,
            )

        console.print(table)
        console.print(f"[dim]{pricing_freshness_note()}[/dim]")
        return

    if is_local_model(model):
        console.print(f"[bold]{model}[/bold]: [dim]Free (local model)[/dim]")
        return

    entry = PRICING.get(model)
    if entry is None:
        _print_error(f"Unknown model: {model}. Run `cli-modelarium list-models` to see options.")
        sys.exit(EXIT_CALL_FAILED)

    cached = entry.get("cached_input")
    console.print(f"[bold]{model}[/bold] ([dim]{entry['provider']}[/dim])")
    console.print(f"  Input:   ${float(entry['input']):.4f} / 1M tokens")
    console.print(f"  Output:  ${float(entry['output']):.4f} / 1M tokens")
    if cached is not None:
        console.print(f"  Cached:  ${float(cached):.4f} / 1M tokens")
    console.print(f"\n[dim]{pricing_freshness_note()}[/dim]")


# ===== helpers =====


def _parse_temperatures(raw: str) -> list[float]:
    """Parse a comma-separated temperatures string into floats. Defaults to [0.0]."""
    if not raw.strip():
        return [0.0]
    out: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except ValueError:
            raise ValueError(f"Invalid temperature value: {token!r}") from None
    return out or [0.0]


def _resolve_system_prompts(
    *,
    system_prompt: str | None,
    system_prompts: str | None,
    system_prompt_file: str | None,
) -> list[str | None]:
    """Resolve the three mutually-exclusive system-prompt flags into a list.

    Returns `[None]` when no system prompt is configured (the orchestrator
    treats this as "one task with no system prompt"). Returns a list of
    strings otherwise - never an empty list.

    Raises:
        click.UsageError: if more than one of the three flags is set.
        FileNotFoundError / ValueError: from `load_system_prompt`.
    """
    used = [
        name
        for name, val in (
            ("--system-prompt", system_prompt),
            ("--system-prompts", system_prompts),
            ("--system-prompt-file", system_prompt_file),
        )
        if val
    ]
    if len(used) > 1:
        raise click.UsageError(
            f"{', '.join(used)} are mutually exclusive - pick one."
        )

    if system_prompt_file:
        return [load_system_prompt(system_prompt_file)]
    if system_prompts:
        parsed = _split_system_prompts(system_prompts)
        return parsed or [None]
    if system_prompt:
        # Empty-string case is also caught here; we'd have failed the `if`
        # above. But guard anyway: a stripped-empty value means no prompt.
        stripped = system_prompt.strip()
        return [stripped] if stripped else [None]
    return [None]


def _split_escaped_csv(value: str) -> list[str]:
    """Split a comma-separated string with `\\,` as a literal-comma escape.

    Whitespace around each piece is stripped; empty pieces (e.g. trailing
    comma) are dropped. Any other backslash is kept verbatim.

    Example: `_split_escaped_csv(r"a,b,c\\,d")` -> `["a", "b", "c,d"]`
    """
    out: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(value):
        c = value[i]
        if c == "\\" and i + 1 < len(value) and value[i + 1] == ",":
            buf.append(",")
            i += 2
            continue
        if c == ",":
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    last = "".join(buf).strip()
    if last:
        out.append(last)
    return out


# Backwards-compat alias - Phase 6 tests import this name.
_split_system_prompts = _split_escaped_csv


def _resolve_judge_models(
    *, judge: str | None, judges: str | None
) -> list[str]:
    """Resolve --judge / --judges into a list of judge model IDs.

    Returns [] when neither flag is set. Raises click.UsageError if both
    flags are set simultaneously (they're mutually exclusive).
    """
    if judge and judges:
        raise click.UsageError(
            "--judge and --judges are mutually exclusive - pick one."
        )
    if judges:
        return _split_escaped_csv(judges)
    if judge and judge.strip():
        return [judge.strip()]
    return []


def _resolve_judge_criteria_and_template(
    *,
    judge_criteria: str | None,
    judge_template: str | None,
) -> tuple[list[str], str]:
    """Resolve --judge-criteria and --judge-template into (criteria, template).

    Returns (DEFAULT_CRITERIA, JUDGE_PROMPT_TEMPLATE) when neither is set.
    Raises click.UsageError if both are set simultaneously.

    --judge-template loads a custom prompt template from disk (UTF-8, max 1 MB).
    --judge-criteria splits on commas with the same `\\,` escape as system prompts.
    """
    if judge_criteria and judge_template:
        raise click.UsageError(
            "--judge-criteria and --judge-template are mutually exclusive - pick one."
        )
    criteria = list(DEFAULT_CRITERIA)
    template = JUDGE_PROMPT_TEMPLATE
    if judge_criteria:
        criteria = _split_escaped_csv(judge_criteria) or list(DEFAULT_CRITERIA)
    if judge_template:
        # Reuse the system-prompt-file loader: same size + encoding contract.
        template = load_system_prompt(judge_template)
    return criteria, template


def _validate_judge_models(
    judge_models: list[str], *, local_url: str | None
) -> None:
    """Ensure every judge model is in the registry AND has a configured key.

    This runs BEFORE any main API calls - the build prompt's contract is
    that a misconfigured judge fails the run immediately, not after burning
    money on the comparison.
    """
    from cli_modelarium.models_registry import get_provider_for_model
    from cli_modelarium.security import is_key_configured

    seen_providers: set[str] = set()
    for model in judge_models:
        provider_name = get_provider_for_model(model)  # raises UnknownModelError
        if provider_name in seen_providers:
            continue
        seen_providers.add(provider_name)
        if provider_name == "local":
            # Local needs no key; the URL check happens at construction time.
            continue
        if not is_key_configured(provider_name):
            raise KeyNotConfiguredError(provider_name)


def _get_provider_instance(
    provider_name: str, *, local_url: str | None = None
) -> BaseProvider:
    """Instantiate the provider for `provider_name`.

    For cloud providers: loads the API key from env var / keychain and
    raises `KeyNotConfiguredError` if missing.

    For the local provider: skips the API-key path entirely. The URL is taken
    from `local_url` if provided, else from the keychain/env var, else
    `LocalProvider.DEFAULT_URL`.
    """
    if provider_name not in PROVIDER_REGISTRY:
        raise UnknownProviderError(
            f"Provider '{provider_name}' is not yet wired up. "
            f"Currently supported: {', '.join(sorted(PROVIDER_REGISTRY))}."
        )

    if provider_name == "local":
        url = local_url or load_local_url()
        return LocalProvider(base_url=url)

    from cli_modelarium.security import load_key

    api_key = load_key(provider_name)
    if not api_key:
        raise KeyNotConfiguredError(provider_name)

    module_path, _, class_name = PROVIDER_REGISTRY[provider_name].partition(":")
    import importlib

    module = importlib.import_module(module_path)
    provider_cls = getattr(module, class_name)
    return provider_cls(api_key=api_key)


def _display_results(
    states: list[StreamState],
    judge_results: list[JudgeResult] | None = None,
    include_reasoning: bool = False,
) -> None:
    """Render the comparison results as a Rich table plus per-model output blocks.

    `judge_results` is parallel to `states` when provided; it adds a Score
    column to the table and a Reasoning line under each output block when
    `include_reasoning=True`.
    """
    # Only surface the SP column when there are 2+ distinct non-empty
    # system prompts. The streaming legend has already printed the full
    # mapping; we just need the index here.
    from cli_modelarium.streaming import prompt_index_map

    prompt_indices = prompt_index_map(states)
    show_sp_column = bool(prompt_indices)
    show_score_column = judge_results is not None

    table = Table(
        title=f"Comparing {len(states)} completion{'s' if len(states) != 1 else ''}",
        border_style="dim",
        title_justify="left",
    )
    table.add_column("Model", style="bold")
    if show_sp_column:
        table.add_column("SP", style="magenta", justify="right")
    table.add_column("Temp", justify="right")
    table.add_column("TTFT", justify="right", style="dim")
    table.add_column("Latency", justify="right", style="dim")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Cost", justify="right")
    if show_score_column:
        table.add_column("Score", style="magenta", justify="right")
    table.add_column("Status")

    total_cost = 0.0
    for i, s in enumerate(states):
        if s.error:
            status = "[red]error[/red]"
            cost_text = "[dim]-[/dim]"
            ttft_text = "[dim]-[/dim]"
            latency_text = "[dim]-[/dim]"
            in_text = "[dim]-[/dim]"
            out_text = "[dim]-[/dim]"
        else:
            status = "[green]ok[/green]"
            total_cost += s.cost_usd
            cost_text = "[dim]Free[/dim]" if is_local_model(s.model) else f"${s.cost_usd:.6f}"
            ttft_text = f"{s.ttft_ms / 1000:.2f}s" if s.ttft_ms is not None else "[dim]-[/dim]"
            latency_text = f"{s.latency_ms / 1000:.2f}s" if s.latency_ms is not None else "[dim]-[/dim]"
            in_text = str(s.input_tokens)
            out_text = str(s.output_tokens)

        row: list[str] = [s.model]
        if show_sp_column:
            if s.system_prompt and s.system_prompt in prompt_indices:
                row.append(f"SP {prompt_indices[s.system_prompt]}")
            else:
                row.append("[dim]-[/dim]")
        row.extend([
            f"{s.temperature:.1f}",
            ttft_text,
            latency_text,
            in_text,
            out_text,
            cost_text,
        ])
        if show_score_column:
            assert judge_results is not None
            row.append(_score_cell_for_compare(judge_results[i]))
        row.append(status)
        table.add_row(*row)

    console.print(table)
    console.print()

    # Per-model output blocks. When multiple SPs are in play, identify which
    # one produced each block so the reader can cross-reference the legend.
    for i, s in enumerate(states):
        header = f"[bold cyan]>[/bold cyan] [bold]{s.model}[/bold] @ {s.temperature:.1f}"
        if show_sp_column and s.system_prompt and s.system_prompt in prompt_indices:
            header += f"  [magenta]SP {prompt_indices[s.system_prompt]}[/magenta]"
        console.print(header)
        if s.error:
            console.print(f"  [red]{s.error}[/red]")
        else:
            for line in s.text.splitlines() or [""]:
                console.print(f"  {line}")
        # Optional judge reasoning lines.
        if include_reasoning and judge_results is not None:
            for j in judge_results[i].judges:
                score_str = j.score if j.score is not None else "?"
                if j.parse_error:
                    console.print(
                        f"  [magenta dim]judge {j.model}: parse error - "
                        f"{j.parse_error}[/magenta dim]"
                    )
                else:
                    console.print(
                        f"  [magenta dim]judge {j.model} ({score_str}/10): "
                        f"{j.reasoning}[/magenta dim]"
                    )
        console.print()

    console.print(f"[dim]Total cost: ${total_cost:.6f}[/dim]")
    if judge_results is not None:
        j_cost = total_judge_cost(judge_results)
        j_calls = total_judge_calls(judge_results)
        console.print(
            f"[dim]Judge cost: ${j_cost:.6f} "
            f"({j_calls} judge call{'s' if j_calls != 1 else ''})[/dim]"
        )
    console.print(f"[dim]{pricing_freshness_note()}[/dim]")


def _score_cell_for_compare(jr: JudgeResult) -> str:
    """Render the Score column cell for the compare command's results table."""
    if not jr.judges:
        if jr.skipped_models:
            return f"[dim]-[/dim]"
        return "[dim]-[/dim]"
    successful = [j for j in jr.judges if j.score is not None]
    if not successful:
        return "[red]N/A[/red]"
    if len(jr.judges) == 1 and successful:
        # Single judge: just the score.
        return str(successful[0].score)
    # Panel: average + count.
    if jr.average_score is not None:
        return f"{jr.average_score:.1f} ({len(successful)})"
    return "[red]N/A[/red]"


def _print_error(message: str) -> None:
    """Print an error inside a red-bordered panel."""
    console.print(Panel(redact_secrets(message), title="Error", border_style="red"))


if __name__ == "__main__":
    main()
