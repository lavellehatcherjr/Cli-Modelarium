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
from cli_modelarium.exceptions import (
    KeyNotConfiguredError,
    ModelariumError,
    ProviderError,
    UnknownModelError,
    UnknownProviderError,
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
@click.option("--system-prompts", help="Comma-separated system prompts to compare (Phase 6).")
@click.option("--system-prompt-file", type=click.Path(), help="Load system prompt from file (Phase 6).")
@click.option("--judge", help="Score outputs using this model as judge (Phase 8).")
@click.option("--judges", help="Comma-separated panel of judges (Phase 8).")
@click.option("--judge-criteria", help="Comma-separated custom scoring criteria (Phase 8).")
@click.option("--judge-template", type=click.Path(), help="Custom judge prompt template (Phase 8).")
@click.option("--include-reasoning", is_flag=True, help="Include judge reasoning in output (Phase 8).")
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
    except (UnknownModelError, ValueError) as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)

    def provider_factory(name: str) -> BaseProvider:
        return _get_provider_instance(name, local_url=local_url)

    try:
        states = asyncio.run(
            run_streaming_comparison(
                prompt=prompt,
                models=model_list,
                temperatures=temp_list,
                system_prompt=system_prompt,
                provider_factory=provider_factory,
                console=console,
                concurrency=concurrency,
                live_display=not no_stream,
            )
        )
    except KeyNotConfiguredError as e:
        _print_error(str(e))
        sys.exit(EXIT_CALL_FAILED)
    except ModelariumError as e:
        _print_error(redact_secrets(str(e)))
        sys.exit(EXIT_CALL_FAILED)

    _display_results(states)

    if any(s.error for s in states):
        sys.exit(EXIT_CALL_FAILED)
    sys.exit(EXIT_OK)


# ===== batch =====


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--models", required=True, help="Comma-separated model IDs or group names.")
@click.option("--temperatures", default="0.0", help="Comma-separated temperatures (default: 0.0).")
@click.option("--system-prompt", help="System prompt applied to every model.")
@click.option("--system-prompt-file", type=click.Path(), help="Load system prompt from a file.")
@click.option("--judge", help="Score outputs using this model as judge.")
@click.option("--judges", help="Comma-separated panel of judges.")
@click.option("--judge-criteria", help="Comma-separated custom scoring criteria.")
@click.option("--judge-template", type=click.Path(), help="Custom judge prompt template.")
@click.option("--include-reasoning", is_flag=True, help="Include judge reasoning in output.")
@click.option("--check-hallucination", is_flag=True, help="Apply the hallucination detection preset.")
@click.option("--expected-facts", help="Comma-separated reference facts.")
@click.option("--expected-facts-file", type=click.Path(), help="Load expected facts from a file.")
@click.option("--output", type=click.Path(), help="Output file path (CSV/JSON/Markdown).")
@click.option(
    "--output-format",
    type=click.Choice(["csv", "json", "markdown"], case_sensitive=False),
    help="Output format.",
)
@click.option("--max-cost", type=float, help="Refuse to run if estimated cost exceeds this USD.")
@click.option("--concurrency", type=int, default=5, help="Max concurrent calls per provider.")
@click.option("--local-url", help="Override the default URL for the local model server.")
@click.option("--no-stream", is_flag=True, help="Disable streaming display.")
@click.option("--min-pass-rate", type=float, help="Exit non-zero if assertion pass rate is below this.")
@click.option("--no-assertions", is_flag=True, help="Skip assertion checks even if defined.")
@click.option("--no-judge", is_flag=True, help="Skip judge scoring even if --judge is set.")
@click.option("--force", is_flag=True, help="Overwrite output file if it exists.")
@click.option("--force-large", is_flag=True, help="Allow batches larger than 1000 prompts.")
def batch(**_kwargs: Any) -> None:
    """Run a multi-prompt batch evaluation from a file."""
    _print_error("batch: not yet implemented (Phase 7 wires this up)")
    sys.exit(EXIT_CALL_FAILED)


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


def _display_results(states: list[StreamState]) -> None:
    """Render the comparison results as a Rich table plus per-model output blocks."""
    table = Table(
        title=f"Comparing {len(states)} completion{'s' if len(states) != 1 else ''}",
        border_style="dim",
        title_justify="left",
    )
    table.add_column("Model", style="bold")
    table.add_column("Temp", justify="right")
    table.add_column("TTFT", justify="right", style="dim")
    table.add_column("Latency", justify="right", style="dim")
    table.add_column("In", justify="right")
    table.add_column("Out", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Status")

    total_cost = 0.0
    for s in states:
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

        table.add_row(
            s.model,
            f"{s.temperature:.1f}",
            ttft_text,
            latency_text,
            in_text,
            out_text,
            cost_text,
            status,
        )

    console.print(table)
    console.print()

    # Per-model output blocks
    for s in states:
        header = f"[bold cyan]>[/bold cyan] [bold]{s.model}[/bold] @ {s.temperature:.1f}"
        console.print(header)
        if s.error:
            console.print(f"  [red]{s.error}[/red]")
        else:
            for line in s.text.splitlines() or [""]:
                console.print(f"  {line}")
        console.print()

    console.print(f"[dim]Total cost: ${total_cost:.6f}[/dim]")
    console.print(f"[dim]{pricing_freshness_note()}[/dim]")


def _print_error(message: str) -> None:
    """Print an error inside a red-bordered panel."""
    console.print(Panel(redact_secrets(message), title="Error", border_style="red"))


if __name__ == "__main__":
    main()
