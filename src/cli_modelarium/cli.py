"""Click CLI entry point for Cli Modelarium.

Phase 1 skeleton: command structure is in place but business logic is stubbed
out and will be wired up in subsequent phases.

The `_DefaultCommandGroup` class routes invocations like
`cli-modelarium "prompt" --models X` to the `compare` subcommand, so users
do not have to type the verb explicitly.
"""
from __future__ import annotations

import click

from cli_modelarium import __version__


class _DefaultCommandGroup(click.Group):
    """A Click group that routes unknown bare arguments to the `compare` subcommand.

    Lets users run `cli-modelarium "prompt" --models X` without typing the
    `compare` verb, while preserving `cli-modelarium batch ...`,
    `cli-modelarium configure`, etc.
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


@main.command()
@click.argument("prompt", required=True)
@click.option("--models", required=True, help="Comma-separated model IDs or group names.")
@click.option("--temperatures", default="0.0", help="Comma-separated temperatures (default: 0.0).")
@click.option("--system-prompt", help="System prompt applied to every model.")
@click.option("--system-prompts", help="Comma-separated system prompts to compare against the user prompt.")
@click.option("--system-prompt-file", type=click.Path(), help="Load system prompt from a file.")
@click.option("--judge", help="Score outputs using this model as judge.")
@click.option("--judges", help="Comma-separated panel of judges (scores averaged).")
@click.option("--judge-criteria", help="Comma-separated custom scoring criteria.")
@click.option("--judge-template", type=click.Path(), help="Custom judge prompt template file.")
@click.option("--include-reasoning", is_flag=True, help="Include each judge's reasoning in output.")
@click.option("--check-hallucination", is_flag=True, help="Apply the hallucination detection preset.")
@click.option("--expected-facts", help="Comma-separated reference facts for hallucination check.")
@click.option("--expected-facts-file", type=click.Path(), help="Load expected facts from a file.")
@click.option("--output", type=click.Path(), help="Output file path.")
@click.option(
    "--output-format",
    type=click.Choice(["csv", "json", "markdown"], case_sensitive=False),
    help="Output format (csv, json, markdown).",
)
@click.option("--max-cost", type=float, help="Refuse to run if estimated cost exceeds this (USD).")
@click.option("--concurrency", type=int, default=5, help="Max concurrent calls per provider (default: 5).")
@click.option("--local-url", help="Override the default URL for the local model server.")
@click.option("--no-stream", is_flag=True, help="Disable streaming display (collect then print).")
def compare(**kwargs: object) -> None:
    """Run a side-by-side comparison of LLMs on a single prompt."""
    click.echo("compare: not yet implemented (Phase 2 wires this up)")


@main.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
@click.option("--models", required=True, help="Comma-separated model IDs or group names.")
@click.option("--temperatures", default="0.0", help="Comma-separated temperatures (default: 0.0).")
@click.option("--system-prompt", help="System prompt applied to every model.")
@click.option("--system-prompt-file", type=click.Path(), help="Load system prompt from a file.")
@click.option("--judge", help="Score outputs using this model as judge.")
@click.option("--judges", help="Comma-separated panel of judges (scores averaged).")
@click.option("--judge-criteria", help="Comma-separated custom scoring criteria.")
@click.option("--judge-template", type=click.Path(), help="Custom judge prompt template file.")
@click.option("--include-reasoning", is_flag=True, help="Include each judge's reasoning in output.")
@click.option("--check-hallucination", is_flag=True, help="Apply the hallucination detection preset.")
@click.option("--expected-facts", help="Comma-separated reference facts for hallucination check.")
@click.option("--expected-facts-file", type=click.Path(), help="Load expected facts from a file.")
@click.option("--output", type=click.Path(), help="Output file path (CSV/JSON/Markdown).")
@click.option(
    "--output-format",
    type=click.Choice(["csv", "json", "markdown"], case_sensitive=False),
    help="Output format (csv, json, markdown).",
)
@click.option("--max-cost", type=float, help="Refuse to run if estimated cost exceeds this (USD).")
@click.option("--concurrency", type=int, default=5, help="Max concurrent calls per provider (default: 5).")
@click.option("--local-url", help="Override the default URL for the local model server.")
@click.option("--no-stream", is_flag=True, help="Disable streaming display (collect then print).")
@click.option("--min-pass-rate", type=float, help="Exit non-zero if assertion pass rate falls below this.")
@click.option("--no-assertions", is_flag=True, help="Skip assertion checks even if defined.")
@click.option("--no-judge", is_flag=True, help="Skip judge scoring even if --judge is set.")
@click.option("--force", is_flag=True, help="Overwrite output file if it exists.")
@click.option("--force-large", is_flag=True, help="Allow batches larger than 1000 prompts.")
def batch(**kwargs: object) -> None:
    """Run a multi-prompt batch evaluation from a file."""
    click.echo("batch: not yet implemented (Phase 7 wires this up)")


@main.command()
def configure() -> None:
    """Interactively set API keys for each provider."""
    click.echo("configure: not yet implemented (Phase 2 wires this up)")


@main.group()
def keys() -> None:
    """Manage API keys (stored in OS-native keychain)."""


@keys.command("list")
def keys_list() -> None:
    """Show which providers have keys configured."""
    click.echo("keys list: not yet implemented (Phase 2 wires this up)")


@keys.command("set")
@click.argument("provider")
@click.option("--base-url", help="(local provider only) Override default base URL.")
def keys_set(provider: str, base_url: str | None) -> None:
    """Set or update the API key for a provider (prompts securely)."""
    click.echo(f"keys set {provider}: not yet implemented (Phase 2 wires this up)")


@keys.command("delete")
@click.argument("provider")
def keys_delete(provider: str) -> None:
    """Remove the API key for a provider from the keychain."""
    click.echo(f"keys delete {provider}: not yet implemented (Phase 2 wires this up)")


@main.command("list-models")
@click.option("--local", "local_only", is_flag=True, help="Show only local models.")
def list_models(local_only: bool) -> None:
    """List supported models, grouped by provider."""
    click.echo("list-models: not yet implemented (Phase 2 wires this up)")


@main.command()
@click.argument("model", required=False)
@click.option("--all", "show_all", is_flag=True, help="Show pricing for every model.")
def pricing(model: str | None, show_all: bool) -> None:
    """Show pricing for a model or all models."""
    click.echo("pricing: not yet implemented (Phase 2 wires this up)")


if __name__ == "__main__":
    main()
