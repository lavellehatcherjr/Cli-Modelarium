"""Model registry: maps model IDs to providers and defines group shortcuts."""

from __future__ import annotations

from cli_modelarium.exceptions import UnknownModelError
from cli_modelarium.pricing import PRICING, is_local_model

# Group shortcuts let users write `--models all-premium` instead of listing
# six model IDs. Groups are filtered at resolution time against the user's
# configured providers, so requesting `all-premium` without an Anthropic key
# just yields the providers the user actually has.
MODEL_GROUPS: dict[str, list[str]] = {
    "all-premium": [
        "gpt-5.5",
        "claude-opus-4-8",
        "gemini-3.1-pro-preview",
        "grok-4.3",
        "deepseek-v4-pro",
        "mistral-large-latest",
        "qwen3.7-max",
        "glm-5.2",
    ],
    "all-flagship": [
        "gpt-5.5",
        "claude-opus-4-8",
        "gemini-3.1-pro-preview",
        "grok-4.3",
        "deepseek-v4-pro",
        "mistral-large-latest",
        "qwen3.7-max",
        "glm-5.2",
    ],
    "all-budget": [
        "gpt-5.4-nano",
        "claude-haiku-4-5",
        "gemini-3.1-flash-lite",
        "grok-4.20-0309-non-reasoning",
        "deepseek-v4-flash",
        "mistral-small-latest",
        "qwen3.7-plus",
        "glm-4.5-air",
    ],
    "all-reasoning": [
        "o3",
        "o4-mini",
        "deepseek-reasoner",
        "magistral-medium-latest",
        "magistral-small-latest",
        "glm-5.2",
    ],
    "all-fast": [
        "claude-haiku-4-5",
        "gemini-3.5-flash",
        "grok-4.20-0309-non-reasoning",
        "deepseek-v4-flash",
        "llama-3.3-70b-versatile",
        "qwen3.6-flash",
        "glm-5-turbo",
    ],
    "all-cheap": [
        "gpt-4o-mini",
        "claude-haiku-4-5",
        "gemini-2.5-flash-lite",
        "deepseek-v4-flash",
        "mistral-small-latest",
        "qwen-flash",
        "glm-4.7-flashx",
    ],
    "all-open-weight": [
        "gpt-oss-120b",
        "gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ],
    # Resolved dynamically at call time against the user's configured providers
    # and local models. Listed here so `is_group_name()` recognizes them.
    "all-local": [],
    "all": [],
}

# Group names that need dynamic resolution by the caller (not just lookup).
DYNAMIC_GROUPS = frozenset({"all-local", "all"})


def get_provider_for_model(model: str) -> str:
    """Return the provider name for a model ID.

    Raises UnknownModelError if the model is not in the registry.
    """
    if is_local_model(model):
        return "local"
    pricing = PRICING.get(model)
    if pricing is None:
        raise UnknownModelError(
            f"Unknown model: {model}. Run `cli-modelarium list-models` to see supported models."
        )
    return str(pricing["provider"])


def list_models_for_provider(provider: str) -> list[str]:
    """Return all concrete model IDs registered for a provider, sorted alphabetically."""
    return sorted(
        model
        for model, p in PRICING.items()
        if p.get("provider") == provider and not model.endswith("/*")
    )


def all_known_providers() -> list[str]:
    """Return all unique provider names present in the pricing registry, sorted."""
    providers = {str(p["provider"]) for p in PRICING.values()}
    return sorted(providers)


def is_group_name(name: str) -> bool:
    """Return True if `name` is a registered model group shortcut."""
    return name in MODEL_GROUPS


def expand_group(group: str) -> list[str]:
    """Expand a group name to its constituent model IDs.

    Dynamic groups (`all-local`, `all`) return an empty list - the caller is
    expected to populate them with configured models.
    """
    return list(MODEL_GROUPS.get(group, []))


def parse_models_arg(models_arg: str) -> list[str]:
    """Parse a `--models` CLI value into a flat list of model IDs.

    Accepts a comma-separated string mixing concrete model IDs and group names.
    Groups are expanded in place. Dynamic groups are returned as-is for the
    caller to resolve against runtime context.
    """
    result: list[str] = []
    for token in (t.strip() for t in models_arg.split(",")):
        if not token:
            continue
        if is_group_name(token):
            expanded = expand_group(token)
            if not expanded and token in DYNAMIC_GROUPS:
                result.append(token)
            else:
                result.extend(expanded)
        else:
            result.append(token)
    return result
