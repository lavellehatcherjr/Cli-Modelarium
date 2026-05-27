"""Pricing data and cost calculation for all supported models.

Pricing is per 1M tokens, in USD. Verified May 25, 2026 from each provider's
official documentation. LLM pricing changes frequently - re-verify against
the provider's pricing page before relying on these values for production
budgeting.

Schema per entry:
    input         - cost per 1M input tokens (required)
    output        - cost per 1M output tokens (required)
    cached_input  - cost per 1M cached input tokens (optional; typically ~90% off)
    provider      - provider name matching BaseProvider.name (required)
    is_local      - True for local models (optional; always free)
"""

from __future__ import annotations

from cli_modelarium.exceptions import UnknownModelError

PRICING_AS_OF = "2026-05-25"

PRICING: dict[str, dict[str, float | str | bool]] = {
    # ===== OpenAI =====
    "gpt-5.5": {"input": 5.00, "output": 30.00, "cached_input": 0.50, "provider": "openai"},
    "gpt-5.5-pro": {"input": 30.00, "output": 180.00, "provider": "openai"},
    "gpt-5.4": {"input": 2.50, "output": 15.00, "cached_input": 1.25, "provider": "openai"},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50, "cached_input": 0.40, "provider": "openai"},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25, "cached_input": 0.10, "provider": "openai"},
    "gpt-5.4-pro": {"input": 15.00, "output": 60.00, "provider": "openai"},
    "gpt-5.3-codex": {"input": 1.75, "output": 14.00, "provider": "openai"},
    "gpt-5.3-codex-spark": {"input": 0.50, "output": 2.00, "provider": "openai"},
    "o3": {"input": 15.00, "output": 60.00, "cached_input": 7.50, "provider": "openai"},
    "o3-pro": {"input": 30.00, "output": 120.00, "provider": "openai"},
    "o4-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55, "provider": "openai"},
    "gpt-oss-120b": {"input": 0.30, "output": 0.60, "provider": "openai"},
    "gpt-oss-20b": {"input": 0.10, "output": 0.30, "provider": "openai"},
    "gpt-5": {"input": 1.25, "output": 10.00, "provider": "openai"},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "provider": "openai"},
    "gpt-4o": {"input": 2.50, "output": 10.00, "provider": "openai"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "openai"},
    # ===== Anthropic =====
    # Verified against Anthropic's pricing page May 25, 2026.
    "claude-opus-4-7": {
        "input": 5.00,
        "output": 25.00,
        "cached_input": 0.50,
        "provider": "anthropic",
    },
    "claude-sonnet-4-6": {
        "input": 3.00,
        "output": 15.00,
        "cached_input": 0.30,
        "provider": "anthropic",
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cached_input": 0.10,
        "provider": "anthropic",
    },
    "claude-opus-4-6": {
        "input": 5.00,
        "output": 25.00,
        "cached_input": 0.50,
        "provider": "anthropic",
    },
    # ===== Google Gemini (Google uses dots in model IDs) =====
    "gemini-3.5-flash": {"input": 0.50, "output": 3.00, "provider": "google"},
    "gemini-3.1-pro": {"input": 1.25, "output": 5.00, "cached_input": 0.625, "provider": "google"},
    "gemini-3.1-flash-lite": {"input": 0.10, "output": 0.40, "provider": "google"},
    "gemini-3-flash": {"input": 0.30, "output": 2.50, "provider": "google"},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50, "provider": "google"},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40, "provider": "google"},
    # ===== xAI Grok (xAI uses dots in model IDs) =====
    "grok-4.3": {"input": 1.25, "output": 2.50, "cached_input": 0.20, "provider": "xai"},
    "grok-4.20": {"input": 2.00, "output": 6.00, "cached_input": 0.20, "provider": "xai"},
    "grok-4.20-multi-agent-beta": {
        "input": 2.00,
        "output": 6.00,
        "cached_input": 0.20,
        "provider": "xai",
    },
    "grok-4.1-fast": {"input": 0.20, "output": 0.50, "cached_input": 0.05, "provider": "xai"},
    "grok-build-0.1": {"input": 1.00, "output": 2.00, "cached_input": 0.20, "provider": "xai"},
    # ===== DeepSeek =====
    "deepseek-v4-pro": {
        "input": 0.55,
        "output": 2.19,
        "cached_input": 0.055,
        "provider": "deepseek",
    },
    "deepseek-v4-flash": {
        "input": 0.27,
        "output": 1.10,
        "cached_input": 0.027,
        "provider": "deepseek",
    },
    # Legacy aliases - deprecating July 24, 2026; kept for backward compatibility.
    "deepseek-chat": {"input": 0.27, "output": 1.10, "cached_input": 0.027, "provider": "deepseek"},
    "deepseek-reasoner": {
        "input": 0.27,
        "output": 1.10,
        "cached_input": 0.027,
        "provider": "deepseek",
    },
    # ===== Mistral =====
    "mistral-medium-3.5": {"input": 0.40, "output": 2.00, "provider": "mistral"},
    "mistral-medium-latest": {"input": 0.40, "output": 2.00, "provider": "mistral"},
    "mistral-large-latest": {"input": 0.50, "output": 1.50, "provider": "mistral"},
    "mistral-small-latest": {"input": 0.15, "output": 0.60, "provider": "mistral"},
    "codestral-latest": {"input": 0.30, "output": 0.90, "provider": "mistral"},
    "magistral-medium-latest": {"input": 2.00, "output": 5.00, "provider": "mistral"},
    "magistral-small-latest": {"input": 0.50, "output": 1.50, "provider": "mistral"},
    # ===== Groq =====
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79, "provider": "groq"},
    "openai/gpt-oss-120b": {"input": 0.30, "output": 0.60, "provider": "groq"},
    "openai/gpt-oss-safeguard-20b": {"input": 0.10, "output": 0.30, "provider": "groq"},
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "input": 0.20,
        "output": 0.30,
        "provider": "groq",
    },
    # ===== OpenRouter =====
    # OpenRouter aggregates 315+ models behind one API. We register a few
    # representative entries here; users can pass any OpenRouter model ID and
    # we'll route it. Models not in this dict get a fallback cost of $0 (the
    # provider returns real usage data in the response either way).
    "qwen/qwen3.7-max": {"input": 2.50, "output": 7.50, "provider": "openrouter"},
    "qwen/qwen3.5-plus": {"input": 0.30, "output": 1.80, "provider": "openrouter"},
    "qwen/qwen3.6-flash": {"input": 0.19, "output": 1.13, "provider": "openrouter"},
    "qwen/qwen3-coder:free": {"input": 0.0, "output": 0.0, "provider": "openrouter"},
    "deepseek/deepseek-r1:free": {"input": 0.0, "output": 0.0, "provider": "openrouter"},
    "meta-llama/llama-3-3-70b-instruct:free": {
        "input": 0.0,
        "output": 0.0,
        "provider": "openrouter",
    },
    "openai/gpt-oss-120b:free": {"input": 0.0, "output": 0.0, "provider": "openrouter"},
    "zhipuai/glm-4.7-flash:free": {"input": 0.0, "output": 0.0, "provider": "openrouter"},
    # ===== Local =====
    # Wildcard entry. Any model with `local/` prefix resolves here and costs $0.
    "local/*": {"input": 0.0, "output": 0.0, "provider": "local", "is_local": True},
}


def is_local_model(model: str) -> bool:
    """Return True for any model with the `local/` prefix."""
    return model.startswith("local/")


def get_pricing(model: str) -> dict[str, float | str | bool] | None:
    """Look up the pricing entry for a model.

    Local models always resolve to the `local/*` entry.
    Returns None if the model is unknown (callers handle the error).
    """
    if is_local_model(model):
        return PRICING["local/*"]
    return PRICING.get(model)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Return the USD cost for a single completion.

    Cached input tokens use the discounted `cached_input` rate if the model's
    pricing entry includes one; otherwise they fall back to the normal input
    rate. Local models always cost $0.
    """
    if is_local_model(model):
        return 0.0

    pricing = PRICING.get(model)
    if pricing is None:
        raise UnknownModelError(
            f"Unknown model: {model}. Run `cli-modelarium list-models` to see supported models."
        )

    cached_tokens = max(0, min(cached_tokens, input_tokens))
    non_cached = input_tokens - cached_tokens

    cost = (non_cached / 1_000_000) * float(pricing["input"])

    if cached_tokens > 0:
        cached_rate = pricing.get("cached_input", pricing["input"])
        cost += (cached_tokens / 1_000_000) * float(cached_rate)

    cost += (output_tokens / 1_000_000) * float(pricing["output"])

    return cost


def pricing_freshness_note() -> str:
    """Return the standard pricing-freshness disclaimer for user-facing output."""
    return f"Note: Pricing data as of {PRICING_AS_OF}. Verify current pricing at provider websites."
