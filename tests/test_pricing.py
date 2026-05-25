"""Tests for cli_modelarium.pricing."""
from __future__ import annotations

import pytest

from cli_modelarium.exceptions import UnknownModelError
from cli_modelarium.pricing import (
    PRICING,
    PRICING_AS_OF,
    calculate_cost,
    get_pricing,
    is_local_model,
    pricing_freshness_note,
)


class TestPricingAsOf:
    def test_constant_format(self) -> None:
        assert PRICING_AS_OF == "2026-05-25"

    def test_freshness_note_includes_date(self) -> None:
        assert PRICING_AS_OF in pricing_freshness_note()


class TestIsLocalModel:
    def test_local_prefix(self) -> None:
        assert is_local_model("local/llama-3.3-70b")

    def test_no_prefix(self) -> None:
        assert not is_local_model("gpt-5.5")

    def test_local_anywhere_else(self) -> None:
        assert not is_local_model("not-local/anything")


class TestGetPricing:
    def test_known_cloud_model(self) -> None:
        entry = get_pricing("claude-opus-4-7")
        assert entry is not None
        assert entry["provider"] == "anthropic"

    def test_local_resolves_to_wildcard(self) -> None:
        entry = get_pricing("local/anything-goes")
        assert entry is not None
        assert entry.get("is_local") is True

    def test_unknown_returns_none(self) -> None:
        assert get_pricing("nonexistent-model-xyz") is None


class TestCalculateCost:
    def test_input_only(self) -> None:
        # claude-opus-4-7: $5.00 input / $25.00 output per 1M tokens
        cost = calculate_cost("claude-opus-4-7", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(5.00)

    def test_output_only(self) -> None:
        cost = calculate_cost("claude-opus-4-7", input_tokens=0, output_tokens=1_000_000)
        assert cost == pytest.approx(25.00)

    def test_input_plus_output(self) -> None:
        # Sonnet 4.6: $3.00 input / $15.00 output
        cost = calculate_cost("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(18.00)

    def test_haiku_4_5_user_specified_pricing(self) -> None:
        # User-corrected: $1.00 / $5.00 per 1M.
        cost = calculate_cost("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost == pytest.approx(6.00)

    def test_cached_tokens_apply_cached_rate(self) -> None:
        # claude-opus-4-7 cached_input = $0.50/M.
        cost = calculate_cost(
            "claude-opus-4-7",
            input_tokens=1_000_000,
            output_tokens=0,
            cached_tokens=1_000_000,
        )
        assert cost == pytest.approx(0.50)

    def test_partially_cached_input(self) -> None:
        # 500k regular @ $5/M + 500k cached @ $0.50/M = $2.50 + $0.25 = $2.75
        cost = calculate_cost(
            "claude-opus-4-7",
            input_tokens=1_000_000,
            output_tokens=0,
            cached_tokens=500_000,
        )
        assert cost == pytest.approx(2.75)

    def test_cached_without_cached_rate_falls_back_to_input(self) -> None:
        # gpt-5.5-pro doesn't have cached_input; cached tokens should bill at input rate.
        # gpt-5.5-pro input = $30/M
        cost = calculate_cost(
            "gpt-5.5-pro",
            input_tokens=1_000_000,
            output_tokens=0,
            cached_tokens=1_000_000,
        )
        assert cost == pytest.approx(30.00)

    def test_cached_clamped_to_input(self) -> None:
        # Caller passing cached > input should be clamped (defensive).
        cost = calculate_cost(
            "claude-opus-4-7",
            input_tokens=1_000,
            output_tokens=0,
            cached_tokens=10_000_000,
        )
        # All 1000 input tokens billed at cached rate.
        assert cost == pytest.approx(0.50 * 1_000 / 1_000_000)

    def test_local_model_is_free(self) -> None:
        assert calculate_cost("local/llama-3.3-70b", 100_000, 100_000) == 0.0
        assert calculate_cost("local/anything", 1_000_000, 1_000_000, cached_tokens=999) == 0.0

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(UnknownModelError):
            calculate_cost("totally-fake-model", 100, 100)

    def test_zero_tokens(self) -> None:
        assert calculate_cost("claude-opus-4-7", 0, 0) == 0.0


class TestPricingTableCoverage:
    """Smoke checks across the registry so adding a model later isn't silently broken."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.5",
            "gpt-5.4-mini",
            "o3",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "gemini-3.5-flash",
            "gemini-3.1-pro",
            "grok-4.3",
            "grok-4.1-fast",
            "deepseek-v4-pro",
            "mistral-large-latest",
            "llama-3.3-70b-versatile",
        ],
    )
    def test_every_headline_model_has_provider(self, model: str) -> None:
        entry = PRICING[model]
        assert "provider" in entry
        assert "input" in entry
        assert "output" in entry

    def test_anthropic_opus_47_pricing_user_corrected(self) -> None:
        # Phase 1 review: user explicitly corrected Opus 4.7 to $5/$25.
        entry = PRICING["claude-opus-4-7"]
        assert entry["input"] == 5.00
        assert entry["output"] == 25.00

    def test_anthropic_haiku_45_pricing_user_corrected(self) -> None:
        entry = PRICING["claude-haiku-4-5"]
        assert entry["input"] == 1.00
        assert entry["output"] == 5.00
