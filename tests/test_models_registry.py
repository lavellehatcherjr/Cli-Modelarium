"""Tests for cli_modelarium.models_registry."""

from __future__ import annotations

import pytest

from cli_modelarium.exceptions import UnknownModelError
from cli_modelarium.models_registry import (
    DYNAMIC_GROUPS,
    MODEL_GROUPS,
    all_known_providers,
    expand_group,
    get_provider_for_model,
    is_group_name,
    list_models_for_provider,
    parse_models_arg,
)


class TestGetProviderForModel:
    @pytest.mark.parametrize(
        "model, provider",
        [
            ("gpt-5.5", "openai"),
            ("claude-opus-4-7", "anthropic"),
            ("gemini-3.1-pro", "google"),
            ("grok-4.3", "xai"),
            ("deepseek-v4-pro", "deepseek"),
            ("mistral-large-latest", "mistral"),
            ("llama-3.3-70b-versatile", "groq"),
            ("local/llama-3.3", "local"),
            ("local/any-string", "local"),
        ],
    )
    def test_routes_correctly(self, model: str, provider: str) -> None:
        assert get_provider_for_model(model) == provider

    def test_unknown_raises(self) -> None:
        with pytest.raises(UnknownModelError):
            get_provider_for_model("definitely-not-a-model")


class TestListModelsForProvider:
    def test_openai_models_present(self) -> None:
        models = list_models_for_provider("openai")
        assert "gpt-5.5" in models
        assert "o3" in models

    def test_models_sorted(self) -> None:
        models = list_models_for_provider("openai")
        assert models == sorted(models)

    def test_wildcard_entry_excluded(self) -> None:
        # local/* is a wildcard entry, not a real model ID
        models = list_models_for_provider("local")
        assert "local/*" not in models

    def test_unknown_provider_returns_empty(self) -> None:
        assert list_models_for_provider("nonexistent-provider") == []


class TestAllKnownProviders:
    def test_includes_all_eight_cloud_providers_plus_local(self) -> None:
        providers = set(all_known_providers())
        expected = {
            "openai",
            "anthropic",
            "google",
            "xai",
            "deepseek",
            "mistral",
            "groq",
            "local",
        }
        # The registry should contain at least the 8 cloud providers + local.
        # (Phase 3 will add openrouter; for now assert what's present.)
        assert expected.issubset(providers)


class TestIsGroupName:
    @pytest.mark.parametrize(
        "name",
        ["all-premium", "all-flagship", "all-budget", "all-reasoning", "all-local", "all"],
    )
    def test_recognized_groups(self, name: str) -> None:
        assert is_group_name(name)

    def test_unknown_group(self) -> None:
        assert not is_group_name("all-unicorns")

    def test_plain_model_not_a_group(self) -> None:
        assert not is_group_name("gpt-5.5")


class TestExpandGroup:
    def test_premium_includes_flagships(self) -> None:
        expanded = expand_group("all-premium")
        assert "gpt-5.5" in expanded
        assert "claude-opus-4-7" in expanded
        assert "gemini-3.1-pro" in expanded

    def test_budget_includes_cheap_models(self) -> None:
        expanded = expand_group("all-budget")
        assert "claude-haiku-4-5" in expanded
        assert "gemini-3.1-flash-lite" in expanded

    def test_reasoning_includes_o3(self) -> None:
        expanded = expand_group("all-reasoning")
        assert "o3" in expanded

    def test_dynamic_groups_resolve_empty(self) -> None:
        # all-local and all are resolved at runtime by the caller.
        assert expand_group("all-local") == []
        assert expand_group("all") == []

    def test_unknown_group_returns_empty(self) -> None:
        assert expand_group("does-not-exist") == []


class TestDynamicGroups:
    def test_only_all_and_all_local(self) -> None:
        assert DYNAMIC_GROUPS == frozenset({"all-local", "all"})


class TestParseModelsArg:
    def test_single_model(self) -> None:
        assert parse_models_arg("gpt-5.5") == ["gpt-5.5"]

    def test_multiple_models(self) -> None:
        assert parse_models_arg("gpt-5.5,claude-opus-4-7") == ["gpt-5.5", "claude-opus-4-7"]

    def test_whitespace_tolerated(self) -> None:
        assert parse_models_arg(" gpt-5.5 , claude-opus-4-7 ") == [
            "gpt-5.5",
            "claude-opus-4-7",
        ]

    def test_empty_tokens_dropped(self) -> None:
        assert parse_models_arg("gpt-5.5,,") == ["gpt-5.5"]

    def test_group_expanded_in_place(self) -> None:
        expanded = parse_models_arg("all-premium")
        assert "gpt-5.5" in expanded
        assert "claude-opus-4-7" in expanded
        assert "all-premium" not in expanded

    def test_group_and_explicit_mixed(self) -> None:
        expanded = parse_models_arg("all-premium,local/llama-3.3")
        assert "claude-opus-4-7" in expanded
        assert "local/llama-3.3" in expanded

    def test_dynamic_group_preserved_for_caller(self) -> None:
        # all-local should be passed through so the caller can resolve it.
        expanded = parse_models_arg("all-local")
        assert expanded == ["all-local"]

    def test_dynamic_all_preserved(self) -> None:
        expanded = parse_models_arg("all")
        assert expanded == ["all"]


class TestGroupCoverage:
    """Make sure every group name promised to the user is registered."""

    @pytest.mark.parametrize(
        "group_name",
        ["all-premium", "all-budget", "all-flagship", "all-reasoning", "all-local"],
    )
    def test_user_promised_groups_exist(self, group_name: str) -> None:
        assert group_name in MODEL_GROUPS
