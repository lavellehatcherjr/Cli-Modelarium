"""Tests for `--models all` / `--models all-local` dynamic-group resolution.

Resolution lives in the cli.py caller (not in parse_models_arg, which stays a
pure pass-through). These cover the cloud half (`all`), the live half
(`all-local`, mocked - never a real server), dedupe, exclusions, the
zero-keys/no-server error paths, and batch parity. All headless.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from cli_modelarium import security
from cli_modelarium.cli import (
    _resolve_all_cloud,
    _resolve_all_local,
    _resolve_dynamic_groups,
)
from cli_modelarium.cli import main as cli_main
from cli_modelarium.pricing import PRICING
from cli_modelarium.providers.local_provider import LocalProvider


def _models_for(*providers: str) -> set[str]:
    return {
        m
        for m, e in PRICING.items()
        if e.get("provider") in providers and not m.endswith("/*")
    }


# ===== (a) `all` cloud resolution =====


def test_all_resolves_to_configured_providers_models() -> None:
    security.save_key("openai", "sk-proj-test1234567890abcdefghi")
    security.save_key("anthropic", "sk-ant-test1234567890abcdefghi")

    resolved = _resolve_dynamic_groups(["all"], None)

    assert set(resolved) == _models_for("openai", "anthropic")
    # deduped + only those two providers
    assert len(resolved) == len(set(resolved))
    assert all(PRICING[m]["provider"] in ("openai", "anthropic") for m in resolved)


# ===== (b) `all` with zero keys -> clear error, exit 2 =====


def test_all_zero_keys_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="No API keys configured"):
        _resolve_dynamic_groups(["all"], None)


def test_compare_all_zero_keys_exits_2_not_unknown_model() -> None:
    result = CliRunner().invoke(cli_main, ["compare", "--models", "all", "hi"])
    assert result.exit_code == 2, result.output
    assert "No API keys configured" in result.output
    # Proves resolution happened: the old failure was "Unknown model: all".
    assert "Unknown model" not in result.output


# ===== (c) `all-local` happy path (mocked discovery) =====


def test_all_local_maps_discovered_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_discover(url: str) -> list[dict[str, Any]]:
        return [{"id": "llama3.3"}, {"id": "qwen2.5"}]

    monkeypatch.setattr(LocalProvider, "discover_models", fake_discover)
    resolved = _resolve_dynamic_groups(["all-local"], None)
    assert resolved == ["local/llama3.3", "local/qwen2.5"]


# ===== (d) `all-local` unreachable -> graceful, no traceback [C1] =====


def test_all_local_unreachable_returns_empty_no_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(url: str) -> list[dict[str, Any]]:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(LocalProvider, "discover_models", boom)
    # The helper must swallow httpx errors and return [] (the outer try/except
    # does NOT catch httpx) - no exception, no hang.
    assert _resolve_all_local(None) == []


# ===== (e) exclusions: `all` never includes local/* or OpenRouter =====


def test_all_excludes_local_and_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure providers whose presence would otherwise leak local/openrouter ids.
    security.save_key("openai", "sk-proj-test1234567890abcdefghi")
    security.save_key("openrouter", "sk-or-test1234567890abcdefghi")
    resolved = _resolve_all_cloud()
    assert all(not m.startswith("local/") for m in resolved)
    assert all(PRICING[m]["provider"] != "openrouter" for m in resolved)
    assert all(PRICING[m]["provider"] != "local" for m in resolved)


# ===== (f) dedupe [C2] =====


def test_all_plus_explicit_model_dedupes() -> None:
    security.save_key("openai", "sk-proj-test1234567890abcdefghi")
    resolved = _resolve_dynamic_groups(["all", "gpt-5.5"], None)
    assert resolved.count("gpt-5.5") == 1


# ===== (g) batch parity =====


def test_batch_all_resolves_via_batch_seam(tmp_path: Any) -> None:
    batch_file = tmp_path / "prompts.txt"
    batch_file.write_text("hello\n")
    result = CliRunner().invoke(cli_main, ["batch", str(batch_file), "--models", "all"])
    # Zero keys -> resolver's clear error (not "Unknown model: all") -> exit 2.
    assert result.exit_code == 2, result.output
    assert "No API keys configured" in result.output
    assert "Unknown model" not in result.output
