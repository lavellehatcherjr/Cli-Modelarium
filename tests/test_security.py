"""Tests for cli_modelarium.security."""
from __future__ import annotations

import pytest

from cli_modelarium import security


class TestNormalizeKey:
    def test_strips_whitespace(self) -> None:
        assert security.normalize_key("  sk-test123  ") == "sk-test123"

    def test_strips_double_quotes(self) -> None:
        assert security.normalize_key('"sk-test123"') == "sk-test123"

    def test_strips_single_quotes(self) -> None:
        assert security.normalize_key("'sk-test123'") == "sk-test123"

    def test_strips_whitespace_then_quotes(self) -> None:
        assert security.normalize_key('  "sk-test123"\n') == "sk-test123"


class TestValidateKey:
    @pytest.mark.parametrize(
        "provider, key",
        [
            ("openai", "sk-proj-abc123XYZ456DEF789ghi0jklmnop"),
            ("openai", "sk-abc123XYZ456DEF789ghi0jklmnop"),
            ("anthropic", "sk-ant-api03-abc123XYZ456DEF789ghi"),
            ("anthropic", "sk-ant-abc123XYZ456DEF789ghi"),
            ("xai", "xai-abc123XYZ456DEF789ghi"),
            ("groq", "gsk_abc123XYZ456DEF789ghi"),
            ("openrouter", "sk-or-abc123XYZ456DEF789ghi"),
            ("deepseek", "sk-abc123XYZ456DEF789ghi"),
            ("mistral", "abc123XYZ456DEF789ghi0"),
        ],
    )
    def test_valid_formats(self, provider: str, key: str) -> None:
        assert security.validate_key(provider, key)

    @pytest.mark.parametrize(
        "provider, key",
        [
            ("openai", "wrong-prefix-1234567890abc"),
            ("openai", "sk-tiny"),
            ("anthropic", "sk-not-ant-1234567890abc"),
            ("xai", "no-prefix-1234567890abc"),
            ("groq", "gsk-not-underscore-1234567890abc"),
            ("openrouter", "sk-not-or-1234567890abc"),
        ],
    )
    def test_invalid_formats(self, provider: str, key: str) -> None:
        assert not security.validate_key(provider, key)

    def test_unknown_provider_returns_true(self) -> None:
        # Cannot validate what we don't have a pattern for; accept and let the
        # provider reject at API time.
        assert security.validate_key("brand-new-provider", "anything-goes-here-1234")


class TestRedactSecrets:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-proj-abc123XYZ456DEF789ghijklmnop",
            "sk-ant-api03-abc123XYZ456DEF789ghi",
            "sk-or-abc123XYZ456DEF789ghi",
            "xai-abc123XYZ456DEF789ghi",
            "gsk_abc123XYZ456DEF789ghi",
            "AIzaSyABC123def456GHI789jkl012MNO345pqr678",
        ],
    )
    def test_each_provider_key_redacted(self, secret: str) -> None:
        message = f"Request failed with key {secret} in body"
        redacted = security.redact_secrets(message)
        assert secret not in redacted
        assert "REDACTED" in redacted

    def test_authorization_header(self) -> None:
        redacted = security.redact_secrets("Authorization: Bearer abc123XYZ456DEF789ghi")
        assert "abc123" not in redacted
        assert "REDACTED" in redacted

    def test_x_api_key_header(self) -> None:
        redacted = security.redact_secrets("x-api-key: sk-someValue1234567890")
        assert "someValue" not in redacted

    def test_api_key_in_query_string(self) -> None:
        redacted = security.redact_secrets(
            "GET https://api.example.com/v1?api_key=abc123XYZ&other=val"
        )
        assert "abc123XYZ" not in redacted
        assert "REDACTED" in redacted

    def test_multiple_secrets_in_one_string(self) -> None:
        message = (
            "Error: sk-proj-aaaaaaaaaaaaaaaaaaaa and sk-ant-bbbbbbbbbbbbbbbbbbbb both failed"
        )
        redacted = security.redact_secrets(message)
        assert "aaaa" not in redacted
        assert "bbbb" not in redacted

    def test_no_secret_passes_through_unchanged(self) -> None:
        assert security.redact_secrets("Hello world") == "Hello world"

    def test_specific_prefixes_not_swallowed_by_generic_sk(self) -> None:
        # sk-ant-* should land on the anthropic placeholder, not the generic one.
        redacted = security.redact_secrets("sk-ant-api03-abc123XYZ456DEF789ghi")
        assert "sk-ant-***REDACTED***" in redacted

        redacted = security.redact_secrets("sk-or-abc123XYZ456DEF789ghi")
        assert "sk-or-***REDACTED***" in redacted

    def test_non_string_input(self) -> None:
        # Defensive: object that stringifies to something with a key.
        class Obj:
            def __str__(self) -> str:
                return "key=sk-proj-abc123XYZ456DEF789ghi"

        redacted = security.redact_secrets(Obj())  # type: ignore[arg-type]
        assert "abc123" not in redacted


class TestKeyringIntegration:
    def test_save_and_load(self) -> None:
        security.save_key("openai", "sk-proj-test1234567890abcdefghi")
        assert security.load_key("openai") == "sk-proj-test1234567890abcdefghi"

    def test_save_strips_whitespace(self) -> None:
        security.save_key("openai", "  sk-proj-test1234567890abcdefghi  \n")
        assert security.load_key("openai") == "sk-proj-test1234567890abcdefghi"

    def test_save_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError):
            security.save_key("openai", "wrong-format-key")

    def test_load_returns_none_when_missing(self) -> None:
        assert security.load_key("openai") is None

    def test_delete_silent_when_missing(self) -> None:
        # Should not raise.
        security.delete_key("openai")

    def test_delete_removes_existing(self) -> None:
        security.save_key("openai", "sk-proj-test1234567890abcdefghi")
        security.delete_key("openai")
        assert security.load_key("openai") is None

    def test_env_var_takes_precedence_over_keychain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        security.save_key("openai", "sk-proj-kerring1234567890abcdefghi")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-fromenv1234567890abcdefghi")
        assert security.load_key("openai") == "sk-proj-fromenv1234567890abcdefghi"

    def test_env_var_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fromenv1234567890abcdefghi")
        assert security.load_key("anthropic") == "sk-ant-fromenv1234567890abcdefghi"

    def test_env_var_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "  sk-proj-padded1234567890abcdefghi  ")
        assert security.load_key("openai") == "sk-proj-padded1234567890abcdefghi"

    def test_is_key_configured_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-test1234567890abcdefghi")
        assert security.is_key_configured("xai")

    def test_is_key_configured_via_keychain(self) -> None:
        security.save_key("groq", "gsk_test1234567890abcdefghi")
        assert security.is_key_configured("groq")

    def test_is_key_configured_returns_false_when_missing(self) -> None:
        assert not security.is_key_configured("openrouter")
