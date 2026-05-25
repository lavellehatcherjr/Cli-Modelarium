"""Security: API key validation, redaction, and OS keychain storage.

Three layered defenses:
    1. validate_key()   - reject malformed keys before they reach the keychain
    2. redact_secrets() - scrub key-shaped strings from any error message
    3. keyring          - store keys in OS-native encrypted storage

Environment variables take precedence over the keychain so CI/CD pipelines
work without touching the system keychain.
"""
from __future__ import annotations

import os
import re

# `keyring` lazily probes its native backends on first access. On Linux the
# default backend is SecretService, which pulls in `secretstorage` ->
# `cryptography` -> a Rust extension that needs the `_cffi_backend` module
# from the `cffi` package. Minimal containers sometimes ship cryptography
# without cffi, which produces a Rust-side panic the moment any keyring call
# runs (including get_keyring()). The fix on the system side is one of:
#
#     pip install cffi                                 (pure-pip environments)
#     apt-get install build-essential libffi-dev       (debian-derived images)
#
# CI must install one of these before running this package's tests.
import keyring
import keyring.errors

SERVICE_NAME = "cli-modelarium"

# Format-validation patterns per provider. These check shape only; a
# correctly-formed key may still be revoked or unauthorized - authentication
# happens at the provider on the first API call.
KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai": re.compile(r"^sk-(?:proj-)?[A-Za-z0-9_-]{20,}$"),
    "anthropic": re.compile(r"^sk-ant-(?:api03-)?[A-Za-z0-9_-]{20,}$"),
    "google": re.compile(r"^[A-Za-z0-9_-]{30,}$"),
    "xai": re.compile(r"^xai-[A-Za-z0-9_-]{20,}$"),
    "deepseek": re.compile(r"^sk-[A-Za-z0-9_-]{20,}$"),
    "mistral": re.compile(r"^[A-Za-z0-9]{20,}$"),
    "groq": re.compile(r"^gsk_[A-Za-z0-9_-]{20,}$"),
    "openrouter": re.compile(r"^sk-or-[A-Za-z0-9_-]{20,}$"),
}

# Ordered most-specific-first so prefixed keys aren't swallowed by the
# generic sk-* pattern.
REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-proj-[A-Za-z0-9_-]{10,}"), "sk-proj-***REDACTED***"),
    (re.compile(r"sk-ant-(?:api03-)?[A-Za-z0-9_-]{10,}"), "sk-ant-***REDACTED***"),
    (re.compile(r"sk-or-[A-Za-z0-9_-]{10,}"), "sk-or-***REDACTED***"),
    (re.compile(r"xai-[A-Za-z0-9_-]{10,}"), "xai-***REDACTED***"),
    (re.compile(r"gsk_[A-Za-z0-9_-]{10,}"), "gsk_***REDACTED***"),
    (re.compile(r"sk-[A-Za-z0-9_-]{10,}"), "sk-***REDACTED***"),
    (re.compile(r"AIza[A-Za-z0-9_-]{20,}"), "AIza***REDACTED***"),
    (
        re.compile(r"Authorization:\s*Bearer\s+\S+", re.IGNORECASE),
        "Authorization: Bearer ***REDACTED***",
    ),
    (
        re.compile(r"x-api-key:\s*\S+", re.IGNORECASE),
        "x-api-key: ***REDACTED***",
    ),
    (
        re.compile(r"api[-_]?key=[A-Za-z0-9_\-.]+", re.IGNORECASE),
        "api_key=***REDACTED***",
    ),
]


def normalize_key(key: str) -> str:
    """Strip common paste artifacts (whitespace, surrounding quotes) from a key."""
    return key.strip().strip('"').strip("'")


def validate_key(provider: str, key: str) -> bool:
    """Return True if `key` matches the expected format for `provider`.

    Providers without a known pattern return True (we cannot reject what we
    cannot validate).
    """
    pattern = KEY_PATTERNS.get(provider)
    if pattern is None:
        return True
    return bool(pattern.match(key))


def redact_secrets(text: str) -> str:
    """Replace key-shaped substrings in `text` with ***REDACTED*** placeholders."""
    if not isinstance(text, str):
        text = str(text)
    for pattern, replacement in REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def save_key(provider: str, key: str) -> None:
    """Save an API key to the OS-native keychain after format validation.

    Raises:
        ValueError: if the key format does not match the provider's pattern.
        keyring.errors.KeyringError: if the keychain backend is unavailable.
    """
    normalized = normalize_key(key)
    if not validate_key(provider, normalized):
        raise ValueError(
            f"Invalid API key format for {provider}. "
            f"Please check the key and try again."
        )
    keyring.set_password(SERVICE_NAME, provider, normalized)


def load_key(provider: str) -> str | None:
    """Load an API key. Environment variables take precedence over the keychain.

    Returns None if no key is configured anywhere (CI/CD-friendly behavior).
    """
    env_var = f"{provider.upper()}_API_KEY"
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value.strip()

    try:
        return keyring.get_password(SERVICE_NAME, provider)
    except keyring.errors.KeyringError:
        return None


def delete_key(provider: str) -> None:
    """Remove an API key from the keychain. Silent if no key was stored."""
    try:
        keyring.delete_password(SERVICE_NAME, provider)
    except (keyring.errors.PasswordDeleteError, keyring.errors.KeyringError):
        pass


def is_key_configured(provider: str) -> bool:
    """Return True if an API key is configured for `provider` (env var or keychain)."""
    return load_key(provider) is not None
