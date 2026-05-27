"""Custom exception classes for Cli Modelarium."""

from __future__ import annotations


class ModelariumError(Exception):
    """Base class for all Cli Modelarium exceptions."""


class ProviderError(ModelariumError):
    """Raised when a provider API call fails."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class RateLimitError(ProviderError):
    """Raised when a provider returns HTTP 429."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.retry_after = retry_after


class ProviderOverloadedError(ProviderError):
    """Raised when Anthropic returns HTTP 529 overloaded_error (distinct from 429)."""


class AuthenticationError(ProviderError):
    """Raised when API authentication fails (HTTP 401/403)."""


class ConfigurationError(ModelariumError):
    """Raised when configuration is invalid or missing."""


class KeyNotConfiguredError(ConfigurationError):
    """Raised when an API key is required but not configured."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"No API key configured for {provider}.\n"
            f"  Run: cli-modelarium keys set {provider}\n"
            f"  Or set environment variable: {provider.upper()}_API_KEY"
        )


class InvalidKeyFormatError(ConfigurationError):
    """Raised when an API key fails format validation."""


class UnknownModelError(ConfigurationError):
    """Raised when a requested model is not in the registry."""


class UnknownProviderError(ConfigurationError):
    """Raised when a requested provider is not recognized."""


class CostLimitExceededError(ModelariumError):
    """Raised when estimated or actual cost exceeds the user's --max-cost ceiling."""


class BatchValidationError(ModelariumError):
    """Raised when a batch file fails validation (format, content)."""


class BatchSizeError(BatchValidationError):
    """Raised when batch dimensions exceed safety limits without --force-large."""


class OutputFormatError(ModelariumError):
    """Raised when the output format cannot be inferred or is unsupported."""


class AssertionConfigError(BatchValidationError):
    """Raised when an assertion config is malformed (unknown type, missing value)."""


class LocalURLError(ConfigurationError):
    """Raised when a local provider URL fails the localhost-only safety check."""
