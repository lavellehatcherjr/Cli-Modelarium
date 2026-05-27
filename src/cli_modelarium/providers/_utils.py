"""Small helpers shared across provider modules."""

from __future__ import annotations

from typing import Any


def extract_retry_after(error: Any) -> float | None:
    """Read a numeric `Retry-After` header from an SDK exception's response.

    Handles either case (`retry-after` or `Retry-After`). Returns None if the
    header is missing or not a parseable number (HTTP-date format is ignored).
    """
    response = getattr(error, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
