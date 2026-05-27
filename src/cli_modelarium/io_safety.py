"""Path safety helpers for user-provided input files.

Used by:
    * --system-prompt-file (Phase 6)
    * batch input files (Phase 7)

Security model
==============

This module does NOT block path traversal (`../foo`, `/etc/passwd`, etc.).
The user explicitly typed the path on their CLI; we trust their intent.
What we DO protect against:

    * Reading directories, special devices, or sockets by accident.
    * Loading multi-gigabyte files that would OOM the process.
    * Silently consuming garbage from a non-UTF-8 file - we read with
      `utf-8-sig` so BOMs are tolerated but mismatched encodings surface as
      a clear UnicodeDecodeError rather than producing mojibake downstream.

If you ever want to restrict paths (e.g. a "must be inside repo root" check
for batch mode CI workflows), add it on top - don't bake it in here, because
this module is shared by every command that loads a user file.
"""

from __future__ import annotations

from pathlib import Path

# Maximum size for a system-prompt file. 1 MB allows for very long prompts
# (a typical system prompt is well under 10 KB) while preventing OOM.
SYSTEM_PROMPT_MAX_BYTES = 1_000_000  # 1 MB

# Phase 7 batch files can be larger - bumped to 10 MB.
BATCH_INPUT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB


def safe_input_path(user_path: str, *, max_size_bytes: int) -> Path:
    """Resolve a user-provided path and check it's safe to read.

    Returns the resolved `Path` on success.

    Raises:
        FileNotFoundError: the path does not exist.
        ValueError: the path is not a regular file, or the file exceeds
            `max_size_bytes`.
    """
    path = Path(user_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a regular file: {path}")
    size = path.stat().st_size
    if size > max_size_bytes:
        raise ValueError(
            f"File too large: {size / 1_048_576:.2f} MB "
            f"(max {max_size_bytes / 1_048_576:.2f} MB at {path})"
        )
    return path


def split_escaped_csv(value: str) -> list[str]:
    """Split a comma-separated string with `\\,` as a literal-comma escape.

    Whitespace around each piece is stripped; empty pieces (e.g. trailing
    comma) are dropped. Any other backslash is kept verbatim.

    Used by --system-prompts, --judge-criteria, --expected-facts. Lives in
    io_safety so the various callers don't have to depend on cli.py.

    Example: `split_escaped_csv(r"a,b,c\\,d")` -> `["a", "b", "c,d"]`
    """
    out: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(value):
        c = value[i]
        if c == "\\" and i + 1 < len(value) and value[i + 1] == ",":
            buf.append(",")
            i += 2
            continue
        if c == ",":
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    last = "".join(buf).strip()
    if last:
        out.append(last)
    return out


def load_system_prompt(file_path: str) -> str:
    """Load a system prompt from disk.

    Returns the file contents with leading/trailing whitespace stripped.
    Empty files yield an empty string.

    Limits:
        * Size: 1 MB (see `SYSTEM_PROMPT_MAX_BYTES`).
        * Encoding: UTF-8. A leading BOM is tolerated (`utf-8-sig`).

    Security: the file path is trusted - the user typed it. We do not
    block path traversal. We DO prevent the foot-guns (size + encoding).
    See the module docstring for the full rationale.
    """
    path = safe_input_path(file_path, max_size_bytes=SYSTEM_PROMPT_MAX_BYTES)
    text = path.read_text(encoding="utf-8-sig")
    return text.strip()
