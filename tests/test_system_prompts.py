"""Tests for cli_modelarium.io_safety.load_system_prompt.

Security model under test:
    * size limit (1 MB) is enforced
    * directory paths rejected
    * UTF-8 decoded, BOM tolerated
    * symlink target size is what counts (Path.stat follows symlinks)
    * path traversal is NOT blocked - users explicitly pick the path
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cli_modelarium.io_safety import (
    SYSTEM_PROMPT_MAX_BYTES,
    load_system_prompt,
    safe_input_path,
)


class TestLoadSystemPrompt:
    def test_valid_utf8_with_whitespace(self, tmp_path: Path) -> None:
        path = tmp_path / "prompt.txt"
        path.write_text("  You are a helpful assistant.\nBe concise.\n  ", encoding="utf-8")

        result = load_system_prompt(str(path))

        # Leading/trailing whitespace stripped; interior preserved.
        assert result == "You are a helpful assistant.\nBe concise."

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_system_prompt(str(missing))

        assert str(missing) in str(exc_info.value)

    def test_directory_path_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError) as exc_info:
            load_system_prompt(str(tmp_path))

        assert "regular file" in str(exc_info.value)

    def test_file_too_large_raises_value_error(self, tmp_path: Path) -> None:
        big = tmp_path / "huge.txt"
        # Just past the 1 MB threshold.
        big.write_bytes(b"x" * (SYSTEM_PROMPT_MAX_BYTES + 1))

        with pytest.raises(ValueError) as exc_info:
            load_system_prompt(str(big))

        msg = str(exc_info.value)
        assert "too large" in msg.lower()
        # The error message should give the user actionable size info.
        assert "MB" in msg

    def test_empty_file_returns_empty_string(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")

        assert load_system_prompt(str(path)) == ""

    def test_whitespace_only_file_returns_empty_string(self, tmp_path: Path) -> None:
        path = tmp_path / "spaces.txt"
        path.write_text("   \n\t  \n", encoding="utf-8")

        assert load_system_prompt(str(path)) == ""

    def test_utf8_with_bom_is_stripped(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.txt"
        # Editors like notepad.exe add a UTF-8 BOM. We use utf-8-sig to strip it.
        path.write_bytes(b"\xef\xbb\xbfHello world")

        assert load_system_prompt(str(path)) == "Hello world"

    def test_non_utf8_raises_unicode_decode_error(self, tmp_path: Path) -> None:
        """Surfacing UnicodeDecodeError is preferable to silently mangling text."""
        path = tmp_path / "latin1.txt"
        path.write_bytes(b"\xe9\xe9\xe9")  # valid latin-1, invalid UTF-8

        with pytest.raises(UnicodeDecodeError):
            load_system_prompt(str(path))

    def test_path_traversal_is_not_blocked_but_documented(self, tmp_path: Path) -> None:
        """Path traversal handling is intentionally NOT a feature.

        The user typed `../foo` on their CLI - they know what they're
        targeting. We don't second-guess. Security is via size + encoding,
        documented in the module docstring.
        """
        # Create a real file and verify a traversal path that resolves to it
        # is loaded without complaint.
        actual = tmp_path / "real.txt"
        actual.write_text("loaded via traversal", encoding="utf-8")

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        traversal_path = subdir / ".." / "real.txt"

        result = load_system_prompt(str(traversal_path))
        assert result == "loaded via traversal"

    def test_symlink_to_large_file_size_check_still_works(self, tmp_path: Path) -> None:
        """Path.stat() follows symlinks, so a symlink to an oversize file
        is still caught by the size guard.
        """
        target = tmp_path / "big_target.txt"
        target.write_bytes(b"x" * (SYSTEM_PROMPT_MAX_BYTES + 1))

        link = tmp_path / "link.txt"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            pytest.skip("Platform doesn't support symlink creation here")

        with pytest.raises(ValueError) as exc_info:
            load_system_prompt(str(link))

        assert "too large" in str(exc_info.value).lower()

    def test_tilde_expansion(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """~ in the path expands to the user's home (verified via $HOME override)."""
        monkeypatch.setenv("HOME", str(tmp_path))
        path = tmp_path / "from_home.txt"
        path.write_text("expanded", encoding="utf-8")

        result = load_system_prompt("~/from_home.txt")

        assert result == "expanded"


class TestSafeInputPath:
    def test_returns_resolved_absolute_path(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("hi", encoding="utf-8")

        result = safe_input_path(str(path), max_size_bytes=1000)

        assert result.is_absolute()
        assert result == path.resolve()

    def test_relative_path_is_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "rel.txt"
        path.write_text("rel", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = safe_input_path("rel.txt", max_size_bytes=1000)

        assert result == path.resolve()

    def test_max_size_bytes_is_parameter(self, tmp_path: Path) -> None:
        """Confirms the size limit is per-caller; Phase 7 batch uses 10 MB."""
        path = tmp_path / "small.txt"
        path.write_bytes(b"x" * 500)

        # 100 byte limit rejects.
        with pytest.raises(ValueError):
            safe_input_path(str(path), max_size_bytes=100)

        # 1000 byte limit accepts.
        result = safe_input_path(str(path), max_size_bytes=1000)
        assert result == path.resolve()
