"""Tests for the startup banner.

Verifies the banner:
- Renders without error
- Is hidden when stdout is not a TTY (the default under CliRunner)
- Is shown when stdout IS a TTY
- Never contaminates stdout (it goes to stderr)
- Does not change the bare-invocation exit code
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from cli_modelarium.banner import (
    _BLUE,
    _PURPLE,
    _gradient_line,
    render_banner,
    should_show_banner,
)
from cli_modelarium.cli import main as cli_main


class TestBannerHelpers:
    def test_gradient_line_preserves_text(self) -> None:
        """The gradient styling must not alter the characters."""
        line = "Cli Modelarium"
        result = _gradient_line(line, _BLUE, _PURPLE)
        assert result.plain == line

    def test_gradient_line_empty(self) -> None:
        result = _gradient_line("", _BLUE, _PURPLE)
        assert result.plain == ""

    def test_render_banner_runs_without_error(self) -> None:
        """render_banner should not raise (it writes to stderr)."""
        render_banner()  # Should complete cleanly


class TestShouldShowBanner:
    def test_hidden_when_not_tty(self) -> None:
        """Non-TTY (pipe/redirect/CI) must hide the banner."""
        with patch("sys.stdout.isatty", return_value=False):
            assert should_show_banner() is False

    def test_shown_when_tty(self) -> None:
        """Interactive terminal should allow the banner."""
        with patch("sys.stdout.isatty", return_value=True):
            assert should_show_banner() is True

    def test_defensive_on_isatty_error(self) -> None:
        """If isatty is unavailable, default to hidden."""
        with patch("sys.stdout.isatty", side_effect=ValueError):
            assert should_show_banner() is False


class TestBareInvocation:
    def test_bare_invocation_exit_zero(self) -> None:
        """Bare `cli-modelarium` must still exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli_main, [])
        assert result.exit_code == 0

    def test_bare_invocation_shows_help(self) -> None:
        """Help text must still print on bare invocation."""
        runner = CliRunner()
        result = runner.invoke(cli_main, [])
        # CliRunner output combines stdout; help text should be present.
        assert "Usage" in result.output or "Commands" in result.output

    def test_banner_hidden_under_clirunner(self) -> None:
        """CliRunner simulates non-TTY, so the banner must NOT appear.

        This proves the isatty gate works: in CI/test/pipe contexts the
        banner is suppressed.
        """
        runner = CliRunner()
        result = runner.invoke(cli_main, [])
        # The banner tagline must not leak into output under non-TTY.
        assert "Statistically rigorous LLM comparison" not in result.output

    def test_banner_shown_when_tty_forced(self) -> None:
        """When isatty is forced True, the banner code path is exercised.

        We patch should_show_banner's isatty source. The banner goes to
        stderr; under CliRunner the call must still not raise and the
        command must still exit 0.
        """
        runner = CliRunner()
        with patch("sys.stdout.isatty", return_value=True):
            result = runner.invoke(cli_main, [])
        assert result.exit_code == 0


class TestSubcommandsUnaffected:
    def test_compare_help_no_banner(self) -> None:
        """`compare --help` must not show the banner."""
        runner = CliRunner()
        result = runner.invoke(cli_main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "Statistically rigorous LLM comparison" not in result.output

    def test_version_no_banner(self) -> None:
        """`--version` must not show the banner."""
        runner = CliRunner()
        result = runner.invoke(cli_main, ["--version"])
        assert result.exit_code == 0
        assert "Statistically rigorous LLM comparison" not in result.output

    def test_help_no_banner(self) -> None:
        """`--help` must not show the banner."""
        runner = CliRunner()
        result = runner.invoke(cli_main, ["--help"])
        assert result.exit_code == 0
        assert "Statistically rigorous LLM comparison" not in result.output
