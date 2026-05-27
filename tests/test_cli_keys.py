"""CLI-level tests for the `keys delete` subcommand.

Covers two UX bugs fixed together:
    * misleading "Removed X" when nothing was stored
    * `keys delete` accepting any string as a provider name
"""

from __future__ import annotations

from click.testing import CliRunner

from cli_modelarium import security
from cli_modelarium.cli import main as cli_main


class TestKeysDeleteCloudProvider:
    def test_keys_delete_when_stored_prints_removed_green(self) -> None:
        security.save_key("openai", "sk-proj-test1234567890abcdefghi")

        runner = CliRunner()
        result = runner.invoke(cli_main, ["keys", "delete", "openai"])

        assert result.exit_code == 0, result.output
        assert "Removed openai key from keychain." in result.output
        assert security.load_key("openai") is None

    def test_keys_delete_when_not_stored_prints_noop_dim(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["keys", "delete", "openai"])

        assert result.exit_code == 0, result.output
        assert "No openai key was stored." in result.output
        assert "Removed" not in result.output

    def test_keys_delete_unknown_provider_rejects_exit_2(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["keys", "delete", "totallyfakeprovider"])

        assert result.exit_code == 2, result.output
        assert "Unknown provider: totallyfakeprovider" in result.output

    def test_keys_delete_typo_in_provider_name_rejects(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["keys", "delete", "antrhopic"])

        assert result.exit_code == 2, result.output
        assert "Unknown provider: antrhopic" in result.output
        assert "anthropic" in result.output


class TestKeysDeleteLocal:
    def test_keys_delete_local_when_stored_prints_removed_green(self) -> None:
        security.save_local_url("http://localhost:1234/v1")

        runner = CliRunner()
        result = runner.invoke(cli_main, ["keys", "delete", "local"])

        assert result.exit_code == 0, result.output
        assert "Removed saved local provider URL." in result.output
        assert security.load_local_url() is None

    def test_keys_delete_local_when_not_stored_prints_noop_dim(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli_main, ["keys", "delete", "local"])

        assert result.exit_code == 0, result.output
        assert "No saved local provider URL." in result.output
        assert "Removed" not in result.output
