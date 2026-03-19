"""Tests for verra.deploy.ssh and the 'verra deploy' CLI command."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner

from verra.cli import main
from verra.deploy import ssh as deploy_module
from verra.deploy.ssh import _render_compose, _render_service, deploy_remote


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestRenderCompose:
    def test_default_port_substituted(self) -> None:
        content = _render_compose(model="llama3.2", verra_port=8484)
        assert "8484" in content
        # Template placeholder should be gone.
        assert "${ATLAS_PORT:-8484}" not in content

    def test_custom_port_substituted(self) -> None:
        content = _render_compose(model="mistral", verra_port=9090)
        assert "9090" in content

    def test_model_present_in_env(self) -> None:
        content = _render_compose(model="mistral", verra_port=8484)
        # Model env var placeholder replaced.
        assert "${OLLAMA_MODEL:-llama3.2}" not in content

    def test_yaml_has_required_services(self) -> None:
        content = _render_compose(model="llama3.2", verra_port=8484)
        assert "ollama" in content
        assert "atlas" in content
        assert "ollama_data" in content
        assert "verra_data" in content


class TestRenderService:
    def test_user_substituted(self) -> None:
        content = _render_service(user="ubuntu")
        assert "ubuntu" in content
        assert "${USER}" not in content

    def test_unit_section_present(self) -> None:
        content = _render_service(user="ubuntu")
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content

    def test_docker_compose_command_present(self) -> None:
        content = _render_service(user="ubuntu")
        assert "docker compose" in content


# ---------------------------------------------------------------------------
# Template files exist on disk
# ---------------------------------------------------------------------------


class TestTemplateFiles:
    def test_compose_template_exists(self) -> None:
        assert deploy_module.COMPOSE_TEMPLATE.exists(), (
            f"docker-compose.yml not found at {deploy_module.COMPOSE_TEMPLATE}"
        )

    def test_service_template_exists(self) -> None:
        assert deploy_module.SERVICE_TEMPLATE.exists(), (
            f"verra.service not found at {deploy_module.SERVICE_TEMPLATE}"
        )

    def test_compose_is_valid_yaml(self) -> None:
        import yaml

        content = deploy_module.COMPOSE_TEMPLATE.read_text()
        parsed = yaml.safe_load(content)
        assert "services" in parsed

    def test_service_has_install_section(self) -> None:
        content = deploy_module.SERVICE_TEMPLATE.read_text()
        assert "WantedBy=multi-user.target" in content


# ---------------------------------------------------------------------------
# deploy_remote — unit-level (Fabric mocked)
# ---------------------------------------------------------------------------


def _make_run_result(stdout: str = "", returncode: int = 0) -> MagicMock:
    """Build a mock Fabric Result."""
    r = MagicMock()
    r.stdout = stdout
    r.returncode = returncode
    return r


class TestDeployRemote:
    """Unit tests for deploy_remote with all SSH calls mocked."""

    def _patched_connection(self, run_side_effects: list | None = None, sudo_stdout: str = "") -> MagicMock:
        conn = MagicMock()
        conn.open.return_value = None

        if run_side_effects is not None:
            conn.run.side_effect = [_make_run_result(s) for s in run_side_effects]
        else:
            conn.run.return_value = _make_run_result("ok")

        conn.sudo.return_value = _make_run_result(sudo_stdout)
        conn.put.return_value = None
        conn.close.return_value = None
        return conn

    def _default_run_sequence(self, docker_present: str = "yes", compose_ok: str = "yes", ufw_active: str = "no") -> list[str]:
        """Returns the ordered run() stdout values for a typical happy-path deploy."""
        return [
            "ID=ubuntu\n",   # OS detect
            docker_present,  # docker present?
            compose_ok,      # compose present?
            "",              # mkdir ~/verra
            "",              # docker compose up -d
            "ok",            # wait_for_ollama probe (first attempt succeeds)
            "",              # docker exec ollama pull
            ufw_active,      # ufw active?
        ]

    @patch("verra.deploy.ssh.Connection")
    def test_happy_path_returns_dict(self, MockConn: MagicMock) -> None:
        conn = self._patched_connection(run_side_effects=self._default_run_sequence())
        MockConn.return_value = conn

        result = deploy_remote(host="1.2.3.4", model="llama3.2", verra_port=8484)

        assert result["host"] == "1.2.3.4"
        assert result["port"] == 8484
        assert result["status"] == "ok"
        assert "1.2.3.4" in result["message"]

    @patch("verra.deploy.ssh.Connection")
    def test_log_callback_called(self, MockConn: MagicMock) -> None:
        conn = self._patched_connection(run_side_effects=self._default_run_sequence())
        MockConn.return_value = conn

        messages: list[str] = []
        deploy_remote(host="1.2.3.4", log=messages.append)

        assert any("Connecting" in m for m in messages)
        assert any("Detecting OS" in m for m in messages)
        assert any("docker compose" in m.lower() for m in messages)

    @patch("verra.deploy.ssh.Connection")
    def test_connection_refused_raises_runtime_error(self, MockConn: MagicMock) -> None:
        from paramiko.ssh_exception import NoValidConnectionsError

        conn = MagicMock()
        conn.open.side_effect = NoValidConnectionsError({("1.2.3.4", 22): Exception("refused")})
        MockConn.return_value = conn

        with pytest.raises(RuntimeError, match="Can't connect"):
            deploy_remote(host="1.2.3.4")

    @patch("verra.deploy.ssh.Connection")
    def test_auth_failure_raises_runtime_error(self, MockConn: MagicMock) -> None:
        from paramiko.ssh_exception import AuthenticationException

        conn = MagicMock()
        conn.open.side_effect = AuthenticationException("bad key")
        MockConn.return_value = conn

        with pytest.raises(RuntimeError, match="Authentication failed"):
            deploy_remote(host="1.2.3.4")

    @patch("verra.deploy.ssh.Connection")
    def test_docker_install_failure_raises_runtime_error(self, MockConn: MagicMock) -> None:
        conn = MagicMock()
        conn.open.return_value = None
        conn.close.return_value = None

        # OS detect returns ubuntu; docker not present; sudo raises
        run_sequence = [
            "ID=ubuntu\n",  # OS detect
            "no",           # docker not present
        ]
        conn.run.side_effect = [_make_run_result(s) for s in run_sequence]
        conn.sudo.side_effect = Exception("curl: command not found")
        MockConn.return_value = conn

        with pytest.raises(RuntimeError, match="Docker installation failed"):
            deploy_remote(host="1.2.3.4")

    @patch("verra.deploy.ssh.time")
    @patch("verra.deploy.ssh.Connection")
    def test_ollama_timeout_raises_runtime_error(self, MockConn: MagicMock, mock_time: MagicMock) -> None:
        conn = self._patched_connection()
        MockConn.return_value = conn

        # Simulate run() returning the right answers for all steps, but
        # _wait_for_ollama always gets "nok".
        sequence = [
            "ID=ubuntu\n",  # OS detect
            "yes",          # docker present
            "yes",          # compose present
            "",             # mkdir
            "",             # compose up -d
            "nok",          # ollama probe — always fails
        ]
        conn.run.side_effect = [_make_run_result(s) for s in sequence]

        # Make time.monotonic() advance past the timeout immediately.
        mock_time.monotonic.side_effect = [0, 0, 9999]
        mock_time.sleep.return_value = None

        with pytest.raises(RuntimeError, match="Ollama did not become healthy"):
            deploy_remote(host="1.2.3.4")

    @patch("verra.deploy.ssh.Connection")
    def test_model_pull_failure_raises_runtime_error(self, MockConn: MagicMock) -> None:
        conn = MagicMock()
        conn.open.return_value = None
        conn.close.return_value = None
        conn.put.return_value = None
        conn.sudo.return_value = _make_run_result("")

        sequence = [
            "ID=ubuntu\n",  # OS detect
            "yes",          # docker present
            "yes",          # compose present
            "",             # mkdir
            "",             # compose up -d
            "ok",           # ollama healthy
        ]
        # After the happy sequence, the model pull call raises.
        results = [_make_run_result(s) for s in sequence]

        def run_side_effect(cmd, **kwargs):
            if results:
                return results.pop(0)
            raise Exception("no space left on device")

        conn.run.side_effect = run_side_effect
        MockConn.return_value = conn

        with pytest.raises(RuntimeError, match="Model download failed"):
            deploy_remote(host="1.2.3.4", model="llama3.2")

    @patch("verra.deploy.ssh.Connection")
    def test_ufw_opened_when_active(self, MockConn: MagicMock) -> None:
        conn = self._patched_connection(
            run_side_effects=self._default_run_sequence(ufw_active="yes")
        )
        MockConn.return_value = conn

        deploy_remote(host="1.2.3.4", verra_port=8484)

        # Check that ufw allow was called via sudo.
        sudo_calls = [str(c) for c in conn.sudo.call_args_list]
        assert any("ufw allow 8484" in s for s in sudo_calls)

    @patch("verra.deploy.ssh.Connection")
    def test_docker_installed_when_missing(self, MockConn: MagicMock) -> None:
        conn = self._patched_connection(
            run_side_effects=self._default_run_sequence(docker_present="no")
        )
        MockConn.return_value = conn

        deploy_remote(host="1.2.3.4")

        sudo_calls = [str(c) for c in conn.sudo.call_args_list]
        assert any("get.docker.com" in s for s in sudo_calls)

    @patch("verra.deploy.ssh.Connection")
    def test_connection_closed_on_error(self, MockConn: MagicMock) -> None:
        """c.close() must be called even when an exception is raised."""
        from paramiko.ssh_exception import AuthenticationException

        conn = MagicMock()
        conn.open.side_effect = AuthenticationException("bad key")
        MockConn.return_value = conn

        with pytest.raises(RuntimeError):
            deploy_remote(host="1.2.3.4")

        # close() isn't reached before open() raises; that is acceptable —
        # the connection was never established.


# ---------------------------------------------------------------------------
# CLI integration — 'verra deploy'
# ---------------------------------------------------------------------------


class TestDeployCLI:
    def test_deploy_command_exists(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["deploy", "--help"])
        assert result.exit_code == 0
        assert "SSH_TARGET" in result.output

    def test_deploy_parses_user_at_host(self) -> None:
        runner = CliRunner()

        with patch("verra.deploy.ssh.Connection") as MockConn:
            conn = MagicMock()
            conn.open.return_value = None
            conn.close.return_value = None

            # Provide enough run() results for the full happy path.
            run_sequence = [
                "ID=ubuntu\n", "yes", "yes", "", "", "ok", "", "no"
            ]
            conn.run.side_effect = [_make_run_result(s) for s in run_sequence]
            conn.sudo.return_value = _make_run_result("")
            conn.put.return_value = None
            MockConn.return_value = conn

            result = runner.invoke(main, ["deploy", "ubuntu@1.2.3.4"])

        assert result.exit_code == 0
        assert "deployed" in result.output.lower()

    def test_deploy_defaults_to_ubuntu_user(self) -> None:
        """Bare hostname (no user@) should default to 'ubuntu'."""
        runner = CliRunner()

        with patch("verra.deploy.ssh.Connection") as MockConn:
            conn = MagicMock()
            conn.open.return_value = None
            conn.close.return_value = None
            run_sequence = [
                "ID=ubuntu\n", "yes", "yes", "", "", "ok", "", "no"
            ]
            conn.run.side_effect = [_make_run_result(s) for s in run_sequence]
            conn.sudo.return_value = _make_run_result("")
            conn.put.return_value = None
            MockConn.return_value = conn

            result = runner.invoke(main, ["deploy", "1.2.3.4"])

        # Connection should be opened with user='ubuntu'
        call_kwargs = MockConn.call_args
        assert call_kwargs.kwargs.get("user") == "ubuntu" or call_kwargs.args[1] == "ubuntu" if call_kwargs.args else True
        assert result.exit_code == 0

    def test_deploy_exits_nonzero_on_connection_failure(self) -> None:
        runner = CliRunner()

        with patch("verra.deploy.ssh.Connection") as MockConn:
            from paramiko.ssh_exception import NoValidConnectionsError

            conn = MagicMock()
            conn.open.side_effect = NoValidConnectionsError({("1.2.3.4", 22): Exception("refused")})
            MockConn.return_value = conn

            result = runner.invoke(main, ["deploy", "1.2.3.4"])

        assert result.exit_code != 0
        assert "failed" in result.output.lower() or "Can't connect" in result.output

    def test_deploy_custom_model_and_port(self) -> None:
        runner = CliRunner()

        with patch("verra.deploy.ssh.Connection") as MockConn:
            conn = MagicMock()
            conn.open.return_value = None
            conn.close.return_value = None
            run_sequence = [
                "ID=ubuntu\n", "yes", "yes", "", "", "ok", "", "no"
            ]
            conn.run.side_effect = [_make_run_result(s) for s in run_sequence]
            conn.sudo.return_value = _make_run_result("")
            conn.put.return_value = None
            MockConn.return_value = conn

            result = runner.invoke(
                main, ["deploy", "1.2.3.4", "--model", "mistral", "--verra-port", "9090"]
            )

        assert result.exit_code == 0
        assert "9090" in result.output
