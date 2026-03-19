"""SSH auto-deploy for Verra.

Deploys Verra + Ollama to a remote Linux server via Fabric (SSH).

Steps:
  1. Connect via SSH
  2. Detect OS
  3. Install Docker + Docker Compose plugin if missing
  4. Create ~/verra directory
  5. Upload docker-compose.yml (with model/port substituted)
  6. Start services (docker compose up -d)
  7. Wait for Ollama to be healthy
  8. Pull the selected Ollama model
  9. Install systemd service for auto-restart on reboot
 10. Open firewall port via ufw if ufw is active
 11. Return connection details
"""


from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Callable

# Fabric and Paramiko are listed in pyproject.toml dependencies, so they are
# always available in a correctly installed environment.  Importing at module
# level means tests can patch "verra.deploy.ssh.Connection" cleanly.
from fabric import Connection
from paramiko.ssh_exception import AuthenticationException, NoValidConnectionsError

# Input validation

# Allowed characters for shell-safe identifiers
_SAFE_PATTERN = __import__("re").compile(r"^[a-zA-Z0-9._:/-]+$")


def _validate_shell_arg(value: str, name: str) -> str:
    """Validate that a value is safe to interpolate into a shell command."""
    if not value or not _SAFE_PATTERN.match(value):
        raise ValueError(
            f"Invalid {name}: {value!r} — only alphanumeric, dots, hyphens, "
            f"underscores, colons, and forward slashes are allowed."
        )
    return value


# Template paths

_DEPLOY_DIR = Path(__file__).parent

COMPOSE_TEMPLATE = _DEPLOY_DIR / "docker-compose.yml"
SERVICE_TEMPLATE = _DEPLOY_DIR / "verra.service"

# Public API


def deploy_remote(
    host: str,
    user: str = "ubuntu",
    port: int = 22,
    model: str = "llama3.2",
    verra_port: int = 8484,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Deploy Verra to a remote server via SSH.

    Args:
        host: Remote hostname or IP address.
        user: SSH username.
        port: SSH port number.
        model: Ollama model to pull on the remote (e.g. "llama3.2").
        verra_port: Port to expose the Verra API on the remote host.
        log: Optional progress-message callback. Called with plain strings.

    Returns:
        dict with keys: 'host', 'port', 'status', 'message'

    Raises:
        RuntimeError: If any step fails (wraps the underlying exception with
            a human-readable message).
    """
    # Validate inputs to prevent shell injection
    _validate_shell_arg(host, "host")
    _validate_shell_arg(user, "user")
    _validate_shell_arg(model, "model")
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid SSH port: {port}")
    if not (1 <= verra_port <= 65535):
        raise ValueError(f"Invalid API port: {verra_port}")

    def _log(msg: str) -> None:
        if log:
            log(msg)

    # ------------------------------------------------------------------
    # Step 1 — Connect
    # ------------------------------------------------------------------
    _log(f"Connecting to {user}@{host}:{port} …")
    try:
        c = Connection(host=host, user=user, port=port)
        # Force the connection open so we surface auth errors early.
        c.open()
    except NoValidConnectionsError as exc:
        raise RuntimeError(
            f"Can't connect. Check that SSH is enabled and the IP is correct. ({exc})"
        ) from exc
    except AuthenticationException as exc:
        raise RuntimeError(
            "Authentication failed. Check your username and SSH key."
        ) from exc

    try:
        # ------------------------------------------------------------------
        # Step 2 — Detect OS
        # ------------------------------------------------------------------
        _log("Detecting OS …")
        os_info = _run(c, "cat /etc/os-release 2>/dev/null || echo 'unknown'")
        if "ubuntu" in os_info.lower() or "debian" in os_info.lower():
            _log("  Detected Ubuntu/Debian")
        elif "centos" in os_info.lower() or "rhel" in os_info.lower() or "fedora" in os_info.lower():
            _log("  Detected RHEL/CentOS/Fedora")
        else:
            _log("  OS: unrecognised — proceeding with generic Docker install script")

        # ------------------------------------------------------------------
        # Step 3 — Install Docker if missing
        # ------------------------------------------------------------------
        docker_present = _run(c, "command -v docker >/dev/null 2>&1 && echo yes || echo no").strip()
        if docker_present == "no":
            _log("Docker not found — installing via get.docker.com …")
            try:
                # Security note: curl|sh is the official Docker install method.
                # We download first, then execute, to reduce TOCTOU window.
                _sudo(c, "curl -fsSL https://get.docker.com -o /tmp/get-docker.sh")
                _sudo(c, "sh /tmp/get-docker.sh")
                _sudo(c, "rm -f /tmp/get-docker.sh")
                _sudo(c, f"usermod -aG docker {user}")
            except Exception as exc:
                import logging as _log_mod
                _log_mod.getLogger(__name__).debug("Docker install failed: %s", exc)
                raise RuntimeError(
                    "Docker installation failed. Try installing Docker manually first."
                ) from exc
        else:
            _log("Docker already installed")

        # Ensure Docker Compose plugin is available (v2 `docker compose`).
        compose_ok = _run(
            c, "docker compose version >/dev/null 2>&1 && echo yes || echo no"
        ).strip()
        if compose_ok == "no":
            _log("Docker Compose plugin not found — installing …")
            try:
                # Works for Debian/Ubuntu; RHEL users should pre-install.
                _sudo(
                    c,
                    "apt-get install -y docker-compose-plugin 2>/dev/null || "
                    "yum install -y docker-compose-plugin 2>/dev/null || true",
                )
            except Exception as exc:
                import logging as _log_mod
                _log_mod.getLogger(__name__).debug("Compose plugin install failed: %s", exc)
                raise RuntimeError(
                    "Docker Compose plugin installation failed. Try installing it manually."
                ) from exc
        else:
            _log("Docker Compose plugin already available")

        # ------------------------------------------------------------------
        # Step 4 — Create ~/verra directory
        # ------------------------------------------------------------------
        _log("Creating ~/verra directory …")
        _run(c, "mkdir -p ~/verra")

        # ------------------------------------------------------------------
        # Step 5 — Upload docker-compose.yml
        # ------------------------------------------------------------------
        _log("Uploading docker-compose.yml …")
        compose_content = _render_compose(model=model, verra_port=verra_port)
        c.put(io.BytesIO(compose_content.encode()), remote="~/verra/docker-compose.yml")

        # ------------------------------------------------------------------
        # Step 6 — Start services
        # ------------------------------------------------------------------
        _log("Starting Verra + Ollama via docker compose …")
        try:
            _run(c, "cd ~/verra && docker compose up -d")
        except Exception as exc:
            import logging as _log_mod
            _log_mod.getLogger(__name__).debug("Docker compose up failed: %s", exc)
            raise RuntimeError(
                "Failed to start services. Check Docker logs on the server."
            ) from exc

        # ------------------------------------------------------------------
        # Step 7 — Wait for Ollama to be healthy
        # ------------------------------------------------------------------
        _log("Waiting for Ollama to become healthy …")
        _wait_for_ollama(c, timeout=60)

        # ------------------------------------------------------------------
        # Step 8 — Pull the Ollama model
        # ------------------------------------------------------------------
        _log(f"Pulling Ollama model '{model}' (this may take a few minutes) …")
        try:
            _run(c, f"docker exec verra-ollama ollama pull {model}", timeout=600)
        except Exception as exc:
            raise RuntimeError(
                "Model download failed. Check disk space on the server."
            ) from exc
        _log(f"  Model '{model}' ready")

        # ------------------------------------------------------------------
        # Step 9 — Install systemd service
        # ------------------------------------------------------------------
        _log("Installing systemd service for auto-restart …")
        service_content = _render_service(user=user)
        c.put(io.BytesIO(service_content.encode()), remote="/tmp/verra.service")
        _sudo(c, "mv /tmp/verra.service /etc/systemd/system/verra.service")
        _sudo(c, "systemctl daemon-reload")
        _sudo(c, "systemctl enable verra.service")
        _log("  Systemd service 'verra' enabled")

        # ------------------------------------------------------------------
        # Step 10 — Configure firewall (ufw) if active
        # ------------------------------------------------------------------
        ufw_active = _run(
            c,
            "command -v ufw >/dev/null 2>&1 && ufw status | grep -q 'Status: active' "
            "&& echo yes || echo no",
        ).strip()
        if ufw_active == "yes":
            _log(f"Opening firewall port {verra_port}/tcp via ufw …")
            _sudo(c, f"ufw allow {verra_port}/tcp")
        else:
            _log("ufw not active — skipping firewall configuration")

        # ------------------------------------------------------------------
        # Done
        # ------------------------------------------------------------------
        message = (
            f"Verra deployed successfully to {host}. "
            f"Access the API at http://{host}:{verra_port}"
        )
        _log(message)
        return {"host": host, "port": verra_port, "status": "ok", "message": message}

    finally:
        c.close()


# Internal helpers


def _run(c, cmd: str, timeout: int = 120) -> str:
    """Run a shell command and return stdout as a string."""
    result = c.run(cmd, hide=True, warn=True, timeout=timeout)
    return result.stdout or ""


def _sudo(c, cmd: str, timeout: int = 120) -> str:
    """Run a shell command with sudo and return stdout."""
    result = c.sudo(cmd, hide=True, warn=True, timeout=timeout)
    return result.stdout or ""


def _wait_for_ollama(c, timeout: int = 60) -> None:
    """Poll until Ollama's HTTP endpoint responds or timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = _run(
            c,
            "curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && echo ok || echo nok",
        ).strip()
        if probe == "ok":
            return
        time.sleep(3)
    raise RuntimeError(
        "Ollama did not become healthy within the timeout. "
        "Check `docker logs verra-ollama` on the server."
    )


def _render_compose(model: str, verra_port: int) -> str:
    """Read the docker-compose.yml template and substitute variables."""
    template = COMPOSE_TEMPLATE.read_text()
    template = template.replace("${ATLAS_PORT:-8484}", str(verra_port))
    template = template.replace("${OLLAMA_MODEL:-llama3.2}", model)
    return template


def _render_service(user: str) -> str:
    """Read the systemd unit template and substitute the home directory user."""
    template = SERVICE_TEMPLATE.read_text()
    template = template.replace("${USER}", user)
    return template
