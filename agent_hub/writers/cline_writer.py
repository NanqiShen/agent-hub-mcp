"""Write adapter for VS Code Cline (via local VS Code extension bridge, with Cline Hub WS fallback)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
from typing import Any
from urllib.parse import urlparse
import urllib.error
import urllib.request
import uuid

from ._ws import SimpleWebSocket


DEFAULT_BRIDGE_URL = "http://127.0.0.1:43918"
DEFAULT_BRIDGE_TOKEN_PATH = Path.home() / ".hermes" / "agent-hub" / "cline-bridge.token"
DEFAULT_BRIDGE_STATE_PATH = Path.home() / ".hermes" / "agent-hub" / "cline-bridge.json"
DEFAULT_CLINE_LOCK_PATH = Path.home() / ".cline" / "data" / "locks" / "hub" / "production.json"


class ClineWriter:
    """Routes writes to the VS Code Cline bridge first, falling back to Cline Hub WS if active."""

    app_name = "cline"

    def __init__(
        self,
        bridge_url: str | None = None,
        token_path: Path | None = None,
        state_path: Path | None = None,
        lock_path: Path | None = None,
    ) -> None:
        self._explicit_bridge_url = bridge_url
        env_tok = os.environ.get("AGENT_HUB_CLINE_BRIDGE_TOKEN_PATH", "").strip()
        self.token_path = token_path or (Path(env_tok).expanduser() if env_tok else DEFAULT_BRIDGE_TOKEN_PATH)
        env_state = os.environ.get("AGENT_HUB_CLINE_BRIDGE_STATE_PATH", "").strip()
        self.state_path = state_path or (Path(env_state).expanduser() if env_state else DEFAULT_BRIDGE_STATE_PATH)
        env_lock = os.environ.get("AGENT_HUB_CLINE_HUB_LOCK", "").strip()
        self.lock_path = lock_path or (Path(env_lock).expanduser() if env_lock else DEFAULT_CLINE_LOCK_PATH)

    def _resolve_bridge_url(self) -> str:
        if self._explicit_bridge_url:
            return self._explicit_bridge_url.rstrip("/")
        env_url = os.environ.get("AGENT_HUB_CLINE_BRIDGE_URL", "").strip()
        if env_url:
            return env_url.rstrip("/")
        if self.state_path.exists():
            try:
                st = json.loads(self.state_path.read_text(encoding="utf-8"))
                port = int(st.get("port") or 0)
                if port > 0:
                    return f"http://127.0.0.1:{port}"
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        return DEFAULT_BRIDGE_URL

    def _load_bridge_token(self) -> str:
        if not self.token_path.exists():
            raise FileNotFoundError(f"VS Code Cline bridge token not found at {self.token_path}")
        return self.token_path.read_text(encoding="utf-8").strip()

    def _bridge_request(
        self,
        method: str,
        route: str,
        body: dict[str, Any] | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        base_url = self._resolve_bridge_url()
        token = self._load_bridge_token()
        url = f"{base_url}{route}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(raw).get("error") or raw
            except json.JSONDecodeError:
                err = raw
            raise RuntimeError(f"VS Code Cline bridge HTTP {exc.code}: {err}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"VS Code Cline bridge unreachable at {base_url}: {exc.reason}") from exc

    def _probe_vscode_bridge(self) -> dict[str, Any] | None:
        if not self.token_path.exists():
            return None
        try:
            health = self._bridge_request("GET", "/health", timeout=2.0)
            if health.get("ok") and health.get("hasStartNewTask"):
                out: dict[str, Any] = {
                    "app": self.app_name,
                    "ready": True,
                    "transport": "vscode-cline-api",
                    "endpoint": self._resolve_bridge_url(),
                    "workspaces": health.get("workspaces", []),
                    "clineActive": bool(health.get("clineActive")),
                    "hasStartNewTask": bool(health.get("hasStartNewTask")),
                    "hasSendMessage": bool(health.get("hasSendMessage")),
                }
                for key in ("activeTaskId", "hasController", "hasShowTaskWithId"):
                    if key in health:
                        out[key] = health[key]
                return out
        except Exception:  # noqa: BLE001
            return None
        return None

    def _read_hub_lock(self) -> dict[str, Any] | None:
        if not self.lock_path.exists():
            return None
        try:
            data = json.loads(self.lock_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("url"):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def _probe_cline_hub(self) -> dict[str, Any] | None:
        hub = self._read_hub_lock()
        if not hub:
            return None
        u = urlparse(str(hub["url"]))
        host = u.hostname or "127.0.0.1"
        port = u.port or 80
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return {
                    "app": self.app_name,
                    "ready": True,
                    "transport": "cline-hub-ws",
                    "endpoint": f"{host}:{port}",
                }
        except OSError:
            return None

    def status(self) -> dict[str, Any]:
        """Inspect VS Code Cline bridge first, then Cline Hub WebSocket fallback."""
        vsc = self._probe_vscode_bridge()
        if vsc is not None:
            return vsc
        hub = self._probe_cline_hub()
        if hub is not None:
            return hub
        return {
            "app": self.app_name,
            "ready": False,
            "transport": "vscode-cline-api",
            "detail": (
                "In VS Code, run 'Developer: Reload Window' once to activate "
                "local.agent-hub-cline-bridge (127.0.0.1:43918)"
            ),
        }

    def create_session(
        self,
        *,
        cwd: str = "",
        title: str = "",
        initial_message: str = "",
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        """Start a new real Cline task inside the active VS Code Cline sidebar."""
        prompt = (initial_message or title).strip()
        if not prompt:
            raise ValueError("Cline create_session requires non-empty initial_message (or title).")
        vsc = self._probe_vscode_bridge()
        if vsc is None:
            raise ConnectionError(
                "VS Code Cline bridge is not reachable. In VS Code, run 'Developer: Reload Window' once "
                "to activate local.agent-hub-cline-bridge (127.0.0.1:43918)."
            )
        payload: dict[str, Any] = {"text": prompt, "focusSidebar": True}
        if model.strip():
            payload["model"] = model.strip()
        if provider.strip():
            payload["provider"] = provider.strip()
        if reasoning_effort.strip():
            payload["reasoningEffort"] = reasoning_effort.strip()
        res = self._bridge_request("POST", "/new-task", payload, timeout=20.0)
        if not res.get("ok"):
            raise RuntimeError(f"VS Code Cline bridge rejected /new-task: {res}")
        out: dict[str, Any] = {
            "ok": True,
            "sessionId": res.get("sessionId") or "vscode-cline-active",
            "transport": "vscode-cline-api",
            "action": res.get("action", "startNewTask"),
            "workspaces": res.get("workspaces", []),
            "cwd": cwd or (res.get("workspaces")[0] if res.get("workspaces") else ""),
        }
        if "modelSelection" in res:
            out["modelSelection"] = res["modelSelection"]
        return out

    def send_message(
        self,
        session_id: str,
        text: str,
        *,
        mode: str = "active",
        timeout_seconds: int = 180,
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        """Send into active VS Code Cline task (or start new task if mode='new'), falling back to Cline Hub."""
        model_clean = model.strip()
        provider_clean = provider.strip()
        effort_clean = reasoning_effort.strip()
        sid_clean = (session_id or "").strip()
        vsc = self._probe_vscode_bridge()
        if vsc is not None:
            if (mode or "").strip().lower() == "new":
                payload: dict[str, Any] = {"text": text, "focusSidebar": True}
                if model_clean:
                    payload["model"] = model_clean
                if provider_clean:
                    payload["provider"] = provider_clean
                if effort_clean:
                    payload["reasoningEffort"] = effort_clean
                res = self._bridge_request("POST", "/new-task", payload, timeout=20.0)
                if not res.get("ok"):
                    raise RuntimeError(f"VS Code Cline bridge rejected /new-task: {res}")
                resolved_new_sid = str(res.get("sessionId") or "vscode-cline-active")
                return {
                    **res,
                    "ok": True,
                    "transport": "vscode-cline-api",
                    "mode": "new",
                    "sessionId": resolved_new_sid,
                    "verified_session_id": resolved_new_sid,
                }
            payload = {"text": text, "focusSidebar": False}
            if sid_clean:
                payload["sessionId"] = sid_clean
            if model_clean:
                payload["model"] = model_clean
            if provider_clean:
                payload["provider"] = provider_clean
            if effort_clean:
                payload["reasoningEffort"] = effort_clean
            res = self._bridge_request("POST", "/send", payload, timeout=30.0)
            if not res.get("ok"):
                raise RuntimeError(f"VS Code Cline bridge rejected /send: {res}")
            resolved_sid = str(res.get("sessionId") or sid_clean or "vscode-cline-active")
            return {
                **res,
                "ok": True,
                "transport": "vscode-cline-api",
                "mode": "active",
                "sessionId": resolved_sid,
                "verified_session_id": resolved_sid,
            }

        hub = self._read_hub_lock()
        if not hub:
            raise ConnectionError(
                "In VS Code, run 'Developer: Reload Window' once to activate "
                "local.agent-hub-cline-bridge (127.0.0.1:43918)"
            )

        ws_url = str(hub["url"])
        auth_token = str(hub.get("authToken") or "")
        subprotocols = [auth_token] if auth_token else None
        run_id = f"agent-hub-{uuid.uuid4().hex[:10]}"
        ws_payload: dict[str, Any] = {
            "sessionId": session_id,
            "input": text,
            "mode": mode if mode in ("act", "plan", "yolo", "zen") else "act",
            "timeoutSeconds": timeout_seconds,
        }
        if model_clean:
            ws_payload["model"] = model_clean
        if provider_clean:
            ws_payload["provider"] = provider_clean
        if effort_clean:
            ws_payload["reasoningEffort"] = effort_clean
        cmd = {
            "kind": "command",
            "envelope": {
                "version": "v1",
                "command": "run.start",
                "requestId": run_id,
                "clientId": "agent-hub-mcp",
                "sessionId": session_id,
                "payload": ws_payload,
            },
        }
        with SimpleWebSocket(ws_url, timeout=10.0, subprotocols=subprotocols) as ws:
            ws.send_text(json.dumps(cmd))
            ws.sock.settimeout(2.5)
            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline:
                try:
                    raw = ws.recv_text()
                except (TimeoutError, OSError):
                    break
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                env = obj.get("envelope") or {}
                if obj.get("kind") == "reply" and env.get("requestId") == run_id:
                    if env.get("error"):
                        raise RuntimeError(f"Cline Hub rejected run.start: {env['error']}")
                    return {
                        "ok": True,
                        "transport": "cline-hub-ws",
                        "sessionId": session_id,
                        "verified_session_id": session_id,
                        "requestId": run_id,
                        "reply": obj,
                    }
        return {
            "ok": True,
            "transport": "cline-hub-ws",
            "sessionId": session_id,
            "verified_session_id": session_id,
            "requestId": run_id,
            "note": "submitted to Cline Hub",
        }
