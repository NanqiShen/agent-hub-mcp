"""Write adapter for DeepSeek Harness (DSH Desktop) via local Cordis HTTP bridge."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request


DEFAULT_BRIDGE_URL = "http://127.0.0.1:43917"
DEFAULT_TOKEN_PATH = Path.home() / ".hermes" / "agent-hub" / "dsh-bridge.token"


class DshWriter:
    """Talks to the in-process DSH Cordis plugin over authenticated 127.0.0.1 HTTP."""

    app_name = "dsh"

    def __init__(
        self,
        base_url: str | None = None,
        token_path: Path | None = None,
    ) -> None:
        env_url = os.environ.get("AGENT_HUB_DSH_BRIDGE_URL", "").strip()
        self.base_url = (base_url or env_url or DEFAULT_BRIDGE_URL).rstrip("/")
        env_tok = os.environ.get("AGENT_HUB_DSH_BRIDGE_TOKEN_PATH", "").strip()
        self.token_path = token_path or (Path(env_tok).expanduser() if env_tok else DEFAULT_TOKEN_PATH)

    def _load_token(self) -> str:
        if not self.token_path.exists():
            raise FileNotFoundError(f"DSH bridge token not found at {self.token_path}")
        return self.token_path.read_text(encoding="utf-8").strip()

    def _request(self, method: str, route: str, body: dict[str, Any] | None = None, timeout: float = 15.0) -> dict[str, Any]:
        token = self._load_token()
        url = f"{self.base_url}{route}"
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
                parsed = json.loads(raw)
                err = parsed.get("error") or parsed
            except json.JSONDecodeError:
                err = raw
            raise RuntimeError(f"DSH bridge HTTP {exc.code}: {err}") from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"DSH bridge unreachable at {self.base_url}: {exc.reason}") from exc

    def status(self) -> dict[str, Any]:
        """Probe DSH bridge readiness without mutating any state."""
        try:
            health = self._request("GET", "/health", timeout=2.0)
            return {
                "app": self.app_name,
                "ready": bool(health.get("ok")),
                "transport": "cordis-http",
                "endpoint": self.base_url,
                "pid": health.get("pid"),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "app": self.app_name,
                "ready": False,
                "transport": "cordis-http",
                "endpoint": self.base_url,
                "detail": str(exc),
            }

    def send_message(
        self,
        session_id: str,
        text: str,
        *,
        mode: str = "queue",
        request_id: str | None = None,
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        """Inject a message into a real DSH session so it runs and stays visible in UI."""
        payload: dict[str, Any] = {
            "sessionId": session_id,
            "text": text,
            "mode": mode or "queue",
        }
        if request_id:
            payload["requestId"] = request_id
        if model.strip():
            payload["model"] = model.strip()
        if provider.strip():
            payload["provider"] = provider.strip()
        if reasoning_effort.strip():
            payload["reasoningEffort"] = reasoning_effort.strip()
        res = self._request("POST", "/send", payload, timeout=30.0)
        if not res.get("ok"):
            raise RuntimeError(f"DSH bridge send rejected: {res}")
        out = dict(res)
        out["verified_session_id"] = str(res.get("sessionId") or session_id)
        return out

    def create_session(
        self,
        *,
        cwd: str = "",
        title: str = "",
        agent_preset: str = "",
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        """Create a real DSH session via the host gateway."""
        payload: dict[str, Any] = {}
        if cwd:
            payload["cwd"] = cwd
        if title:
            payload["title"] = title
        if agent_preset:
            payload["agentPreset"] = agent_preset
        if model.strip():
            payload["model"] = model.strip()
        if provider.strip():
            payload["provider"] = provider.strip()
        if reasoning_effort.strip():
            payload["reasoningEffort"] = reasoning_effort.strip()
        res = self._request("POST", "/create", payload, timeout=30.0)
        if not res.get("ok"):
            raise RuntimeError(f"DSH bridge create rejected: {res}")
        return res

    def cancel_session(self, session_id: str) -> dict[str, Any]:
        """Cancel an active turn in a DSH session."""
        res = self._request("POST", "/cancel", {"sessionId": session_id}, timeout=15.0)
        if not res.get("ok"):
            raise RuntimeError(f"DSH bridge cancel rejected: {res}")
        return res
