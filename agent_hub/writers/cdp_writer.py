"""Opt-in Electron CDP write adapters for Antigravity App and Codex (ChatGPT Desktop)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any
import urllib.error
from urllib.parse import urlparse
import urllib.request

from ._ws import SimpleWebSocket


class SessionMismatchError(RuntimeError):
    """Raised when the active desktop UI session does not match the requested session_id."""

    def __init__(
        self,
        *,
        app_name: str,
        expected_session_id: str,
        actual_session_id: str = "",
        actual_url: str = "",
        actual_title: str = "",
        reason: str = "",
    ) -> None:
        self.app_name = app_name
        self.expected_session_id = expected_session_id
        self.actual_session_id = actual_session_id
        self.actual_url = actual_url
        self.actual_title = actual_title
        self.reason = reason
        msg = (
            f"{app_name} session mismatch: expected '{expected_session_id}', "
            f"got '{actual_session_id or 'none'}' "
            f"(url={actual_url or '-'}, title={actual_title or '-'})"
        )
        if reason:
            msg += f"; {reason}"
        msg += " — aborted before text injection to prevent cross-talk or unintended new session creation."
        super().__init__(msg)


_EFFORT_ALIASES: dict[str, list[str]] = {
    "none": ["none", "off", "无", "关闭"],
    "minimal": ["minimal", "min", "极低"],
    "low": ["low", "低", "fast", "快"],
    "medium": ["medium", "med", "mid", "中", "standard", "标准"],
    "high": ["high", "高"],
    "xhigh": ["extra high", "extra-high", "xhigh", "x-high", "超高", "极高"],
    "max": ["max", "maximum", "最高", "最大"],
    "ultra": ["ultra", "极限"],
}

_SESSION_URL_RE = re.compile(r"/(?:c|t)/([A-Za-z0-9_-]{6,})")
_ANTIGRAVITY_PATH_RE = re.compile(r"^/c/([A-Za-z0-9_-]{6,})/?$")
_CODEX_PATH_RE = re.compile(r"^/(?:c|t|local)/([A-Za-z0-9_-]{6,})/?$")

_JS_DOM_PRELUDE = """
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const visible = (el) => {
  if (!el) return false;
  const r = el.getBoundingClientRect();
  const st = window.getComputedStyle(el);
  return r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none';
};
const findComposer = () => {
  const selectors = [
    '[aria-label="Message input"]',
    'div.ProseMirror[aria-label*="询问"]',
    'div.ProseMirror[aria-label*="Ask" i]',
    'textarea[placeholder*="Ask"]',
    'textarea[placeholder*="Message"]',
    'textarea[placeholder*="message"]',
    'textarea',
    '[role="textbox"][contenteditable="true"]',
    '[contenteditable="true"]'
  ];
  for (const sel of selectors) {
    const nodes = [...document.querySelectorAll(sel)].filter(visible);
    if (nodes.length > 0) return nodes[nodes.length - 1];
  }
  return null;
};
const readDraft = (el) => {
  if (!el) return '';
  return (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT')
    ? (el.value || '')
    : (el.innerText || el.textContent || '');
};
"""


def _discover_pid_loopback_ports(process_regex: str) -> list[int]:
    """Return TCP LISTEN ports owned by processes matching process_regex (macOS lsof)."""
    try:
        ps_out = subprocess.run(
            ["ps", "-ax", "-o", "pid=,command="],
            text=True,
            capture_output=True,
            timeout=2.5,
        ).stdout
    except Exception:  # noqa: BLE001
        return []

    pattern = re.compile(process_regex)
    pids = [
        line.strip().split(None, 1)[0]
        for line in ps_out.splitlines()
        if pattern.search(line) and "Helper" not in line
    ]
    if not pids:
        return []

    ports: list[int] = []
    try:
        lsof_out = subprocess.run(
            ["lsof", "-nP", "-a", "-iTCP", "-sTCP:LISTEN", "-p", ",".join(pids[:4])],
            text=True,
            capture_output=True,
            timeout=2.5,
        ).stdout
    except Exception:  # noqa: BLE001
        return []

    for line in lsof_out.splitlines():
        m = re.search(r"(?:127\.0\.0\.1|localhost|\*):(\d+)\s+\(LISTEN\)", line)
        if m:
            p = int(m.group(1))
            if p not in ports:
                ports.append(p)
    return ports


class CdpWriter:
    """Thread-safe Electron CDP writer with draft protection and optional session switching."""

    def __init__(
        self,
        *,
        app_name: str,
        env_port_var: str,
        default_port: int,
        process_regex: str,
        launcher_hint: str,
        new_chat_selectors: list[str] | None = None,
        new_chat_texts: list[str] | None = None,
        model_trigger_selectors: list[str] | None = None,
        reasoning_trigger_selectors: list[str] | None = None,
        new_chat_shortcut: dict[str, Any] | None = None,
        switch_verify_timeout_s: float = 2.5,
        switch_poll_interval_s: float = 0.15,
    ) -> None:
        self.app_name = app_name
        self.env_port_var = env_port_var
        self.default_port = default_port
        self.process_regex = process_regex
        self.launcher_hint = launcher_hint
        self.new_chat_selectors = list(new_chat_selectors or [])
        self.new_chat_texts = list(new_chat_texts or [])
        self.model_trigger_selectors = list(model_trigger_selectors or [])
        self.reasoning_trigger_selectors = list(reasoning_trigger_selectors or [])
        self.new_chat_shortcut = dict(new_chat_shortcut or {"key": "n", "code": "KeyN", "modifiers": 4})
        self.switch_verify_timeout_s = max(0.05, float(switch_verify_timeout_s))
        self.switch_poll_interval_s = max(0.01, float(switch_poll_interval_s))
        self._lock = threading.Lock()
        self._msg_id = 0

    def _candidate_ports(self) -> list[int]:
        ports: list[int] = []
        env_raw = os.environ.get(self.env_port_var, "").strip()
        if env_raw.isdigit():
            ports.append(int(env_raw))
        if self.default_port not in ports:
            ports.append(self.default_port)
        for p in _discover_pid_loopback_ports(self.process_regex):
            if p not in ports:
                ports.append(p)
        return ports

    @staticmethod
    def _fetch_targets(port: int) -> list[dict[str, Any]] | None:
        url = f"http://127.0.0.1:{port}/json/list"
        try:
            with urllib.request.urlopen(url, timeout=1.2) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                if isinstance(data, list):
                    return data
        except Exception:  # noqa: BLE001
            return None
        return None

    def _discover_endpoint(self) -> tuple[int | None, list[dict[str, Any]]]:
        for port in self._candidate_ports():
            targets = self._fetch_targets(port)
            if targets is None:
                continue
            pages = [t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
            if pages:
                pages.sort(
                    key=lambda t: (
                        1 if t.get("url", "").startswith("data:") or "overlay" in t.get("url", "").lower() else 0,
                        0 if t.get("title") in ("Antigravity", "ChatGPT") or "chatgpt.com" in t.get("url", "") else 1,
                    )
                )
                return port, pages
        return None, []

    def _probe_dom_readiness(self, ws_url: str) -> dict[str, Any]:
        """Run a fast, strictly read-only DOM probe for composer visibility and unsaved drafts."""
        script = """
        (() => {
          const visible = (el) => {
            if (!el) return false;
            const r = el.getBoundingClientRect();
            const st = window.getComputedStyle(el);
            return r.width > 0 && r.height > 0 && st.visibility !== 'hidden' && st.display !== 'none';
          };
          const selectors = [
            'textarea[placeholder*="Ask"]',
            'textarea[placeholder*="Message"]',
            'textarea[placeholder*="message"]',
            'textarea',
            '[role="textbox"][contenteditable="true"]',
            '[contenteditable="true"]'
          ];
          let composer = null;
          for (const sel of selectors) {
            const nodes = [...document.querySelectorAll(sel)].filter(visible);
            if (nodes.length > 0) { composer = nodes[nodes.length - 1]; break; }
          }
          const raw = composer
            ? ((composer.tagName === 'TEXTAREA' || composer.tagName === 'INPUT')
                ? (composer.value || '')
                : (composer.innerText || composer.textContent || ''))
            : '';
          return {
            composer_found: Boolean(composer),
            has_unsaved_draft: raw.trim().length > 0
          };
        })()
        """
        with SimpleWebSocket(ws_url, timeout=2.0) as ws:
            res = self._cdp_eval(ws, script, await_promise=False)
        return res if isinstance(res, dict) else {}

    def status(self) -> dict[str, Any]:
        """Check whether an Electron CDP page target is currently reachable and probe composer DOM."""
        port, pages = self._discover_endpoint()
        if port is not None:
            info: dict[str, Any] = {
                "app": self.app_name,
                "ready": True,
                "transport": "electron-cdp",
                "endpoint": f"http://127.0.0.1:{port}",
                "page_count": len(pages),
                "page_titles": [str(p.get("title") or "") for p in pages[:6]],
            }
            try:
                probe = self._probe_dom_readiness(str(pages[0]["webSocketDebuggerUrl"]))
                info["composer_found"] = bool(probe.get("composer_found"))
                info["has_unsaved_draft"] = bool(probe.get("has_unsaved_draft"))
            except Exception as exc:  # noqa: BLE001
                info["dom_probe_error"] = str(exc)
            return info
        return {
            "app": self.app_name,
            "ready": False,
            "transport": "electron-cdp",
            "detail": (
                f"No active CDP page target found for {self.app_name}. "
                f"When ready, relaunch via: {self.launcher_hint}"
            ),
        }

    def _cdp_call(self, ws: SimpleWebSocket, method: str, params: dict[str, Any] | None = None) -> Any:
        self._msg_id += 1
        call_id = self._msg_id
        ws.send_text(
            json.dumps(
                {
                    "id": call_id,
                    "method": method,
                    "params": params or {},
                }
            )
        )
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            raw = ws.recv_text()
            msg = json.loads(raw)
            if msg.get("id") != call_id:
                continue
            if "error" in msg:
                raise RuntimeError(f"CDP error: {msg['error']}")
            return msg.get("result")
        raise TimeoutError(f"CDP {method} timed out")

    def _cdp_eval(self, ws: SimpleWebSocket, expression: str, *, await_promise: bool = True) -> Any:
        self._msg_id += 1
        call_id = self._msg_id
        ws.send_text(
            json.dumps(
                {
                    "id": call_id,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": expression,
                        "awaitPromise": await_promise,
                        "returnByValue": True,
                    },
                }
            )
        )
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            raw = ws.recv_text()
            msg = json.loads(raw)
            if msg.get("id") != call_id:
                continue
            if "error" in msg:
                raise RuntimeError(f"CDP error: {msg['error']}")
            result = (msg.get("result") or {}).get("result") or {}
            if (msg.get("result") or {}).get("exceptionDetails"):
                raise RuntimeError(f"CDP exception: {(msg['result']['exceptionDetails']).get('text')}")
            return result.get("value")
        raise TimeoutError("CDP Runtime.evaluate timed out")

    @staticmethod
    def _normalize_effort_aliases(reasoning_effort: str) -> list[str]:
        raw = reasoning_effort.strip().lower()
        if not raw:
            return []
        compact = re.sub(r"[\s_-]+", "", raw)
        if compact in ("extrahigh", "xhigh"):
            return _EFFORT_ALIASES["xhigh"]
        if raw in _EFFORT_ALIASES:
            return _EFFORT_ALIASES[raw]
        if compact in _EFFORT_ALIASES:
            return _EFFORT_ALIASES[compact]
        return [raw]

    def _apply_model_config_via_cdp(
        self,
        ws: SimpleWebSocket,
        *,
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any] | None:
        model_clean = model.strip()
        provider_clean = provider.strip()
        effort_clean = reasoning_effort.strip()
        if not model_clean and not provider_clean and not effort_clean:
            return None

        model_tokens = [t for t in re.split(r"[-_\s/]+", model_clean.lower()) if t]
        provider_tokens = [t for t in re.split(r"[-_\s/]+", provider_clean.lower()) if t]
        effort_aliases = self._normalize_effort_aliases(effort_clean)
        cfg = json.dumps(
            {
                "modelRaw": model_clean.lower(),
                "modelTokens": model_tokens,
                "providerTokens": provider_tokens,
                "effortRaw": effort_clean.lower(),
                "effortAliases": effort_aliases,
                "modelSelectors": self.model_trigger_selectors,
                "reasoningSelectors": self.reasoning_trigger_selectors,
            }
        )
        script = f"""
        (async () => {{
          const cfg = {cfg};
          {_JS_DOM_PRELUDE}
          const esc = () => document.dispatchEvent(new KeyboardEvent('keydown', {{
            key: 'Escape', code: 'Escape', keyCode: 27, which: 27, bubbles: true
          }}));
          const elText = (el) => ((el.innerText || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.getAttribute('title') || '')).trim().toLowerCase();
          const hasEffort = (txt, aliases) => {{
            if (!aliases || aliases.length === 0) return false;
            const wantsXHigh = aliases.some(a => a === 'extra high' || a === 'extra-high' || a === 'xhigh' || a === '超高' || a === '极高');
            const clean = wantsXHigh ? txt : txt.replace(/extra[\\s_-]*high|x[\\s_-]*high|超高|极高/gi, ' ');
            for (const alias of aliases) {{
              if (/^[a-z0-9 .-]+$/.test(alias)) {{
                const re = new RegExp('(?:^|[^a-z0-9])' + alias.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + '(?:[^a-z0-9]|$)', 'i');
                if (re.test(clean)) return true;
              }} else {{
                if (clean.includes(alias)) return true;
              }}
            }}
            return false;
          }};
          const findTrigger = (selectors, fallbackRe) => {{
            for (const sel of selectors) {{
              const hit = [...document.querySelectorAll(sel)].filter(visible)[0];
              if (hit) return hit;
            }}
            return [...document.querySelectorAll('button, [role="button"], [role="combobox"], select')]
              .filter(visible)
              .find((el) => fallbackRe.test(elText(el))) || null;
          }};
          const collectMenuItems = () => {{
            const sels = [
              '[role="menuitem"]',
              '[role="menuitemradio"]',
              '[role="option"]',
              '[role="radio"]',
              '[data-radix-collection-item]',
              '[role="menu"] button',
              '[role="listbox"] button',
              '[role="dialog"] button',
              '[data-radix-popper-content-wrapper] button'
            ];
            const seen = new Set();
            const out = [];
            for (const sel of sels) {{
              for (const el of document.querySelectorAll(sel)) {{
                if (!seen.has(el) && visible(el)) {{
                  seen.add(el);
                  out.push(el);
                }}
              }}
            }}
            return out;
          }};

          const modelBtn = findTrigger(cfg.modelSelectors, /(select model|选择.*模型|model)/i);
          const reasoningBtn = findTrigger(cfg.reasoningSelectors, /(思考强度|reasoning|effort|thinking)/i);
          const currentSummary = ((modelBtn ? elText(modelBtn) : '') + ' ' + (reasoningBtn ? elText(reasoningBtn) : '')).trim();

          const modelAlready = cfg.modelTokens.length === 0 || cfg.modelTokens.every(t => currentSummary.includes(t));
          const effortAlready = cfg.effortAliases.length === 0 || hasEffort(currentSummary, cfg.effortAliases);
          if (modelAlready && effortAlready && currentSummary) {{
            return {{ applied: true, already_selected: true, current: currentSummary.slice(0, 120) }};
          }}

          let modelClicked = '';
          let effortClicked = '';

          if ((!modelAlready || (!effortAlready && !reasoningBtn)) && modelBtn) {{
            modelBtn.click();
            await sleep(250);
            const items = collectMenuItems();
            let bestItem = null;
            let bestScore = 0;
            for (const item of items) {{
              const txt = elText(item);
              if (!txt) continue;
              let score = 0;
              if (cfg.modelTokens.length > 0) {{
                let hits = 0;
                for (const t of cfg.modelTokens) {{
                  if (txt.includes(t)) hits += 1;
                }}
                if (hits === 0) continue;
                score += hits * 10;
                if (hits === cfg.modelTokens.length) score += 20;
                if (cfg.modelRaw && txt.includes(cfg.modelRaw)) score += 15;
              }}
              if (cfg.providerTokens.length > 0 && cfg.providerTokens.every(t => txt.includes(t))) {{
                score += 5;
              }}
              if (cfg.effortAliases.length > 0 && hasEffort(txt, cfg.effortAliases)) {{
                score += 12;
              }}
              if (score > bestScore) {{
                bestScore = score;
                bestItem = item;
              }}
            }}
            if (bestItem && bestScore > 0) {{
              modelClicked = (bestItem.innerText || bestItem.getAttribute('aria-label') || '').trim().slice(0, 120);
              if (cfg.effortAliases.length > 0 && hasEffort(elText(bestItem), cfg.effortAliases)) {{
                effortClicked = modelClicked;
              }}
              bestItem.click();
              await sleep(200);
            }} else {{
              esc();
              await sleep(100);
            }}
          }}

          if (cfg.effortAliases.length > 0 && !effortClicked) {{
            const targetReasoningBtn = reasoningBtn || modelBtn;
            if (targetReasoningBtn) {{
              const afterText = elText(targetReasoningBtn);
              if (hasEffort(afterText, cfg.effortAliases)) {{
                effortClicked = afterText.slice(0, 120);
              }} else {{
                targetReasoningBtn.click();
                await sleep(250);
                const items = collectMenuItems();
                let bestEff = null;
                let bestEffScore = 0;
                for (const item of items) {{
                  const txt = elText(item);
                  if (!txt || !hasEffort(txt, cfg.effortAliases)) continue;
                  let score = 10;
                  if (cfg.modelTokens.length > 0 && cfg.modelTokens.every(t => txt.includes(t))) score += 10;
                  if (score > bestEffScore) {{
                    bestEffScore = score;
                    bestEff = item;
                  }}
                }}
                if (bestEff) {{
                  effortClicked = (bestEff.innerText || bestEff.getAttribute('aria-label') || '').trim().slice(0, 120);
                  bestEff.click();
                  await sleep(200);
                }} else {{
                  esc();
                  await sleep(100);
                }}
              }}
            }}
          }}

          return {{
            applied: Boolean(modelClicked || effortClicked),
            already_selected: false,
            ...(modelClicked ? {{ model_clicked: modelClicked }} : {{}}),
            ...(effortClicked ? {{ effort_clicked: effortClicked }} : {{}}),
            had_model_trigger: Boolean(modelBtn),
            had_reasoning_trigger: Boolean(reasoningBtn)
          }};
        }})()
        """
        try:
            res = self._cdp_eval(ws, script, await_promise=True)
            if isinstance(res, dict):
                return res
            return {"applied": False, "warning": f"unexpected model selection result: {res}"}
        except Exception as exc:  # noqa: BLE001
            return {"applied": False, "warning": str(exc)}

    def _cdp_inject_and_send(self, ws: SimpleWebSocket, text: str) -> dict[str, Any]:
        payload = json.dumps({"text": text})
        prep_script = f"""
        (() => {{
          const cfg = {payload};
          {_JS_DOM_PRELUDE}
          const composer = findComposer();
          if (!composer) {{
            return {{ ok: false, error: 'composer-not-found', message: 'Could not locate visible chat input element via CDP.' }};
          }}
          if (readDraft(composer).trim().length > 0) {{
            return {{ ok: false, error: 'draft-present', message: 'Target session composer has unsaved draft; aborted.' }};
          }}
          composer.focus();
          const isContentEditable = composer.tagName !== 'TEXTAREA' && composer.tagName !== 'INPUT';
          if (!isContentEditable) {{
            const proto = composer.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
            if (setter) setter.call(composer, cfg.text);
            else composer.value = cfg.text;
            composer.dispatchEvent(new Event('input', {{ bubbles: true }}));
          }}
          window.__hermes_active_composer = composer;
          return {{ ok: true, isContentEditable }};
        }})()
        """
        prep = self._cdp_eval(ws, prep_script, await_promise=False)
        if not isinstance(prep, dict) or not prep.get("ok"):
            raise RuntimeError(f"{self.app_name} CDP send prep failed: {prep}")

        if prep.get("isContentEditable"):
            self._cdp_call(ws, "Input.insertText", {"text": text})

        time.sleep(0.15)
        click_script = f"""
        (() => {{
          {_JS_DOM_PRELUDE}
          const composer = window.__hermes_active_composer || document.activeElement;
          const sendBtn = [...document.querySelectorAll('button[aria-label*="Send" i], button[aria-label*="发送"], button[data-testid*="send" i], button.agy-send-button, button[type="submit"]')]
            .filter((b) => visible(b) && !b.disabled)[0];
          if (sendBtn) {{
            sendBtn.click();
          }} else if (composer) {{
            composer.dispatchEvent(new KeyboardEvent('keydown', {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true }}));
          }}
          return {{ ok: true, method: sendBtn ? 'button' : 'enter' }};
        }})()
        """
        res = self._cdp_eval(ws, click_script, await_promise=True)
        if not isinstance(res, dict) or not res.get("ok"):
            raise RuntimeError(f"{self.app_name} CDP send failed: {res}")
        return res

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
        """Trigger a new conversation/thread via Electron CDP, optionally configure model and send initial_message."""
        with self._lock:
            port, pages = self._discover_endpoint()
            if port is None or not pages:
                raise ConnectionError(
                    f"{self.app_name} CDP is not active. Relaunch with {self.launcher_hint} to enable GUI session creation."
                )

            ws_url = str(pages[0]["webSocketDebuggerUrl"])
            ws_name = Path(cwd.strip()).name.lower() if cwd.strip() else ""
            cfg = json.dumps(
                {
                    "workspaceName": ws_name,
                    "newChatSelectors": self.new_chat_selectors,
                    "newChatTexts": self.new_chat_texts,
                }
            )
            new_chat_script = f"""
            (async () => {{
              const cfg = {cfg};
              {_JS_DOM_PRELUDE}
              const preComposer = findComposer();
              if (preComposer && readDraft(preComposer).trim().length > 0) {{
                return {{ ok: false, error: 'draft-present', message: 'Composer has unsaved user draft; aborted to avoid overwriting.' }};
              }}

              const findNewChatBtn = () => {{
                const allBtns = [...document.querySelectorAll('a, button, [role="button"]')].filter(visible);
                if (cfg.workspaceName) {{
                  const wsHit = allBtns.find((el) => {{
                    const hay = ((el.getAttribute('aria-label') || '') + ' ' + (el.getAttribute('title') || '')).toLowerCase();
                    return hay.includes(cfg.workspaceName) && (hay.includes('新建') || hay.includes('new'));
                  }});
                  if (wsHit) return {{ el: wsHit, method: 'workspace-button' }};
                }}
                for (const sel of cfg.newChatSelectors) {{
                  const hit = [...document.querySelectorAll(sel)].filter(visible)[0];
                  if (hit) return {{ el: hit, method: 'selector' }};
                }}
                for (const el of allBtns) {{
                  const txt = (el.innerText || '').trim().toLowerCase();
                  const aria = (el.getAttribute('aria-label') || '').trim().toLowerCase();
                  const hay = txt || aria;
                  if (!hay || hay.length > 80) continue;
                  if (cfg.newChatTexts.some(t => hay === t || txt === t || aria.includes(t))) {{
                    return {{ el, method: 'text' }};
                  }}
                }}
                return null;
              }};

              let found = findNewChatBtn();
              if (!found) {{
                const sidebarToggle = [...document.querySelectorAll('[data-testid="sidebar-toggle"], button[aria-label*="Sidebar" i], button[aria-label*="侧边栏"]')]
                  .filter(visible)[0];
                if (sidebarToggle) {{
                  sidebarToggle.click();
                  await sleep(250);
                  found = findNewChatBtn();
                  if (found) found.method = 'sidebar-' + found.method;
                }}
              }}

              if (found && found.el) {{
                found.el.click();
                await sleep(450);
                return {{ ok: true, clicked: true, method: found.method }};
              }}
              return {{ ok: true, clicked: false, method: 'shortcut-needed' }};
            }})()
            """

            send_res: dict[str, Any] | None = None
            model_selection: dict[str, Any] | None = None
            with SimpleWebSocket(ws_url, timeout=15.0) as ws:
                prep = self._cdp_eval(ws, new_chat_script, await_promise=True)
                if not isinstance(prep, dict) or not prep.get("ok"):
                    raise RuntimeError(f"{self.app_name} CDP create_session failed: {prep}")

                new_chat_method = str(prep.get("method") or "button")
                if not prep.get("clicked"):
                    sc = self.new_chat_shortcut
                    key = str(sc.get("key") or "n")
                    code = str(sc.get("code") or f"Key{key.upper()}")
                    mods = int(sc.get("modifiers") or 4)
                    vk = int(sc.get("windowsVirtualKeyCode") or ord(key.upper()))
                    key_params = {
                        "key": key,
                        "code": code,
                        "modifiers": mods,
                        "windowsVirtualKeyCode": vk,
                        "nativeVirtualKeyCode": vk,
                    }
                    self._cdp_call(ws, "Input.dispatchKeyEvent", {"type": "keyDown", **key_params})
                    self._cdp_call(ws, "Input.dispatchKeyEvent", {"type": "keyUp", **key_params})
                    new_chat_method = "shortcut"
                    time.sleep(0.45)

                model_selection = self._apply_model_config_via_cdp(
                    ws,
                    model=model,
                    provider=provider,
                    reasoning_effort=reasoning_effort,
                )

                if initial_message.strip():
                    send_res = self._cdp_inject_and_send(ws, initial_message)
                    time.sleep(0.4)

                href = self._cdp_eval(ws, "window.location.href", await_promise=False)
                href_str = str(href or "")
                m = _SESSION_URL_RE.search(href_str)
                session_id = m.group(1) if m else f"{self.app_name}-cdp-{int(time.time() * 1000)}"

            out: dict[str, Any] = {
                "ok": True,
                "sessionId": session_id,
                "transport": "electron-cdp",
                "port": port,
                "cwd": cwd,
                "title": title,
                "new_chat_method": new_chat_method,
                "sent_initial_message": bool(initial_message.strip()),
            }
            if send_res is not None:
                out["send_method"] = send_res.get("method", "")
            if model_selection is not None:
                out["modelSelection"] = model_selection
            return out

    @staticmethod
    def _normalize_session_id(raw: Any) -> str:
        s = str(raw or "").strip()
        for prefix in ("chatgpt:conversation:", "chatgpt:", "local:"):
            if s.startswith(prefix):
                s = s[len(prefix) :].strip()
        return s

    def _read_active_session_identity(self, ws: SimpleWebSocket) -> dict[str, Any]:
        """Inspect the currently active UI session identity via read-only CDP evaluation."""
        if self.app_name == "antigravity":
            script = """
            (() => {
              // __agentHubReadSessionIdentity
              const href = String(window.location.href || '');
              let pathname = String(window.location.pathname || '');
              try {
                if (!pathname && href) pathname = new URL(href).pathname || '';
              } catch {}
              const m = pathname.match(/^\\/c\\/([A-Za-z0-9_-]{6,})\\/?$/);
              const sessionId = m ? m[1] : '';
              let title = '';
              if (sessionId) {
                const row = [...document.querySelectorAll('[data-testid="conversation-row-sidebar"][data-cascade-id]')]
                  .find((el) => el.getAttribute('data-cascade-id') === sessionId)
                  || document.querySelector('[data-testid="conversation-row-sidebar"][data-selected="true"]');
                if (row) {
                  title = (row.innerText || row.textContent || '').trim().split('\\n')[0].trim();
                }
              }
              if (!title) {
                title = String(document.title || '').replace(/\\s*-\\s*Antigravity$/i, '').trim();
              }
              return { sessionId, title, href, pathname, source: sessionId ? 'pathname' : 'none' };
            })()
            """
        else:
            script = """
            (() => {
              // __agentHubReadSessionIdentity
              const href = String(window.location.href || '');
              let pathname = String(window.location.pathname || '');
              try {
                if (!pathname && href) pathname = new URL(href).pathname || '';
              } catch {}
              const norm = (v) => String(v || '').trim()
                .replace(/^chatgpt:conversation:/, '')
                .replace(/^chatgpt:/, '')
                .replace(/^local:/, '')
                .trim();
              const isHidden = (el) => {
                if (!el) return true;
                if (el.closest('[hidden], [aria-hidden="true"], [inert]')) return true;
                const st = window.getComputedStyle(el);
                return st.display === 'none' || st.visibility === 'hidden';
              };

              const pm = pathname.match(/^\\/(?:c|t|local)\\/([A-Za-z0-9_-]{6,})\\/?$/);
              if (pm) {
                return {
                  sessionId: norm(pm[1]),
                  title: String(document.title || '').trim(),
                  href,
                  pathname,
                  source: 'pathname',
                };
              }

              const activeThread = document.querySelector(
                '[data-app-action-sidebar-thread-active="true"], [data-app-action-sidebar-thread-selected="true"]'
              );
              if (activeThread) {
                const sid = norm(activeThread.getAttribute('data-app-action-sidebar-thread-id'));
                if (sid) {
                  const ttl = (
                    activeThread.getAttribute('data-app-action-sidebar-thread-title')
                    || activeThread.getAttribute('aria-label')
                    || activeThread.innerText
                    || document.title
                    || ''
                  ).trim();
                  return { sessionId: sid, title: ttl, href, pathname, source: 'sidebar-active-thread' };
                }
              }

              const activeChatgpt = document.querySelector(
                '[data-sidebar-archive-chatgpt-id][aria-current="page"], [data-sidebar-chatgpt-conversation-key][aria-current="page"]'
              );
              if (activeChatgpt) {
                const sid = norm(
                  activeChatgpt.getAttribute('data-sidebar-archive-chatgpt-id')
                  || activeChatgpt.getAttribute('data-sidebar-chatgpt-conversation-key')
                );
                if (sid) {
                  const ttl = (
                    activeChatgpt.getAttribute('aria-label')
                    || activeChatgpt.innerText
                    || document.title
                    || ''
                  ).trim();
                  return { sessionId: sid, title: ttl, href, pathname, source: 'sidebar-active-chatgpt' };
                }
              }

              const domQueries = [
                ['[data-above-composer-conversation-id]', 'data-above-composer-conversation-id', 'composer-above-dom'],
                ['[data-map-composer-conversation]', 'data-map-composer-conversation', 'composer-map-dom'],
                ['main [data-thread-id], [data-active="true"][data-thread-id]', 'data-thread-id', 'main-dom'],
                ['main [data-conversation-id], [data-active="true"][data-conversation-id]', 'data-conversation-id', 'main-dom'],
              ];
              for (const [sel, attr, src] of domQueries) {
                for (const el of document.querySelectorAll(sel)) {
                  if (isHidden(el)) continue;
                  const sid = norm(el.getAttribute(attr));
                  if (sid) {
                    return {
                      sessionId: sid,
                      title: String(document.title || '').trim(),
                      href,
                      pathname,
                      source: src,
                    };
                  }
                }
              }

              return {
                sessionId: '',
                title: String(document.title || '').trim(),
                href,
                pathname,
                source: 'none',
              };
            })()
            """

        raw = self._cdp_eval(ws, script, await_promise=False)
        data = raw if isinstance(raw, dict) else {}
        url = str(data.get("href") or "")
        pathname = str(data.get("pathname") or "")
        if not pathname and url:
            try:
                pathname = urlparse(url).path or ""
            except ValueError:
                pathname = ""
        sid = self._normalize_session_id(data.get("sessionId"))
        if not sid and pathname:
            pat = _ANTIGRAVITY_PATH_RE if self.app_name == "antigravity" else _CODEX_PATH_RE
            m = pat.match(pathname)
            if m:
                sid = self._normalize_session_id(m.group(1))
        return {
            "session_id": sid,
            "title": str(data.get("title") or "").strip(),
            "url": url,
            "pathname": pathname,
            "source": str(data.get("source") or ("pathname" if sid else "none")),
        }

    def _matches_target_session(self, identity: dict[str, Any], expected_session_id: str) -> bool:
        expected = self._normalize_session_id(expected_session_id)
        if not expected:
            return False
        actual_sid = self._normalize_session_id(identity.get("session_id"))
        if self.app_name == "antigravity":
            pathname = str(identity.get("pathname") or "").rstrip("/")
            return pathname == f"/c/{expected}" and actual_sid == expected
        return actual_sid == expected

    def send_message(
        self,
        session_id: str,
        text: str,
        *,
        target_title: str = "",
        mode: str = "queue",
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        """Inject text into the verified target Electron session via CDP with fail-fast session guard."""
        expected_sid = self._normalize_session_id(session_id)
        if not expected_sid:
            raise ValueError(f"{self.app_name} send_message requires a non-empty session_id")

        with self._lock:
            port, pages = self._discover_endpoint()
            if port is None or not pages:
                raise ConnectionError(
                    f"{self.app_name} CDP is not active. Relaunch with {self.launcher_hint} to enable GUI sends."
                )

            target = pages[0]
            matched_page = False
            for p in pages:
                pu = str(p.get("url") or "")
                if f"/c/{expected_sid}" in pu or f"/t/{expected_sid}" in pu:
                    target = p
                    matched_page = True
                    break
            if not matched_page and target_title:
                tl = target_title.lower()
                for p in pages:
                    pt = str(p.get("title", "")).strip().lower()
                    if pt and (tl in pt or pt in tl):
                        target = p
                        break

            ws_url = str(target["webSocketDebuggerUrl"])
            js_payload = json.dumps({"targetTitle": target_title, "sessionId": expected_sid})

            pre_draft_script = f"""
            (() => {{
              {_JS_DOM_PRELUDE}
              const preComposer = findComposer();
              if (preComposer && readDraft(preComposer).trim().length > 0) {{
                return {{ ok: false, error: 'draft-present', message: 'Composer has unsaved user draft; aborted to avoid overwriting.' }};
              }}
              return {{ ok: true }};
            }})()
            """

            switch_script = f"""
            (async () => {{
              const cfg = {js_payload};
              {_JS_DOM_PRELUDE}
              const norm = (v) => String(v || '').trim()
                .replace(/^chatgpt:conversation:/, '')
                .replace(/^chatgpt:/, '')
                .replace(/^local:/, '')
                .trim();
              const isNewChatButton = (el) => {{
                const hay = ((el.innerText || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.getAttribute('title') || '')).toLowerCase();
                return /(?:^|\\b)(?:new chat|new thread|new conversation)(?:$|\\b)|新聊天|新线程|新对话|新建对话|新建聊天/.test(hay);
              }};
              const findBySessionId = (sid) => {{
                if (!sid) return null;
                for (const a of document.querySelectorAll('a[href]')) {{
                  if (!visible(a) || isNewChatButton(a)) continue;
                  const href = String(a.getAttribute('href') || '');
                  if (href === '/c/' + sid || href.startsWith('/c/' + sid + '?') || href.endsWith('/c/' + sid) || href.includes('/' + sid)) {{
                    return {{ el: a, method: 'id-href' }};
                  }}
                }}
                const idAttrs = [
                  'data-cascade-id',
                  'data-app-action-sidebar-thread-id',
                  'data-sidebar-archive-chatgpt-id',
                  'data-sidebar-chatgpt-conversation-key',
                  'data-thread-id',
                  'data-conversation-id',
                ];
                for (const attr of idAttrs) {{
                  for (const el of document.querySelectorAll('[' + attr + ']')) {{
                    if (!visible(el) || isNewChatButton(el)) continue;
                    if (norm(el.getAttribute(attr)) === sid) {{
                      return {{ el, method: 'id-' + attr }};
                    }}
                  }}
                }}
                return null;
              }};

              let hit = findBySessionId(cfg.sessionId);
              if (!hit) {{
                const sidebarToggle = [...document.querySelectorAll('[data-testid="sidebar-toggle"], button[aria-label*="Sidebar" i], button[aria-label*="侧边栏"]')]
                  .filter(visible)[0];
                if (sidebarToggle) {{
                  sidebarToggle.click();
                  await sleep(250);
                  hit = findBySessionId(cfg.sessionId);
                }}
              }}

              if (!hit && cfg.targetTitle) {{
                const needle = cfg.targetTitle.trim().toLowerCase();
                if (needle) {{
                  const scoped = [...document.querySelectorAll(
                    '[data-testid="conversation-row-sidebar"], a[href^="/c/"], [data-sidebar-cycle-item], [data-app-action-sidebar-thread-row], aside a, aside button, nav a, nav button'
                  )].filter((el) => {{
                    if (!visible(el) || isNewChatButton(el)) return false;
                    const txt = (el.innerText || el.getAttribute('aria-label') || '').trim().toLowerCase();
                    return txt && txt.length < 180 && (txt === needle || txt.includes(needle) || (txt.length >= 4 && needle.includes(txt)));
                  }});
                  if (scoped.length > 0) {{
                    hit = {{ el: scoped[0], method: 'sidebar-title' }};
                  }}
                }}
              }}

              if (hit && hit.el) {{
                hit.el.click();
                await sleep(250);
                return {{ ok: true, switched: true, switchMethod: hit.method }};
              }}
              return {{ ok: true, switched: false, switchMethod: 'not-found' }};
            }})()
            """

            post_switch_composer_script = f"""
            (() => {{
              {_JS_DOM_PRELUDE}
              const composer = findComposer();
              if (!composer) {{
                return {{ ok: false, error: 'composer-not-found', message: 'Could not locate visible chat input element via CDP.' }};
              }}
              if (readDraft(composer).trim().length > 0) {{
                return {{ ok: false, error: 'draft-present', message: 'Target session composer has unsaved draft; aborted.' }};
              }}
              return {{ ok: true }};
            }})()
            """

            with SimpleWebSocket(ws_url, timeout=15.0) as ws:
                pre_draft = self._cdp_eval(ws, pre_draft_script, await_promise=False)
                if not isinstance(pre_draft, dict) or not pre_draft.get("ok"):
                    raise RuntimeError(f"{self.app_name} CDP send prep failed: {pre_draft}")

                identity = self._read_active_session_identity(ws)
                switched = False
                switch_method = "already-active"

                if not self._matches_target_session(identity, expected_sid):
                    sw_res = self._cdp_eval(ws, switch_script, await_promise=True)
                    if not isinstance(sw_res, dict) or not sw_res.get("ok"):
                        raise RuntimeError(f"{self.app_name} CDP switch failed: {sw_res}")
                    switched = bool(sw_res.get("switched"))
                    switch_method = str(sw_res.get("switchMethod") or ("clicked" if switched else "not-found"))

                    if switched:
                        deadline = time.monotonic() + self.switch_verify_timeout_s
                        while True:
                            identity = self._read_active_session_identity(ws)
                            if self._matches_target_session(identity, expected_sid):
                                break
                            if time.monotonic() >= deadline:
                                break
                            time.sleep(self.switch_poll_interval_s)
                    else:
                        identity = self._read_active_session_identity(ws)

                    if not self._matches_target_session(identity, expected_sid):
                        raise SessionMismatchError(
                            app_name=self.app_name,
                            expected_session_id=expected_sid,
                            actual_session_id=str(identity.get("session_id") or ""),
                            actual_url=str(identity.get("url") or ""),
                            actual_title=str(identity.get("title") or ""),
                            reason=(
                                f"session switch did not land on target within {self.switch_verify_timeout_s:.1f}s "
                                f"(switch_method={switch_method})"
                                if switched
                                else "could not locate target session in sidebar to switch"
                            ),
                        )

                composer_check = self._cdp_eval(ws, post_switch_composer_script, await_promise=False)
                if not isinstance(composer_check, dict) or not composer_check.get("ok"):
                    raise RuntimeError(f"{self.app_name} CDP send prep failed: {composer_check}")

                model_selection = self._apply_model_config_via_cdp(
                    ws,
                    model=model,
                    provider=provider,
                    reasoning_effort=reasoning_effort,
                )

                final_identity = self._read_active_session_identity(ws)
                if not self._matches_target_session(final_identity, expected_sid):
                    raise SessionMismatchError(
                        app_name=self.app_name,
                        expected_session_id=expected_sid,
                        actual_session_id=str(final_identity.get("session_id") or ""),
                        actual_url=str(final_identity.get("url") or ""),
                        actual_title=str(final_identity.get("title") or ""),
                        reason="active session drifted prior to text injection",
                    )

                res = self._cdp_inject_and_send(ws, text)
                res["switched"] = switched
                verified_sid = expected_sid
                verified_title = str(final_identity.get("title") or target_title or "").strip()

            out = {
                "ok": True,
                "transport": "electron-cdp",
                "port": port,
                "mode": mode,
                "sessionId": verified_sid,
                "verified_session_id": verified_sid,
                "verified_title": verified_title,
                "verified_url": str(final_identity.get("url") or ""),
                "verified_source": str(final_identity.get("source") or ""),
                "switch_method": switch_method,
                **res,
            }
            if model_selection is not None:
                out["modelSelection"] = model_selection
            return out


class AntigravityWriter(CdpWriter):
    """CDP writer targeting Antigravity.app."""

    def __init__(self) -> None:
        super().__init__(
            app_name="antigravity",
            env_port_var="AGENT_HUB_ANTIGRAVITY_CDP_PORT",
            default_port=9224,
            process_regex=r"/Applications/Antigravity\.app/Contents/MacOS/Antigravity\b",
            launcher_hint="~/.hermes/scripts/launch-antigravity-cdp.sh",
            new_chat_selectors=[
                '[data-testid="new-conversation-button"]',
                'a[aria-label*="New Conversation" i]',
                'button[aria-label*="New Conversation" i]',
                '[data-testid*="new-chat" i]',
            ],
            new_chat_texts=["new conversation", "new chat", "新对话", "新建对话"],
            model_trigger_selectors=[
                '[data-testid="model-selector-trigger"]',
                'button[aria-label*="Select model" i]',
                'button[aria-label*="model" i]',
            ],
            reasoning_trigger_selectors=[
                '[data-testid*="reasoning" i]',
                '[data-testid*="thinking" i]',
                '[data-testid*="effort" i]',
                'button[aria-label*="thinking" i]',
                'button[aria-label*="effort" i]',
            ],
            new_chat_shortcut={"key": "L", "code": "KeyL", "modifiers": 12, "windowsVirtualKeyCode": 76},
        )


class CodexWriter(CdpWriter):
    """CDP writer targeting Codex (ChatGPT.app Aperitif)."""

    def __init__(self) -> None:
        super().__init__(
            app_name="codex",
            env_port_var="AGENT_HUB_CODEX_CDP_PORT",
            default_port=9222,
            process_regex=r"/Applications/ChatGPT\.app/Contents/MacOS/",
            launcher_hint="~/.hermes/scripts/launch-codex-cdp.sh",
            new_chat_selectors=[
                '[data-testid*="new-chat" i]',
                '[data-testid*="new-thread" i]',
                'button[aria-label*="新聊天"]',
                'button[aria-label*="新建"][aria-label*="聊天"]',
                'button[aria-label*="New chat" i]',
                'button[aria-label*="New thread" i]',
            ],
            new_chat_texts=["新聊天", "新线程", "new chat", "new thread", "new conversation"],
            model_trigger_selectors=[
                '[data-testid*="model-switch" i]',
                '[data-testid*="model-select" i]',
                'button[aria-label*="选择 ChatGPT 模型"]',
                'button[aria-label*="选择"][aria-label*="模型"]',
                'button[aria-label*="Select model" i]',
                'button[aria-label*="model" i]',
            ],
            reasoning_trigger_selectors=[
                '[data-testid*="reasoning" i]',
                '[data-testid*="effort" i]',
                'button[aria-label*="思考"]',
                'button[aria-label*="reasoning" i]',
                'button[aria-label*="effort" i]',
            ],
            new_chat_shortcut={"key": "n", "code": "KeyN", "modifiers": 4, "windowsVirtualKeyCode": 78},
        )
