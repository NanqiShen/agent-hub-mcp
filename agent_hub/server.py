"""Agent Hub MCP stdio server exposing session inspection, task bindings, and write/relay bridges."""

from __future__ import annotations

import inspect
import json
import logging
import re
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from .adapters import AntigravityAdapter, ClineAdapter, CodexAdapter, DshAdapter
from .audit import AuditLog
from .bindings import BindingStore
from .models import ACTIVE_STATUSES, AppAdapter, VALID_APPS, format_local_iso
from .watcher import (
    WatchStore,
    WatchdogRunner,
    parse_notify_events,
    parse_notify_targets,
)
from .writers import AntigravityWriter, ClineWriter, CodexWriter, DshWriter, SessionMismatchError


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("agent_hub")

_APP_ALIASES = {
    "codex": "codex",
    "chatgpt": "codex",
    "dsh": "dsh",
    "deepseek": "dsh",
    "deepseek-harness": "dsh",
    "deepseek_harness": "dsh",
    "antigravity": "antigravity",
    "agy": "antigravity",
    "cline": "cline",
    "vscode": "cline",
    "vscode-cline": "cline",
}

_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_TOKEN_SPLIT_RE = re.compile(r"[\s,，。；;!?！？/|]+")
_FUZZY_STOPWORDS = frozenset({"the", "and", "for", "not", "with", "from", "that", "this", "的", "了", "是", "在", "我", "你"})
_SESSION_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    re.compile(r"^session-[0-9a-fA-F-]{8,}$"),
    re.compile(r"^\d{10,}_[A-Za-z0-9]{3,}$"),
    re.compile(r"^(?:antigravity|codex)-cdp-\d+$"),
)


def _looks_like_session_id(value: str) -> bool:
    s = (value or "").strip()
    return any(p.match(s) for p in _SESSION_ID_PATTERNS)


def normalize_app_name(app: str) -> str:
    """Normalize user/model app hints into canonical adapter names or 'all'/'alias'."""
    key = (app or "all").strip().lower()
    if key in ("all", "*", ""):
        return "all"
    if key == "alias":
        return "alias"
    return _APP_ALIASES.get(key, key)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _extract_last_code_block(text: str) -> tuple[str, bool]:
    """Extract the longest (breaking ties to latest) non-empty fenced code block."""
    blocks = [m.group(1).strip() for m in _CODE_BLOCK_RE.finditer(text or "") if m.group(1).strip()]
    if not blocks:
        return text, False
    best = max(enumerate(blocks), key=lambda pair: (len(pair[1]), pair[0]))[1]
    return best, True


class HubService:
    """Orchestrates the 4 desktop read adapters, write bridges, audit log, and task bindings."""

    def __init__(
        self,
        adapters: dict[str, AppAdapter] | None = None,
        bindings: BindingStore | None = None,
        writers: dict[str, Any] | None = None,
        audit: AuditLog | None = None,
        watches: WatchStore | None = None,
    ) -> None:
        self.adapters: dict[str, AppAdapter] = adapters or {
            "codex": CodexAdapter(),
            "dsh": DshAdapter(),
            "antigravity": AntigravityAdapter(),
            "cline": ClineAdapter(),
        }
        self.bindings = bindings or BindingStore()
        self.writers: dict[str, Any] = writers or {
            "dsh": DshWriter(),
            "codex": CodexWriter(),
            "antigravity": AntigravityWriter(),
            "cline": ClineWriter(),
        }
        self.audit = audit or AuditLog()
        self.watches = watches or WatchStore()

    def _target_apps(self, app: str) -> tuple[list[str], str | None]:
        norm = normalize_app_name(app)
        if norm == "all":
            return list(self.adapters.keys()), None
        if norm in self.adapters:
            return [norm], None
        return [], f"Unknown app '{app}'. Expected one of: {', '.join(VALID_APPS)}, or 'all'."

    @staticmethod
    def _fuzzy_match_tier(summary: Any, query_low: str) -> int:
        title_low = str(getattr(summary, "title", "") or "").lower()
        if query_low == title_low:
            return 3
        if query_low in title_low:
            return 2
        preview_low = str(getattr(summary, "preview", "") or "").lower()
        ws_low = str(getattr(summary, "workspace", "") or "").lower()
        if query_low in preview_low or query_low in ws_low:
            return 1
        return 0

    def _fuzzy_resolve_target(self, candidate_apps: list[str], query: str) -> tuple[str, str] | None:
        q = query.strip()
        if not q:
            return None
        q_low = q.lower()
        matches: list[tuple[int, int, str, str]] = []

        for cand_app in candidate_apps:
            adapter = self.adapters.get(cand_app)
            if adapter is None:
                continue
            try:
                hits = adapter.search_sessions(q, limit=10)
            except Exception as exc:  # noqa: BLE001
                logger.warning("_resolve_target search failed for %s: %s", cand_app, exc)
                continue
            for s in hits:
                tier = self._fuzzy_match_tier(s, q_low)
                matches.append((tier, int(getattr(s, "updated_at_ms", 0) or 0), cand_app, s.session_id))

        if not matches and " " in q:
            raw_tokens = [tok for tok in _TOKEN_SPLIT_RE.split(q_low) if len(tok) >= 2 and tok not in _FUZZY_STOPWORDS]
            seen_tok: set[str] = set()
            distinct_tokens: list[str] = []
            for tok in raw_tokens:
                if tok not in seen_tok:
                    seen_tok.add(tok)
                    distinct_tokens.append(tok)
            if len(distinct_tokens) >= 2:
                picked_tokens = sorted(distinct_tokens, key=len, reverse=True)[:3]
                summary_map: dict[tuple[str, str], Any] = {}
                per_token_pairs: list[set[tuple[str, str]]] = []
                for tok in picked_tokens:
                    tok_pairs: set[tuple[str, str]] = set()
                    for cand_app in candidate_apps:
                        adapter = self.adapters.get(cand_app)
                        if adapter is None:
                            continue
                        try:
                            hits = adapter.search_sessions(tok, limit=15)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("_resolve_target token search failed for %s: %s", cand_app, exc)
                            continue
                        for s in hits:
                            pair = (cand_app, s.session_id)
                            tok_pairs.add(pair)
                            summary_map.setdefault(pair, s)
                    per_token_pairs.append(tok_pairs)
                if per_token_pairs:
                    common = set.intersection(*per_token_pairs)
                    for cand_app, sid in common:
                        s = summary_map[(cand_app, sid)]
                        tier = self._fuzzy_match_tier(s, q_low)
                        matches.append((tier, int(getattr(s, "updated_at_ms", 0) or 0), cand_app, sid))

        if not matches:
            return None
        matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return matches[0][2], matches[0][3]

    def _resolve_target(self, app: str, session_id: str) -> tuple[str | None, str | None, str | None]:
        """Resolve app + session_id (supporting 'alias', exact IDs, and fuzzy phrase/content search)."""
        norm_app = normalize_app_name(app)
        sid = (session_id or "").strip()

        resolved = None
        if norm_app == "alias" or sid.lower().startswith("alias:"):
            resolved = self.bindings.resolve(sid)
            if resolved is None:
                return None, None, f"Task alias '{sid}' is not bound."
        elif sid:
            maybe = self.bindings.resolve(sid)
            if maybe is not None and (norm_app in ("all", str(maybe.get("app"))) or norm_app not in self.adapters):
                resolved = maybe

        if resolved is not None:
            norm_app = str(resolved["app"])
            sid = str(resolved["session_id"])

        if norm_app != "all" and norm_app not in self.adapters:
            return None, None, f"Unknown app '{app}'. Expected one of: {', '.join(VALID_APPS)} (or 'alias', 'all')."
        if not sid:
            return None, None, "session_id must not be empty."

        if norm_app == "cline" and sid in ("active", "vscode-cline-active"):
            return "cline", sid, None

        candidate_apps = list(self.adapters.keys()) if norm_app == "all" else [norm_app]
        for cand_app in candidate_apps:
            adapter = self.adapters.get(cand_app)
            if adapter is None:
                continue
            try:
                meta = adapter.get_session_meta(sid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("_resolve_target get_session_meta failed for %s: %s", cand_app, exc)
                continue
            if meta is not None:
                return cand_app, meta.session_id, None

        if _looks_like_session_id(sid):
            if norm_app in self.adapters:
                return norm_app, sid, None
            return None, None, f"Session ID '{sid}' not found across apps: {', '.join(candidate_apps)}."

        fuzzy = self._fuzzy_resolve_target(candidate_apps, sid)
        if fuzzy is not None:
            return fuzzy[0], fuzzy[1], None

        if norm_app in self.adapters:
            return norm_app, sid, None
        return None, None, f"No session matching '{sid}' found across apps: {', '.join(candidate_apps)}."

    def overview(self, limit_per_app: int = 5) -> dict[str, Any]:
        limit_per_app = max(1, min(int(limit_per_app), 20))
        alias_map = self.bindings.aliases_by_session()
        apps_summary: dict[str, Any] = {}

        for name, adapter in self.adapters.items():
            try:
                recent = adapter.list_sessions(limit=max(limit_per_app, 10))
                active = [s for s in recent if s.status in ACTIVE_STATUSES]
                status_counts: dict[str, int] = {}
                for s in recent:
                    status_counts[s.status] = status_counts.get(s.status, 0) + 1
                apps_summary[name] = {
                    "recent_sampled": len(recent),
                    "active_or_attention_count": len(active),
                    "status_counts_in_sample": status_counts,
                    "top_sessions": [
                        s.to_dict(bound_aliases=alias_map.get((name, s.session_id)))
                        for s in recent[:limit_per_app]
                    ],
                }
            except Exception as exc:  # noqa: BLE001
                logger.warning("overview failed for %s: %s", name, exc)
                apps_summary[name] = {"error": str(exc)}

        return {
            "apps": apps_summary,
            "bindings": self.list_bindings().get("bindings", []),
        }

    def list_sessions(
        self,
        *,
        app: str = "all",
        active_only: bool = False,
        include_archived: bool = False,
        include_subagents: bool = False,
        limit: int = 20,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 100))
        target_apps, err = self._target_apps(app)
        if err:
            return {"error": err}

        alias_map = self.bindings.aliases_by_session()
        collected: list[dict[str, Any]] = []
        errors: dict[str, str] = {}

        per_app_limit = limit if len(target_apps) == 1 else max(limit, 15)
        for name in target_apps:
            try:
                items = self.adapters[name].list_sessions(
                    active_only=active_only,
                    include_archived=include_archived,
                    include_subagents=include_subagents,
                    limit=per_app_limit,
                )
                for s in items:
                    collected.append(s.to_dict(bound_aliases=alias_map.get((name, s.session_id))))
            except Exception as exc:  # noqa: BLE001
                logger.warning("list_sessions failed for %s: %s", name, exc)
                errors[name] = str(exc)

        collected.sort(key=lambda item: item.get("updated_at_ms", 0), reverse=True)
        payload: dict[str, Any] = {
            "app_filter": normalize_app_name(app),
            "active_only": active_only,
            "count": min(len(collected), limit),
            "sessions": collected[:limit],
        }
        if errors:
            payload["adapter_errors"] = errors
        return payload

    def search_sessions(
        self,
        query: str,
        *,
        app: str = "all",
        limit: int = 15,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 50))
        target_apps, err = self._target_apps(app)
        if err:
            return {"error": err}

        q = (query or "").strip()
        q_low = q.lower()
        alias_map = self.bindings.aliases_by_session()
        seen_pairs: set[tuple[str, str]] = set()
        results: list[dict[str, Any]] = []
        errors: dict[str, str] = {}

        if q_low:
            for alias, entry in self.bindings.list_all().items():
                b_app = str(entry.get("app", ""))
                b_sid = str(entry.get("session_id", ""))
                if b_app not in target_apps:
                    continue
                if q_low in alias.lower() or q_low in str(entry.get("note", "")).lower():
                    pair = (b_app, b_sid)
                    if pair in seen_pairs:
                        continue
                    try:
                        meta = self.adapters[b_app].get_session_meta(b_sid)
                        if meta is not None:
                            d = meta.to_dict(bound_aliases=alias_map.get(pair))
                            d["matched_by"] = f"binding:{alias}"
                            results.append(d)
                            seen_pairs.add(pair)
                    except Exception as exc:  # noqa: BLE001
                        errors[b_app] = str(exc)

        per_app_limit = limit if len(target_apps) == 1 else max(limit, 10)
        adapter_hits: list[dict[str, Any]] = []
        for name in target_apps:
            try:
                hits = self.adapters[name].search_sessions(q, limit=per_app_limit)
                for s in hits:
                    pair = (name, s.session_id)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    adapter_hits.append(s.to_dict(bound_aliases=alias_map.get(pair)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("search_sessions failed for %s: %s", name, exc)
                errors[name] = str(exc)

        adapter_hits.sort(key=lambda item: item.get("updated_at_ms", 0), reverse=True)
        combined = (results + adapter_hits)[:limit]
        payload: dict[str, Any] = {
            "query": q,
            "app_filter": normalize_app_name(app),
            "count": len(combined),
            "sessions": combined,
        }
        if errors:
            payload["adapter_errors"] = errors
        return payload

    def read_session(
        self,
        app: str,
        session_id: str,
        *,
        tail_messages: int = 10,
        max_chars_per_message: int = 6000,
        include_tools: bool = False,
    ) -> dict[str, Any]:
        norm_app, sid, err = self._resolve_target(app, session_id)
        if err or norm_app is None or sid is None:
            return {"error": err or "Invalid session target"}

        tail_messages = max(1, min(int(tail_messages), 50))
        max_chars_per_message = max(200, min(int(max_chars_per_message), 24000))

        result = self.adapters[norm_app].read_session(
            sid,
            tail_messages=tail_messages,
            max_chars_per_message=max_chars_per_message,
            include_tools=include_tools,
        )
        if "session" in result:
            aliases = self.bindings.aliases_by_session().get((norm_app, result["session"]["session_id"]))
            if aliases:
                result["session"]["bound_aliases"] = aliases
        return result

    def bind_task(self, alias: str, app: str, session_id: str, note: str = "") -> dict[str, Any]:
        norm_app = normalize_app_name(app)
        if norm_app not in self.adapters:
            return {"error": f"Unknown app '{app}'. Expected one of: {', '.join(VALID_APPS)}."}
        sid = (session_id or "").strip()
        if not sid:
            return {"error": "session_id must not be empty."}
        meta = self.adapters[norm_app].get_session_meta(sid)
        if meta is None:
            return {"error": f"Session '{sid}' not found in '{norm_app}'. Search or list first to verify the exact ID."}
        entry = self.bindings.bind(
            alias,
            norm_app,
            meta.session_id,
            title=meta.title,
            workspace=meta.workspace,
            note=note,
        )
        return {
            "ok": True,
            "binding": entry,
            "live_status": meta.status,
            "live_status_detail": meta.status_detail,
        }

    def list_bindings(self) -> dict[str, Any]:
        raw = self.bindings.list_all()
        enriched: list[dict[str, Any]] = []
        for _, entry in raw.items():
            b_app = str(entry.get("app", ""))
            b_sid = str(entry.get("session_id", ""))
            item = dict(entry)
            adapter = self.adapters.get(b_app)
            if adapter is not None:
                try:
                    meta = adapter.get_session_meta(b_sid)
                    if meta is not None:
                        item["live_title"] = meta.title
                        item["live_status"] = meta.status
                        item["live_status_detail"] = meta.status_detail
                        item["live_updated_at"] = meta.to_dict()["updated_at"]
                    else:
                        item["live_status"] = "missing"
                except Exception as exc:  # noqa: BLE001
                    item["live_status"] = "error"
                    item["live_status_detail"] = str(exc)
            enriched.append(item)
        return {"count": len(enriched), "bindings": enriched}

    def unbind_task(self, alias: str) -> dict[str, Any]:
        removed = self.bindings.unbind(alias)
        return {"ok": removed, "alias": alias}

    def writer_status(self) -> dict[str, Any]:
        """Inspect readiness of all 4 desktop write bridges."""
        statuses: dict[str, Any] = {}
        for name, writer in self.writers.items():
            try:
                statuses[name] = writer.status()
            except Exception as exc:  # noqa: BLE001
                statuses[name] = {"app": name, "ready": False, "detail": str(exc)}
        return {"writers": statuses}

    def _invoke_writer_send(
        self,
        writer: Any,
        session_id: str,
        text: str,
        *,
        target_title: str,
        mode: str,
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        sig = inspect.signature(writer.send_message)
        kwargs: dict[str, Any] = {"mode": mode}
        if "target_title" in sig.parameters:
            kwargs["target_title"] = target_title
        if model.strip() and "model" in sig.parameters:
            kwargs["model"] = model.strip()
        if provider.strip() and "provider" in sig.parameters:
            kwargs["provider"] = provider.strip()
        if reasoning_effort.strip() and "reasoning_effort" in sig.parameters:
            kwargs["reasoning_effort"] = reasoning_effort.strip()
        return writer.send_message(session_id, text, **kwargs)

    def _invoke_writer_create(
        self,
        writer: Any,
        *,
        cwd: str = "",
        title: str = "",
        initial_message: str = "",
        agent_preset: str = "",
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        sig = inspect.signature(writer.create_session)
        kwargs: dict[str, Any] = {}
        if "cwd" in sig.parameters:
            kwargs["cwd"] = cwd
        if "title" in sig.parameters:
            kwargs["title"] = title
        if "initial_message" in sig.parameters:
            kwargs["initial_message"] = initial_message
        if agent_preset.strip() and "agent_preset" in sig.parameters:
            kwargs["agent_preset"] = agent_preset.strip()
        if model.strip() and "model" in sig.parameters:
            kwargs["model"] = model.strip()
        if provider.strip() and "provider" in sig.parameters:
            kwargs["provider"] = provider.strip()
        if reasoning_effort.strip() and "reasoning_effort" in sig.parameters:
            kwargs["reasoning_effort"] = reasoning_effort.strip()
        return writer.create_session(**kwargs)

    def send_message(
        self,
        app: str,
        session_id: str,
        message: str,
        *,
        sender_tag: str = "[via Hermes]",
        mode: str = "queue",
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
        dry_run: bool = False,
        allow_duplicate: bool = False,
        source_meta: dict[str, Any] | None = None,
        watch: bool = False,
        watch_alias: str = "",
        notify_targets: str = "qqbot",
        stall_minutes: int = 15,
        notify_on: str = "all",
        verify_agent: bool = False,
        suppress_routine: bool = False,
    ) -> dict[str, Any]:
        """Send (or dry-run preview) a message into a real desktop session."""
        norm_app, sid, err = self._resolve_target(app, session_id)
        if err or norm_app is None or sid is None:
            return {"error": err or "Invalid target session"}

        body = str(message or "")
        if not body.strip():
            return {"error": "message must not be empty."}

        writer = self.writers.get(norm_app)
        if writer is None:
            return {"error": f"No write bridge configured for '{norm_app}'."}

        meta = self.adapters[norm_app].get_session_meta(sid)
        if meta is None:
            if norm_app == "cline" and sid in ("active", "vscode-cline-active"):
                from .models import STATUS_IDLE, SessionSummary

                meta = SessionSummary(
                    app="cline",
                    session_id=sid,
                    title="VS Code Cline (active task)",
                    status=STATUS_IDLE,
                )
            else:
                return {"error": f"Target session '{sid}' not found in '{norm_app}'."}

        tag = (sender_tag or "").strip()
        if tag and not body.lstrip().startswith(tag):
            final_text = f"{tag}\n\n{body}"
        else:
            final_text = body

        model_clean = (model or "").strip()
        provider_clean = (provider or "").strip()
        effort_clean = (reasoning_effort or "").strip()
        model_meta: dict[str, str] = {}
        if model_clean:
            model_meta["model"] = model_clean
        if provider_clean:
            model_meta["provider"] = provider_clean
        if effort_clean:
            model_meta["reasoning_effort"] = effort_clean

        text_hash = self.audit.content_hash(final_text)
        if not allow_duplicate:
            dup = self.audit.check_duplicate(norm_app, meta.session_id, text_hash, window_seconds=60)
            if dup is not None:
                return {
                    "error": (
                        f"Duplicate message suppressed within 60s (previous audit_id={dup.get('audit_id')}, "
                        f"sent at {dup.get('timestamp')}). Pass allow_duplicate=True to force resend."
                    ),
                    "duplicate_audit": dup,
                }

        writer_st = writer.status()
        if dry_run:
            preview = final_text if len(final_text) <= 2400 else (final_text[:1800] + "\n...\n" + final_text[-400:])
            dry_out: dict[str, Any] = {
                "ok": True,
                "dry_run": True,
                "target_app": norm_app,
                "target_session_id": meta.session_id,
                "target_title": meta.title,
                "target_status": meta.status,
                "mode": mode,
                **model_meta,
                "char_count": len(final_text),
                "text_hash": text_hash,
                "message_preview": preview,
                "writer_ready": bool(writer_st.get("ready")),
                "writer_status": writer_st,
                "watch_planned": bool(watch),
            }
            if watch:
                dry_out["notify_targets"] = parse_notify_targets(notify_targets)
            return dry_out

        if not writer_st.get("ready"):
            return {
                "error": f"Write bridge for '{norm_app}' is not ready: {writer_st.get('detail', 'unavailable')}",
                "writer_status": writer_st,
            }

        try:
            res = self._invoke_writer_send(
                writer,
                meta.session_id,
                final_text,
                target_title=meta.title,
                mode=mode,
                model=model_clean,
                provider=provider_clean,
                reasoning_effort=effort_clean,
            )
        except SessionMismatchError as exc:
            self.audit.record(
                {
                    "action": "relay" if source_meta else "send",
                    "status": "error",
                    "error_code": "session_mismatch",
                    "dry_run": False,
                    "target_app": norm_app,
                    "target_session_id": meta.session_id,
                    "target_title": meta.title,
                    "expected_session_id": exc.expected_session_id,
                    "actual_session_id": exc.actual_session_id,
                    "mode": mode,
                    **model_meta,
                    "char_count": len(final_text),
                    "text_hash": text_hash,
                    "error": str(exc),
                    **(source_meta or {}),
                }
            )
            return {
                "error": str(exc),
                "error_code": "session_mismatch",
                "target_app": norm_app,
                "target_session_id": meta.session_id,
                "expected_session_id": exc.expected_session_id,
                "actual_session_id": exc.actual_session_id,
            }
        except Exception as exc:  # noqa: BLE001
            self.audit.record(
                {
                    "action": "relay" if source_meta else "send",
                    "status": "error",
                    "dry_run": False,
                    "target_app": norm_app,
                    "target_session_id": meta.session_id,
                    "target_title": meta.title,
                    "mode": mode,
                    **model_meta,
                    "char_count": len(final_text),
                    "text_hash": text_hash,
                    "error": str(exc),
                    **(source_meta or {}),
                }
            )
            return {"error": str(exc), "target_app": norm_app, "target_session_id": meta.session_id}

        verified_sid = str(res.get("verified_session_id") or res.get("sessionId") or meta.session_id).strip()
        verified_title = str(res.get("verified_title") or meta.title).strip()

        audit_entry = self.audit.record(
            {
                "action": "relay" if source_meta else "send",
                "status": "sent",
                "dry_run": False,
                "target_app": norm_app,
                "target_session_id": verified_sid,
                "target_title": verified_title,
                "verified_session_id": verified_sid,
                "verified_title": verified_title,
                "mode": res.get("mode", mode),
                **model_meta,
                "rpc_id": res.get("requestId", ""),
                "char_count": len(final_text),
                "text_hash": text_hash,
                **(source_meta or {}),
            }
        )
        out: dict[str, Any] = {
            "ok": True,
            "audit_id": audit_entry["audit_id"],
            "app": norm_app,
            "session_id": verified_sid,
            "verified_session_id": verified_sid,
            "verified_title": verified_title,
            "target_title": verified_title,
            "mode": res.get("mode", mode),
            **model_meta,
            "rpc_id": res.get("requestId", ""),
            "char_count": len(final_text),
            "sent_at": audit_entry["timestamp"],
            "bridge_result": res,
        }
        if watch:
            out["watch"] = self.watch_add(
                norm_app,
                verified_sid,
                alias=watch_alias,
                notify_targets=notify_targets,
                stall_minutes=stall_minutes,
                notify_on=notify_on,
                verify_agent=verify_agent,
                suppress_routine=suppress_routine,
            )
        return out

    def relay_message(
        self,
        source_app: str,
        source_session_id: str,
        target_app: str,
        target_session_id: str,
        *,
        source_message_index: int | None = None,
        source_role: str = "assistant",
        extract_code_block: bool = False,
        override_text: str = "",
        prefix_instruction: str = "",
        dry_run: bool = True,
        allow_duplicate: bool = False,
        watch: bool = False,
        watch_alias: str = "",
        notify_targets: str = "qqbot",
        stall_minutes: int = 15,
        notify_on: str = "all",
        verify_agent: bool = False,
        suppress_routine: bool = False,
    ) -> dict[str, Any]:
        """Relay a verbatim message/prompt from a source session into a target session."""
        src_app, src_sid, src_err = self._resolve_target(source_app, source_session_id)
        if src_err or src_app is None or src_sid is None:
            return {"error": f"Source error: {src_err}"}

        dst_app, dst_sid, dst_err = self._resolve_target(target_app, target_session_id)
        if dst_err or dst_app is None or dst_sid is None:
            return {"error": f"Target error: {dst_err}"}

        if src_app == dst_app and src_sid == dst_sid:
            return {"error": "Refusing self-loop relay where source and target are the same session."}

        src_read = self.adapters[src_app].read_session(
            src_sid,
            tail_messages=20,
            max_chars_per_message=24000,
            include_tools=False,
        )
        if "error" in src_read:
            return {"error": f"Failed reading source session: {src_read['error']}"}

        src_session = src_read["session"]
        src_title = str(src_session.get("title") or src_sid)
        msgs = src_read.get("messages") or []

        extracted_code_block = False
        picked_index: int | None = source_message_index
        picked_role = source_role

        if override_text.strip():
            body = override_text.strip()
        else:
            chosen = None
            if source_message_index is not None:
                for m in msgs:
                    if m.get("index") == source_message_index:
                        chosen = m
                        break
                if chosen is None:
                    avail = [(m.get("index"), m.get("role")) for m in msgs[-8:]]
                    return {"error": f"source_message_index={source_message_index} not found in recent messages. Recent: {avail}"}
            else:
                want_role = (source_role or "assistant").strip().lower()
                for m in reversed(msgs):
                    if str(m.get("role", "")).lower() == want_role:
                        chosen = m
                        break
                if chosen is None:
                    avail = [(m.get("index"), m.get("role")) for m in msgs[-8:]]
                    return {"error": f"No recent '{want_role}' message found in {src_app}:{src_title}. Recent: {avail}"}

            body = str(chosen.get("content") or "").strip()
            picked_index = chosen.get("index")
            picked_role = str(chosen.get("role") or source_role)
            if extract_code_block:
                body, extracted_code_block = _extract_last_code_block(body)

        if not body.strip():
            return {"error": "Selected source message body is empty."}

        header = f"[via Hermes | relayed from {src_app}:{src_title}]"
        if prefix_instruction.strip():
            final_msg = f"{header}\n{prefix_instruction.strip()}\n\n{body}"
        else:
            final_msg = f"{header}\n\n{body}"

        source_meta = {
            "source_app": src_app,
            "source_session_id": src_session["session_id"],
            "source_title": src_title,
            "source_message_index": picked_index,
            "source_role": picked_role,
            "extracted_code_block": extracted_code_block,
        }
        res = self.send_message(
            dst_app,
            dst_sid,
            final_msg,
            sender_tag="",
            mode="queue",
            dry_run=dry_run,
            allow_duplicate=allow_duplicate,
            source_meta=source_meta,
            watch=watch,
            watch_alias=watch_alias,
            notify_targets=notify_targets,
            stall_minutes=stall_minutes,
            notify_on=notify_on,
            verify_agent=verify_agent,
            suppress_routine=suppress_routine,
        )
        res.update(source_meta)
        return res

    def create_session(
        self,
        app: str,
        *,
        title: str = "",
        workspace: str = "",
        initial_message: str = "",
        sender_tag: str = "[via Hermes]",
        model: str = "",
        provider: str = "",
        reasoning_effort: str = "",
        agent_preset: str = "",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Create a new real desktop session (dsh, cline, codex, or antigravity) and optionally send the first message."""
        norm_app = normalize_app_name(app)
        if norm_app not in self.adapters:
            return {"error": f"Unknown app '{app}'. Expected one of: {', '.join(VALID_APPS)}."}

        writer = self.writers.get(norm_app)
        if writer is None or not hasattr(writer, "create_session"):
            return {"error": f"Session creation is not supported for '{norm_app}' yet (supported: dsh, cline, codex, antigravity)."}

        create_sig = inspect.signature(writer.create_session)
        takes_initial_directly = "initial_message" in create_sig.parameters
        raw_msg = initial_message.strip() or (title.strip() if norm_app == "cline" else "")
        tag = (sender_tag or "").strip()
        final_text = f"{tag}\n\n{raw_msg}" if raw_msg and tag and not raw_msg.startswith(tag) else raw_msg

        if norm_app == "cline" and not final_text:
            return {"error": f"Creating a '{norm_app}' task requires non-empty initial_message (or title)."}

        model_clean = (model or "").strip()
        provider_clean = (provider or "").strip()
        effort_clean = (reasoning_effort or "").strip()
        preset_clean = (agent_preset or "").strip()
        model_meta: dict[str, str] = {}
        if model_clean:
            model_meta["model"] = model_clean
        if provider_clean:
            model_meta["provider"] = provider_clean
        if effort_clean:
            model_meta["reasoning_effort"] = effort_clean
        if preset_clean:
            model_meta["agent_preset"] = preset_clean

        writer_st = writer.status()
        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "app": norm_app,
                "title": title,
                "workspace": workspace,
                **model_meta,
                "has_initial_message": bool(final_text),
                "writer_ready": bool(writer_st.get("ready")),
                "writer_status": writer_st,
            }

        if not writer_st.get("ready"):
            return {
                "error": f"Write bridge for '{norm_app}' is not ready: {writer_st.get('detail', 'unavailable')}",
                "writer_status": writer_st,
            }

        created = self._invoke_writer_create(
            writer,
            cwd=workspace,
            title=title,
            initial_message=final_text,
            agent_preset=preset_clean,
            model=model_clean,
            provider=provider_clean,
            reasoning_effort=effort_clean,
        )
        session_id = str(created.get("sessionId") or "")
        if not session_id:
            return {"error": f"Failed to create session in '{norm_app}': {created}"}

        self.audit.record(
            {
                "action": "create_session",
                "status": "created",
                "dry_run": False,
                "target_app": norm_app,
                "target_session_id": session_id,
                "target_title": title or session_id,
                "workspace": created.get("cwd", workspace),
                **model_meta,
            }
        )

        out: dict[str, Any] = {
            "ok": True,
            "app": norm_app,
            "session_id": session_id,
            "title": title,
            "workspace": created.get("cwd", workspace),
            **model_meta,
            "created": created,
        }

        if takes_initial_directly and final_text:
            audit_entry = self.audit.record(
                {
                    "action": "send",
                    "status": "sent",
                    "dry_run": False,
                    "target_app": norm_app,
                    "target_session_id": session_id,
                    "target_title": title or session_id,
                    "mode": "new",
                    **model_meta,
                    "rpc_id": "",
                    "char_count": len(final_text),
                    "text_hash": self.audit.content_hash(final_text),
                }
            )
            out["initial_send"] = {
                "ok": True,
                "audit_id": audit_entry["audit_id"],
                "char_count": len(final_text),
            }
        elif not takes_initial_directly and final_text:
            # If writer.create_session already accepted model/provider/reasoning_effort, avoid re-applying twice.
            pass_model = "" if "model" in create_sig.parameters else model_clean
            pass_provider = "" if "provider" in create_sig.parameters else provider_clean
            pass_effort = "" if "reasoning_effort" in create_sig.parameters else effort_clean
            send_res = self._invoke_writer_send(
                writer,
                session_id,
                final_text,
                target_title=title or session_id,
                mode="queue",
                model=pass_model,
                provider=pass_provider,
                reasoning_effort=pass_effort,
            )
            audit_entry = self.audit.record(
                {
                    "action": "send",
                    "status": "sent",
                    "dry_run": False,
                    "target_app": norm_app,
                    "target_session_id": session_id,
                    "target_title": title or session_id,
                    "mode": send_res.get("mode", "queue"),
                    **model_meta,
                    "rpc_id": send_res.get("requestId", ""),
                    "char_count": len(final_text),
                    "text_hash": self.audit.content_hash(final_text),
                }
            )
            out["initial_send"] = {
                "ok": True,
                "audit_id": audit_entry["audit_id"],
                "rpc_id": send_res.get("requestId", ""),
                "char_count": len(final_text),
            }

        return out

    def watch_add(
        self,
        app: str,
        session_id: str,
        *,
        alias: str = "",
        notify_targets: str = "qqbot",
        stall_minutes: int = 15,
        notify_on: str = "all",
        verify_agent: bool = False,
        suppress_routine: bool = False,
    ) -> dict[str, Any]:
        """Register or update a zero-token watch on a verified existing desktop session."""
        req_app = normalize_app_name(app)
        req_sid = (session_id or "").strip()
        if not req_sid:
            return {"error": "session_id must not be empty."}
        if req_sid in ("active", "vscode-cline-active"):
            return {
                "error": (
                    f"Refusing to register watch for placeholder session_id '{req_sid}'; "
                    "pass a verified concrete session_id."
                )
            }

        meta = None
        if req_app in self.adapters and not req_sid.lower().startswith("alias:"):
            bound_entry = self.bindings.resolve(req_sid)
            if bound_entry is None or str(bound_entry.get("app")) != req_app:
                meta = self.adapters[req_app].get_session_meta(req_sid)
                if meta is None:
                    return {
                        "error": (
                            f"Session '{req_sid}' not found in '{req_app}'. "
                            "Refusing to register watch for non-existent session."
                        )
                    }
                norm_app = req_app
                sid = meta.session_id
            else:
                norm_app, sid, err = self._resolve_target(app, session_id)
                if err or norm_app is None or sid is None:
                    return {"error": err or "Invalid session target"}
        else:
            norm_app, sid, err = self._resolve_target(app, session_id)
            if err or norm_app is None or sid is None:
                return {"error": err or "Invalid session target"}

        if sid in ("active", "vscode-cline-active"):
            return {
                "error": (
                    f"Refusing to register watch for placeholder session_id '{sid}'; "
                    "pass a verified concrete session_id."
                )
            }

        if meta is None:
            meta = self.adapters[norm_app].get_session_meta(sid)
        if meta is None:
            return {"error": f"Session '{sid}' not found in '{norm_app}'. Refusing to register watch for non-existent session."}

        clean_alias = (alias or "").strip()
        if not clean_alias:
            bound = self.bindings.aliases_by_session().get((norm_app, meta.session_id))
            if bound:
                clean_alias = bound[0]

        watch_payload: dict[str, Any] = {
            "alias": clean_alias,
            "app": norm_app,
            "session_id": meta.session_id,
            "title": meta.title,
            "notify_targets": parse_notify_targets(notify_targets),
            "stall_minutes": max(1, int(stall_minutes)),
            "cooldown_minutes": max(1, int(stall_minutes)),
            "notify_on": parse_notify_events(notify_on),
            "enabled": True,
            "last_status": meta.status,
            "last_status_detail": meta.status_detail,
            "last_updated_at_ms": meta.updated_at_ms,
            "last_notified_event": "",
            "last_notified_at": 0,
            "last_notified_by_event": {},
            "last_suppressed_by_event": {},
            "pending_transition": None,
        }
        if verify_agent:
            watch_payload["verify_agent"] = True
        if suppress_routine:
            watch_payload["suppress_routine"] = True

        entry = self.watches.upsert(watch_payload)
        return {"ok": True, "watch": entry}

    def watch_list(self) -> dict[str, Any]:
        """List all registered watches enriched with live session status."""
        watches = self.watches.list_all()
        out: list[dict[str, Any]] = []
        for w in watches:
            item = dict(w)
            adapter = self.adapters.get(str(w.get("app", "")))
            sid = str(w.get("session_id", ""))
            if adapter is not None and sid:
                try:
                    meta = adapter.get_session_meta(sid)
                    if meta is not None:
                        item["live_title"] = meta.title
                        item["live_status"] = meta.status
                        item["live_status_detail"] = meta.status_detail
                        item["live_updated_at"] = format_local_iso(meta.updated_at_ms)
                    else:
                        item["live_status"] = "missing"
                except Exception as exc:  # noqa: BLE001
                    item["live_status"] = "error"
                    item["live_status_detail"] = str(exc)
            out.append(item)
        return {"count": len(out), "watches": out}

    def watch_remove(self, watch_id_or_alias: str) -> dict[str, Any]:
        """Remove a registered watch by watch_id, alias, or session_id."""
        removed = self.watches.remove(watch_id_or_alias)
        if removed is None:
            return {"ok": False, "error": f"Watch '{watch_id_or_alias}' not found"}
        return {"ok": True, "removed": removed}

    def watch_check_now(
        self,
        *,
        dry_run: bool = True,
        verify_agent: bool = False,
        suppress_routine: bool = False,
    ) -> dict[str, Any]:
        """Run an immediate poll pass across all watches."""
        runner = WatchdogRunner(
            adapters=self.adapters,
            store=self.watches,
            verify_agent=verify_agent,
            suppress_routine=suppress_routine,
        )
        return runner.poll_once(dry_run=dry_run)


_SERVICE = HubService()

mcp = MCPServer(
    name="agent-hub",
    instructions=(
        "Unified session hub for Codex, DeepSeek Harness (dsh), Antigravity, and VS Code Cline. "
        "Supports read-only session inspection with content-body search and fuzzy target locking, "
        "task alias bindings, sending messages and creating new sessions across all 4 desktop apps "
        "with optional model/provider/reasoning_effort passthrough, verbatim cross-agent prompt relay "
        "with dry_run previews, and zero-token watchdog alerts."
    ),
)


@mcp.tool()
def hub_overview(limit_per_app: int = 5) -> str:
    """Get a cross-app live status overview across Codex, DSH, Antigravity, and Cline, plus saved task bindings.

    Args:
        limit_per_app: Number of most-recent sessions to include per app (1-20, default 5).
    """
    return _dumps(_SERVICE.overview(limit_per_app=limit_per_app))


@mcp.tool()
def hub_list_sessions(
    app: str = "all",
    active_only: bool = False,
    include_archived: bool = False,
    include_subagents: bool = False,
    limit: int = 20,
) -> str:
    """List recent sessions with title, unified status, workspace, update time, and short preview.

    Args:
        app: 'all', 'codex', 'dsh', 'antigravity', or 'cline'.
        active_only: If true, keep only running, waiting_approval, waiting_user, or error sessions.
        include_archived: Include archived Codex threads if true.
        include_subagents: Include spawned subagent sessions if true.
        limit: Max number of sessions returned (1-100, default 20).
    """
    return _dumps(
        _SERVICE.list_sessions(
            app=app,
            active_only=active_only,
            include_archived=include_archived,
            include_subagents=include_subagents,
            limit=limit,
        )
    )


@mcp.tool()
def hub_search_sessions(query: str, app: str = "all", limit: int = 15) -> str:
    """Search sessions across apps by binding alias, title, workspace, preview, or conversation content.

    Args:
        query: Keyword or phrase to match.
        app: 'all', 'codex', 'dsh', 'antigravity', or 'cline'.
        limit: Max matches returned (1-50, default 15).
    """
    return _dumps(_SERVICE.search_sessions(query=query, app=app, limit=limit))


@mcp.tool()
def hub_read_session(
    app: str,
    session_id: str,
    tail_messages: int = 10,
    max_chars_per_message: int = 6000,
    include_tools: bool = False,
) -> str:
    """Read verbatim recent messages from a target session, bound task alias, or fuzzy content match.

    Args:
        app: 'codex', 'dsh', 'antigravity', 'cline', 'alias', or 'all'.
        session_id: Exact session ID, 'alias:<name>' / bound alias, or a natural-language phrase/snippet to auto-locate.
        tail_messages: Number of most-recent user/assistant messages to return (1-50, default 10).
        max_chars_per_message: Max characters per returned message (default 6000).
        include_tools: Include compact tool call / command summaries if true.
    """
    return _dumps(
        _SERVICE.read_session(
            app=app,
            session_id=session_id,
            tail_messages=tail_messages,
            max_chars_per_message=max_chars_per_message,
            include_tools=include_tools,
        )
    )


@mcp.tool()
def hub_bind_task(alias: str, app: str, session_id: str, note: str = "") -> str:
    """Bind a friendly task alias (e.g. '主pipeline', 'Class B', 'Project C') to an exact (app, session_id).

    Args:
        alias: Human-friendly task name.
        app: 'codex', 'dsh', 'antigravity', or 'cline'.
        session_id: Exact verified session ID.
        note: Optional short reminder note.
    """
    return _dumps(_SERVICE.bind_task(alias=alias, app=app, session_id=session_id, note=note))


@mcp.tool()
def hub_list_bindings() -> str:
    """List all saved task alias bindings enriched with current live title and status."""
    return _dumps(_SERVICE.list_bindings())


@mcp.tool()
def hub_unbind_task(alias: str) -> str:
    """Remove a previously saved task alias binding.

    Args:
        alias: Friendly task alias to delete.
    """
    return _dumps(_SERVICE.unbind_task(alias=alias))


@mcp.tool()
def hub_writer_status() -> str:
    """Check live write-bridge availability for dsh, codex, antigravity, and cline."""
    return _dumps(_SERVICE.writer_status())


@mcp.tool()
def hub_send_message(
    app: str,
    session_id: str,
    message: str,
    sender_tag: str = "[via Hermes]",
    mode: str = "queue",
    model: str = "",
    provider: str = "",
    reasoning_effort: str = "",
    dry_run: bool = False,
    allow_duplicate: bool = False,
    watch: bool = False,
    watch_alias: str = "",
    notify_targets: str = "qqbot",
    stall_minutes: int = 15,
    notify_on: str = "all",
    verify_agent: bool = False,
    suppress_routine: bool = False,
) -> str:
    """Send a message on the user's behalf into a real desktop session (with UI visibility, session verification, and audit log).

    Args:
        app: 'dsh', 'codex', 'antigravity', 'cline', 'alias', or 'all'.
        session_id: Exact session ID, bound task alias, or fuzzy search phrase.
        message: Exact text to send.
        sender_tag: Prefix attribution line (default '[via Hermes]', pass '' to omit).
        mode: Delivery mode ('queue' default, or 'steer' for active DSH turns).
        model: Optional model identifier (e.g. 'o3-mini', 'deepseek-chat', 'deepseek-reasoner', 'gemini-2.5-pro').
        provider: Optional provider identifier (e.g. 'openai', 'deepseek', 'anthropic', 'google').
        reasoning_effort: Optional reasoning effort / thinking depth ('low', 'medium', 'high', 'xhigh', etc.).
        dry_run: If true, validate target & preview payload without sending.
        allow_duplicate: Bypass the 60-second duplicate-send guard if true.
        watch: If true, automatically register/refresh a watch on the verified_session_id after send succeeds.
        watch_alias: Optional friendly alias for the auto-registered watch.
        notify_targets: Comma-separated watch targets when watch=True (default 'qqbot').
        stall_minutes: Watch stall threshold in minutes when watch=True (default 15).
        notify_on: Watch event filter when watch=True (default 'all').
        verify_agent: Enable Level-2 `hermes -z` verification on the auto-registered watch.
        suppress_routine: Suppress routine alerts when Level-2 marks needs_attention=false.
    """
    return _dumps(
        _SERVICE.send_message(
            app=app,
            session_id=session_id,
            message=message,
            sender_tag=sender_tag,
            mode=mode,
            model=model,
            provider=provider,
            reasoning_effort=reasoning_effort,
            dry_run=dry_run,
            allow_duplicate=allow_duplicate,
            watch=watch,
            watch_alias=watch_alias,
            notify_targets=notify_targets,
            stall_minutes=stall_minutes,
            notify_on=notify_on,
            verify_agent=verify_agent,
            suppress_routine=suppress_routine,
        )
    )


@mcp.tool()
def hub_relay_message(
    source_app: str,
    source_session_id: str,
    target_app: str,
    target_session_id: str,
    source_message_index: int | None = None,
    source_role: str = "assistant",
    extract_code_block: bool = False,
    override_text: str = "",
    prefix_instruction: str = "",
    dry_run: bool = True,
    allow_duplicate: bool = False,
    watch: bool = False,
    watch_alias: str = "",
    notify_targets: str = "qqbot",
    stall_minutes: int = 15,
    notify_on: str = "all",
    verify_agent: bool = False,
    suppress_routine: bool = False,
) -> str:
    """Relay a verbatim message or extracted prompt block from a source session into a target session.

    Args:
        source_app: Source app ('codex', 'dsh', 'antigravity', 'cline', or 'alias').
        source_session_id: Source session ID or bound alias.
        target_app: Target app ('dsh', 'codex', 'antigravity', 'cline', or 'alias').
        target_session_id: Target session ID or bound alias.
        source_message_index: Optional exact message index from hub_read_session; defaults to latest matching source_role.
        source_role: Role to pick when source_message_index is omitted (default 'assistant').
        extract_code_block: If true, extract the main fenced code block (```...```) from the source message.
        override_text: Optional explicit text to relay instead of auto-picking from source transcript.
        prefix_instruction: Optional short instruction prepended before the relayed body.
        dry_run: Defaults to True (safe preview); set False only once confirmed to actually dispatch.
        allow_duplicate: Bypass the 60-second duplicate-send guard if true.
        watch: If true, automatically register/refresh a watch on the verified target session after relay succeeds.
        watch_alias: Optional friendly alias for the auto-registered watch.
        notify_targets: Comma-separated watch targets when watch=True (default 'qqbot').
        stall_minutes: Watch stall threshold in minutes when watch=True (default 15).
        notify_on: Watch event filter when watch=True (default 'all').
        verify_agent: Enable Level-2 `hermes -z` verification on the auto-registered watch.
        suppress_routine: Suppress routine alerts when Level-2 marks needs_attention=false.
    """
    return _dumps(
        _SERVICE.relay_message(
            source_app=source_app,
            source_session_id=source_session_id,
            target_app=target_app,
            target_session_id=target_session_id,
            source_message_index=source_message_index,
            source_role=source_role,
            extract_code_block=extract_code_block,
            override_text=override_text,
            prefix_instruction=prefix_instruction,
            dry_run=dry_run,
            allow_duplicate=allow_duplicate,
            watch=watch,
            watch_alias=watch_alias,
            notify_targets=notify_targets,
            stall_minutes=stall_minutes,
            notify_on=notify_on,
            verify_agent=verify_agent,
            suppress_routine=suppress_routine,
        )
    )


@mcp.tool()
def hub_create_session(
    app: str,
    title: str = "",
    workspace: str = "",
    initial_message: str = "",
    sender_tag: str = "[via Hermes]",
    model: str = "",
    provider: str = "",
    reasoning_effort: str = "",
    agent_preset: str = "",
    dry_run: bool = False,
) -> str:
    """Create a new real desktop conversation ('dsh', 'cline', 'codex', or 'antigravity') and optionally send an initial message.

    Args:
        app: Target app ('dsh', 'cline', 'codex', or 'antigravity').
        title: Optional title for the new session.
        workspace: Optional workspace directory path (defaults to app workspace).
        initial_message: First prompt to send (required for 'cline'; optional for 'dsh', 'codex', 'antigravity').
        sender_tag: Prefix attribution line for initial_message (default '[via Hermes]').
        model: Optional model identifier (e.g. 'o3-mini', 'deepseek-chat', 'deepseek-reasoner', 'gemini-2.5-pro').
        provider: Optional provider identifier (e.g. 'openai', 'deepseek', 'anthropic', 'google').
        reasoning_effort: Optional reasoning effort / thinking depth ('low', 'medium', 'high', 'xhigh', etc.).
        agent_preset: Optional agent preset identifier (supported by 'dsh').
        dry_run: Preview without creating if true.
    """
    return _dumps(
        _SERVICE.create_session(
            app=app,
            title=title,
            workspace=workspace,
            initial_message=initial_message,
            sender_tag=sender_tag,
            model=model,
            provider=provider,
            reasoning_effort=reasoning_effort,
            agent_preset=agent_preset,
            dry_run=dry_run,
        )
    )


@mcp.tool()
def hub_watch_add(
    app: str,
    session_id: str,
    alias: str = "",
    notify_targets: str = "qqbot",
    stall_minutes: int = 15,
    notify_on: str = "all",
    verify_agent: bool = False,
    suppress_routine: bool = False,
) -> str:
    """Register a zero-token background watch on a session to push QQ/Weixin alerts on completion, errors, approvals, or stalls.

    Args:
        app: 'codex', 'dsh', 'antigravity', 'cline', or 'alias'.
        session_id: Exact session ID or bound task alias.
        alias: Optional friendly label for notifications.
        notify_targets: Comma-separated targets: 'qqbot' (default), 'weixin', or 'qqbot,weixin'.
        stall_minutes: Minutes of inactivity while running before emitting a 'stalled' alert (default 15).
        notify_on: 'all' or comma-separated subset of completed,waiting_approval,waiting_user,error,interrupted,stalled.
        verify_agent: If true, invoke on-demand Level-2 `hermes -z` verification on key state transitions.
        suppress_routine: If true and Level-2 verification marks needs_attention=false, record without pushing alerts.
    """
    return _dumps(
        _SERVICE.watch_add(
            app=app,
            session_id=session_id,
            alias=alias,
            notify_targets=notify_targets,
            stall_minutes=stall_minutes,
            notify_on=notify_on,
            verify_agent=verify_agent,
            suppress_routine=suppress_routine,
        )
    )


@mcp.tool()
def hub_watch_list() -> str:
    """List all active zero-token session watches enriched with live session status."""
    return _dumps(_SERVICE.watch_list())


@mcp.tool()
def hub_watch_remove(watch_id_or_alias: str) -> str:
    """Remove a session watch by watch_id, alias, or session_id.

    Args:
        watch_id_or_alias: Watch ID (e.g. w_1234abcd), alias, or session_id.
    """
    return _dumps(_SERVICE.watch_remove(watch_id_or_alias))


@mcp.tool()
def hub_watch_check_now(
    dry_run: bool = True,
    verify_agent: bool = False,
    suppress_routine: bool = False,
) -> str:
    """Run an immediate check across all watched sessions without waiting for the background interval.

    Args:
        dry_run: Defaults to True (evaluate transitions without sending alerts or mutating watch state).
        verify_agent: If true, enable Level-2 `hermes -z` verification during this immediate check.
        suppress_routine: If true, suppress alerts when Level-2 verification marks needs_attention=false.
    """
    return _dumps(
        _SERVICE.watch_check_now(
            dry_run=dry_run,
            verify_agent=verify_agent,
            suppress_routine=suppress_routine,
        )
    )


def main() -> None:
    """Run the Agent Hub MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
