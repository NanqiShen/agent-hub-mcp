"""Zero-token local watchdog store, transition evaluator, and launchd daemon."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
import uuid

from .adapters import AntigravityAdapter, ClineAdapter, CodexAdapter, DshAdapter
from .models import (
    AppAdapter,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    STATUS_WAITING_APPROVAL,
    STATUS_WAITING_USER,
    SessionSummary,
    clean_single_line,
    truncate_text,
)


DEFAULT_WATCHES_PATH = Path.home() / ".hermes" / "agent-hub" / "watches.json"
DEFAULT_HERMES_BIN = "/Users/example/.local/bin/hermes"

ALL_NOTIFY_EVENTS: tuple[str, ...] = (
    "completed",
    "waiting_approval",
    "waiting_user",
    "error",
    "interrupted",
    "stalled",
)

DEFAULT_VERIFY_EVENTS: tuple[str, ...] = ("completed", "stalled", "error")
DEFAULT_DAEMON_DEBOUNCE_SECONDS = 25
DEBOUNCE_EVENTS: tuple[str, ...] = ("error", "completed")
_TRANSIENT_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        "transient_error",
        "self_correcting",
        "tool_error_recovering",
        "transient_tool_error",
        "false_alarm",
        "recovering",
    }
)

BG_LABEL_SUBMITTED = "【已提交后台作业】"
BG_LABEL_RUNNING = "【阶段汇报·后台作业运行中】"

_EVENT_LABELS: dict[str, str] = {
    "completed": "回合完毕 / 等待输入",
    "waiting_approval": "等待审批",
    "waiting_user": "等待你回复",
    "error": "运行出错",
    "interrupted": "任务中断",
    "stalled": "疑似卡滞",
}

_BG_SUBMITTED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("submitted_batch_job", re.compile(r"\bSubmitted\s+batch\s+job\s+\d+", re.IGNORECASE)),
    ("sbatch", re.compile(r"\bsbatch\b", re.IGNORECASE)),
    ("cn_submitted_job", re.compile(r"已提交[^\n。；;]{0,24}(?:Slurm\s*)?(?:作业|任务|批处理|后台|job)", re.IGNORECASE)),
    ("cn_submitted_job_le", re.compile(r"提交了[^\n。；;]{0,24}(?:Slurm\s*)?(?:作业|任务|批处理|后台|job)", re.IGNORECASE)),
    ("en_submitted_job", re.compile(r"\bsubmitted\s+(?:a\s+)?(?:slurm|batch|background)\s+(?:job|task)s?\b", re.IGNORECASE)),
)

_BG_RUNNING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cn_running_background", re.compile(r"正在后台")),
    ("cn_background_active", re.compile(r"后台(?:作业|任务|进程)[^\n。；;]{0,12}(?:运行中|执行中|排队中|进行中)")),
    ("cn_waiting_result", re.compile(r"等待[^\n。；;]{0,24}(?:出结果|跑完|运行完成|执行完成|作业完成|结束)")),
    ("en_running_background", re.compile(r"\brunning\s+in\s+(?:the\s+)?background\b", re.IGNORECASE)),
    ("en_background_active", re.compile(r"\bbackground\s+(?:job|task|process)\s+(?:is\s+)?(?:running|pending|queued)\b", re.IGNORECASE)),
    ("job_state_active", re.compile(r"(?:作业|job)\s*(?:状态|state|status)\s*[:：=]?\s*(?:RUNNING|PENDING|PD\b)", re.IGNORECASE)),
)

_BG_FINISHED_OVERRIDE_RE = re.compile(
    r"(?:作业|任务|job)\s*\d*\s*(?:已|已经)?(?:全部)?(?:运行|执行)?(?:完成|结束|跑完|成功)",
    re.IGNORECASE,
)

_JOB_ID_RE = re.compile(
    r"(?:Submitted\s+batch\s+job|Slurm\s*(?:作业|任务|job)|(?:作业|job)\s*(?:ID|编号|号)?)\s*[:：=#]?\s*(\d{4,12})",
    re.IGNORECASE,
)

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

logger = logging.getLogger("agent_hub.watcher")


def _resolve_watches_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    env_path = os.environ.get("AGENT_HUB_WATCHES_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_WATCHES_PATH


def parse_notify_targets(raw: str | list[str] | None) -> list[str]:
    """Normalize target string/list (defaulting to ['qqbot'])."""
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
        return items or ["qqbot"]
    txt = str(raw or "").strip()
    if not txt:
        return ["qqbot"]
    items = [part.strip() for part in txt.split(",") if part.strip()]
    return items or ["qqbot"]


def parse_notify_events(raw: str | list[str] | None) -> list[str]:
    """Normalize notify_on event list ('all' expands to ALL_NOTIFY_EVENTS)."""
    if isinstance(raw, list):
        vals = [str(x).strip().lower() for x in raw if str(x).strip()]
    else:
        txt = str(raw or "all").strip().lower()
        if txt in ("", "all", "*"):
            return list(ALL_NOTIFY_EVENTS)
        vals = [p.strip().lower() for p in txt.split(",") if p.strip()]
    valid = [v for v in vals if v in ALL_NOTIFY_EVENTS]
    return valid or list(ALL_NOTIFY_EVENTS)


def parse_verify_events(raw: str | list[str] | None) -> list[str]:
    """Normalize verify_events list (defaults to DEFAULT_VERIFY_EVENTS)."""
    if raw is None:
        return list(DEFAULT_VERIFY_EVENTS)
    if isinstance(raw, list):
        vals = [str(x).strip().lower() for x in raw if str(x).strip()]
        if not vals:
            return list(DEFAULT_VERIFY_EVENTS)
    else:
        txt = str(raw).strip().lower()
        if txt in ("", "default"):
            return list(DEFAULT_VERIFY_EVENTS)
        if txt in ("all", "*"):
            return list(ALL_NOTIFY_EVENTS)
        if txt in ("none", "off"):
            return []
        vals = [p.strip().lower() for p in txt.split(",") if p.strip()]
    return [v for v in vals if v in ALL_NOTIFY_EVENTS]


def detect_background_job(text: str) -> dict[str, Any]:
    """Detect async/Slurm background job submissions or active background work in text."""
    body = str(text or "")
    if not body.strip():
        return {"matched": False, "kind": "", "label": "", "job_ids": [], "signals": []}

    running_signals = [name for name, pat in _BG_RUNNING_PATTERNS if pat.search(body)]
    submitted_signals = [name for name, pat in _BG_SUBMITTED_PATTERNS if pat.search(body)]

    if not running_signals and not submitted_signals:
        return {"matched": False, "kind": "", "label": "", "job_ids": [], "signals": []}

    if submitted_signals and not running_signals and _BG_FINISHED_OVERRIDE_RE.search(body):
        return {"matched": False, "kind": "", "label": "", "job_ids": [], "signals": []}

    job_ids: list[str] = []
    seen_ids: set[str] = set()
    for match in _JOB_ID_RE.finditer(body):
        jid = match.group(1)
        if jid not in seen_ids:
            seen_ids.add(jid)
            job_ids.append(jid)

    if running_signals:
        kind = "running_background"
        label = BG_LABEL_RUNNING
    else:
        kind = "submitted"
        label = BG_LABEL_SUBMITTED

    signals = running_signals + [s for s in submitted_signals if s not in running_signals]
    return {
        "matched": True,
        "kind": kind,
        "label": label,
        "job_ids": job_ids,
        "signals": signals,
    }


class WatchStore:
    """Atomic JSON store for persistent zero-token session watches."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = _resolve_watches_path(path)

    def list_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(data, dict) and isinstance(data.get("watches"), list):
            return [dict(w) for w in data["watches"] if isinstance(w, dict)]
        return []

    def save_all(self, watches: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        payload = {
            "version": 1,
            "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "watches": watches,
        }
        fd, tmp_name = tempfile.mkstemp(prefix=".watches-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def upsert(self, watch: dict[str, Any]) -> dict[str, Any]:
        """Insert or update watch keyed uniquely by (app, session_id)."""
        watches = self.list_all()
        app = str(watch["app"])
        sid = str(watch["session_id"])
        for idx, existing in enumerate(watches):
            if existing.get("app") == app and existing.get("session_id") == sid:
                merged = dict(existing)
                merged.update(watch)
                if "pending_transition" in watch and watch["pending_transition"] is None:
                    merged.pop("pending_transition", None)
                merged["watch_id"] = existing.get("watch_id") or merged.get("watch_id") or f"w_{uuid.uuid4().hex[:8]}"
                watches[idx] = merged
                self.save_all(watches)
                return merged

        new_watch = {k: v for k, v in watch.items() if not (k == "pending_transition" and v is None)}
        new_watch.setdefault("watch_id", f"w_{uuid.uuid4().hex[:8]}")
        new_watch.setdefault(
            "created_at",
            datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        )
        watches.append(new_watch)
        self.save_all(watches)
        return new_watch

    def remove(self, key: str) -> dict[str, Any] | None:
        """Remove a watch by watch_id, alias (case-insensitive), or session_id."""
        target = (key or "").strip()
        if not target:
            return None
        t_low = target.lower()
        watches = self.list_all()
        for idx, w in enumerate(watches):
            if (
                str(w.get("watch_id", "")).lower() == t_low
                or str(w.get("alias", "")).strip().lower() == t_low
                or str(w.get("session_id", "")) == target
            ):
                removed = watches.pop(idx)
                self.save_all(watches)
                return removed
        return None


def evaluate_watch(
    watch: dict[str, Any],
    live_meta: SessionSummary,
    *,
    now_ts: float | None = None,
) -> tuple[str | None, str]:
    """Detect whether a watched session crossed a notifiable transition."""
    now = time.time() if now_ts is None else float(now_ts)
    prev_status = str(watch.get("last_status") or STATUS_IDLE)
    cur_status = live_meta.status
    notify_on = set(parse_notify_events(watch.get("notify_on")))
    if watch.get("cooldown_minutes") is not None:
        cooldown_min = max(0, int(watch["cooldown_minutes"]))
    else:
        cooldown_min = max(1, int(watch.get("stall_minutes") or 15))
    cooldown_s = cooldown_min * 60

    event_type: str | None = None
    reason = ""

    if prev_status == STATUS_RUNNING and cur_status == STATUS_IDLE:
        event_type = "completed"
        reason = "running -> idle"
    elif cur_status == STATUS_WAITING_APPROVAL and prev_status != STATUS_WAITING_APPROVAL:
        event_type = "waiting_approval"
        reason = live_meta.status_detail or "awaiting approval"
    elif cur_status == STATUS_WAITING_USER and prev_status != STATUS_WAITING_USER:
        event_type = "waiting_user"
        reason = live_meta.status_detail or "awaiting user reply"
    elif cur_status == STATUS_ERROR and prev_status != STATUS_ERROR:
        event_type = "error"
        reason = live_meta.status_detail or "session entered error state"
    elif cur_status == STATUS_INTERRUPTED and prev_status != STATUS_INTERRUPTED:
        event_type = "interrupted"
        reason = live_meta.status_detail or "session interrupted"
    elif cur_status == STATUS_RUNNING and live_meta.updated_at_ms > 0:
        stall_min = max(1, int(watch.get("stall_minutes") or 15))
        idle_ms = int(now * 1000) - int(live_meta.updated_at_ms)
        if idle_ms >= stall_min * 60_000:
            last_ev = str(watch.get("last_notified_event") or "")
            last_stalled_ts = int(watch.get("last_stalled_updated_at_ms") or 0)
            last_notified_at = float(watch.get("last_notified_at") or 0)
            already_alerted = (
                last_ev == "stalled"
                and last_stalled_ts == int(live_meta.updated_at_ms)
                and (now - last_notified_at) < cooldown_s
            )
            if not already_alerted:
                event_type = "stalled"
                mins = int(idle_ms // 60_000)
                reason = f"no activity for {mins}m (threshold {stall_min}m)"

    if event_type is None or event_type not in notify_on:
        return None, ""

    if cooldown_s > 0:
        last_notified_at = float(watch.get("last_notified_at") or 0)
        if last_notified_at > 0 and (now - last_notified_at) < cooldown_s:
            return None, ""

        sent_by_event = watch.get("last_notified_by_event")
        if isinstance(sent_by_event, dict):
            last_same_at = float(sent_by_event.get(event_type) or 0)
            if last_same_at > 0 and (now - last_same_at) < cooldown_s:
                return None, ""

        sup_by_event = watch.get("last_suppressed_by_event")
        if isinstance(sup_by_event, dict):
            last_sup_at = float(sup_by_event.get(event_type) or 0)
            if last_sup_at > 0 and (now - last_sup_at) < cooldown_s:
                return None, ""

    return event_type, reason


def format_watch_notification(
    watch: dict[str, Any],
    event_type: str,
    live_meta: SessionSummary,
    *,
    reason: str = "",
    snippet: str = "",
    label_override: str = "",
    verdict: dict[str, Any] | None = None,
) -> str:
    """Format a compact Chinese notification ready for QQ/Weixin push."""
    label = (label_override or "").strip()
    if not label and isinstance(verdict, dict) and verdict.get("ok"):
        label = str(verdict.get("label") or "").strip()
    if not label and event_type in ("completed", "stalled") and snippet:
        bg = detect_background_job(snippet)
        if bg["matched"]:
            label = str(bg["label"])
    if not label:
        label = _EVENT_LABELS.get(event_type, event_type)

    prev_status = str(watch.get("last_status") or STATUS_IDLE)
    alias = str(watch.get("alias") or "").strip()
    display_name = f"{alias} ({live_meta.title})" if alias and alias != live_meta.title else live_meta.title
    detail = live_meta.status_detail or reason
    detail_suffix = f" ({detail})" if detail else ""
    lines = [
        f"[Agent Hub 盯盘] {label} | {live_meta.app}「{display_name}」",
        f"状态: {prev_status} -> {live_meta.status}{detail_suffix}",
    ]
    verdict_summary = (
        str(verdict.get("summary") or "").strip()
        if isinstance(verdict, dict) and verdict.get("ok")
        else ""
    )
    if verdict_summary:
        lines.append(f"研判结论: {verdict_summary}")
    elif snippet:
        lines.append(f"最新进展: {snippet}")
    return "\n".join(lines)


def build_verification_prompt(ctx: dict[str, Any]) -> str:
    """Build a compact fact-check prompt for on-demand Level-2 session verification."""
    watch = ctx.get("watch") or {}
    live_meta = ctx.get("live_meta")
    if isinstance(live_meta, dict):
        app = str(live_meta.get("app") or watch.get("app") or "")
        sid = str(live_meta.get("session_id") or watch.get("session_id") or "")
        title = str(live_meta.get("title") or watch.get("title") or "")
        workspace = str(live_meta.get("workspace") or "")
        cur_status = str(live_meta.get("status") or "")
        detail = str(live_meta.get("status_detail") or "")
    else:
        app = str(getattr(live_meta, "app", "") or watch.get("app") or "")
        sid = str(getattr(live_meta, "session_id", "") or watch.get("session_id") or "")
        title = str(getattr(live_meta, "title", "") or watch.get("title") or "")
        workspace = str(getattr(live_meta, "workspace", "") or "")
        cur_status = str(getattr(live_meta, "status", "") or "")
        detail = str(getattr(live_meta, "status_detail", "") or "")

    alias = str(watch.get("alias") or "")
    prev_status = str(watch.get("last_status") or STATUS_IDLE)
    event_type = str(ctx.get("event_type") or "")
    reason = str(ctx.get("reason") or "")
    snippet = str(ctx.get("snippet") or "")
    bg = ctx.get("background_job") or {}
    context_messages = ctx.get("context_messages") or []

    lines = [
        "你是 Agent Hub 盯盘系统的 Level-2 事实核验员。请仅做只读事实研判，严禁修改任何文件，严禁向会话或聊天软件发送消息。",
        f"目标会话: app={app}, session_id={sid}, alias={alias or '-'}, title={title}",
        f"工作区: {workspace or '-'}",
        f"初筛事件: {event_type} ({prev_status} -> {cur_status}; detail={detail or '-'}; reason={reason or '-'})",
    ]
    if bg.get("matched"):
        lines.append(
            f"规则初筛命中后台作业信号: kind={bg.get('kind')}, job_ids={bg.get('job_ids') or []}, signals={bg.get('signals') or []}"
        )
    if snippet:
        lines.append(f"最新摘要: {snippet}")

    if context_messages:
        lines.append("最近消息片段（由旧到新）：")
        for msg in context_messages[-3:]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "unknown")
            body, _ = truncate_text(str(msg.get("content") or "").strip(), 900)
            if body:
                lines.append(f"[{role}] {body}")

    lines.extend(
        [
            "",
            "注意：如果只是单个命令或工具返回非零退出码/报错（如 HTTP 400、脚本报错、除零等），而模型正在自主排错、重试或继续推进而非整个会话崩溃，请务必将 classification 设为 \"transient_error\" 且 needs_attention 设为 false。",
            "请判断该会话当前真实所处阶段，并严格只输出一个 JSON 对象（不要输出多余解释）：",
            '{"classification":"completed|background_running|still_running|transient_error|needs_user|error","needs_attention":true/false,"label":"简短中文标签，如【阶段汇报·后台作业运行中】或【回合完毕 / 等待输入】","summary":"1-2句高置信度中文结论，说明真实进度、作业ID或是否需要人工介入"}',
        ]
    )
    return "\n".join(lines)


def parse_verification_output(raw_text: str) -> dict[str, Any]:
    """Parse Level-2 verifier output (raw JSON, fenced JSON, or plain-text fallback)."""
    text = str(raw_text or "").strip()
    if not text:
        return {"ok": False, "error": "empty verifier output"}

    parsed: dict[str, Any] | None = None
    candidates: list[str] = [text]
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parsed = obj
            break

    if parsed is not None:
        classification = str(parsed.get("classification") or "unknown").strip().lower() or "unknown"
        if classification in _TRANSIENT_CLASSIFICATIONS:
            classification = "transient_error"
        raw_needs = parsed.get("needs_attention")
        if isinstance(raw_needs, bool):
            needs_attention = raw_needs
        elif isinstance(raw_needs, str) and raw_needs.strip().lower() in ("true", "false"):
            needs_attention = raw_needs.strip().lower() == "true"
        else:
            needs_attention = classification not in ("background_running", "still_running", "transient_error")

        label = clean_single_line(str(parsed.get("label") or ""), 40)
        summary = clean_single_line(str(parsed.get("summary") or ""), 260)
        if not summary and label:
            summary = label
        return {
            "ok": True,
            "classification": classification,
            "needs_attention": needs_attention,
            "label": label,
            "summary": summary,
        }

    return {
        "ok": True,
        "classification": "unknown",
        "needs_attention": True,
        "label": "",
        "summary": clean_single_line(text, 240),
    }


def default_hermes_verify(
    ctx: dict[str, Any],
    *,
    model: str = "",
    reasoning: str = "low",
    toolsets: str = "",
    timeout: int = 45,
    hermes_bin: str = "",
) -> dict[str, Any]:
    """Run an on-demand single-turn `hermes -z` verification pass; fail-open on errors."""
    bin_path = (hermes_bin or "").strip() or (
        DEFAULT_HERMES_BIN if Path(DEFAULT_HERMES_BIN).exists() else "hermes"
    )
    prompt = build_verification_prompt(ctx)
    cmd: list[str] = [bin_path]
    if model.strip():
        cmd.extend(["-m", model.strip()])
    if reasoning.strip():
        cmd.extend(["--reasoning", reasoning.strip()])
    if toolsets.strip():
        cmd.extend(["-t", toolsets.strip()])
    cmd.extend(["-z", prompt])

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout)),
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "source": "hermes-z"}

    if proc.returncode != 0:
        err = clean_single_line(proc.stderr or proc.stdout or f"exit {proc.returncode}", 200)
        return {"ok": False, "error": err, "source": "hermes-z"}

    verdict = parse_verification_output(proc.stdout or "")
    verdict["source"] = "hermes-z"
    return verdict


def default_hermes_send(target: str, message: str) -> tuple[bool, str]:
    """Deliver notification via local Hermes CLI without waking any LLM."""
    hermes_bin = DEFAULT_HERMES_BIN if Path(DEFAULT_HERMES_BIN).exists() else "hermes"
    try:
        proc = subprocess.run(
            [hermes_bin, "send", "-t", target, message],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.returncode == 0:
            return True, (proc.stdout or "sent").strip()
        err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
        return False, err
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


NotifierFn = Callable[[str, str], tuple[bool, str]]
AgentVerifierFn = Callable[[dict[str, Any]], dict[str, Any]]


class WatchdogRunner:
    """Polls registered watches read-only and dispatches notifications on state transitions."""

    def __init__(
        self,
        adapters: dict[str, AppAdapter] | None = None,
        store: WatchStore | None = None,
        notifier: NotifierFn | None = None,
        *,
        verifier: AgentVerifierFn | None = None,
        verify_agent: bool = False,
        verify_events: tuple[str, ...] | list[str] | None = DEFAULT_VERIFY_EVENTS,
        suppress_routine: bool = False,
        verify_model: str = "",
        verify_reasoning: str = "low",
        verify_timeout: int = 45,
        debounce_seconds: int = 0,
    ) -> None:
        self.adapters: dict[str, AppAdapter] = adapters or {
            "codex": CodexAdapter(),
            "dsh": DshAdapter(),
            "antigravity": AntigravityAdapter(),
            "cline": ClineAdapter(),
        }
        self.store = store or WatchStore()
        self.notifier = notifier or default_hermes_send
        self.verify_agent = bool(verify_agent)
        if verify_events is None:
            self.verify_events: tuple[str, ...] = DEFAULT_VERIFY_EVENTS
        else:
            self.verify_events = tuple(v for v in verify_events if v in ALL_NOTIFY_EVENTS)
        self.suppress_routine = bool(suppress_routine)
        self.verify_model = (verify_model or "").strip()
        self.verify_reasoning = (verify_reasoning or "low").strip()
        self.verify_timeout = max(5, int(verify_timeout))
        self.debounce_seconds = max(0, int(debounce_seconds))
        if verifier is not None:
            self.verifier: AgentVerifierFn = verifier
        else:
            self.verifier = lambda ctx: default_hermes_verify(
                ctx,
                model=self.verify_model,
                reasoning=self.verify_reasoning,
                timeout=self.verify_timeout,
            )

    def _recent_context(self, adapter: AppAdapter, session_id: str) -> tuple[str, list[dict[str, Any]]]:
        try:
            rd = adapter.read_session(
                session_id,
                tail_messages=3,
                max_chars_per_message=1000,
                include_tools=True,
            )
            msgs = [m for m in (rd.get("messages") or []) if isinstance(m, dict)]
            if msgs:
                non_tool = [m for m in msgs if m.get("role") != "tool"]
                last = non_tool[-1] if non_tool else msgs[-1]
                snippet = clean_single_line(f"{last.get('role', '')}: {last.get('content', '')}", 160)
                return snippet, msgs
        except Exception:  # noqa: BLE001
            pass
        return "", []

    def _latest_snippet(self, adapter: AppAdapter, session_id: str) -> str:
        snippet, _ = self._recent_context(adapter, session_id)
        return snippet

    def poll_once(self, *, dry_run: bool = False, now_ts: float | None = None) -> dict[str, Any]:
        """Run one poll cycle across all enabled watches."""
        now = time.time() if now_ts is None else float(now_ts)
        watches = self.store.list_all()
        checked = 0
        events: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        dirty = False

        for w in watches:
            if not w.get("enabled", True):
                continue
            checked += 1
            app = str(w.get("app", ""))
            sid = str(w.get("session_id", ""))
            adapter = self.adapters.get(app)
            if adapter is None:
                errors.append({"watch_id": w.get("watch_id"), "error": f"Unknown app '{app}'"})
                continue

            try:
                meta = adapter.get_session_meta(sid)
            except Exception as exc:  # noqa: BLE001
                errors.append({"watch_id": w.get("watch_id"), "error": str(exc)})
                continue

            if meta is None:
                errors.append({"watch_id": w.get("watch_id"), "error": f"Session '{sid}' not found in '{app}'"})
                continue

            event_type, reason = evaluate_watch(w, meta, now_ts=now)
            eff_debounce = (
                max(0, int(w["debounce_seconds"]))
                if w.get("debounce_seconds") is not None
                else self.debounce_seconds
            )
            if event_type is None:
                if w.get("pending_transition") and not dry_run:
                    w.pop("pending_transition", None)
                    dirty = True
            elif event_type in DEBOUNCE_EVENTS and eff_debounce > 0:
                pending = w.get("pending_transition")
                if (
                    isinstance(pending, dict)
                    and pending.get("event") == event_type
                    and pending.get("status") == meta.status
                ):
                    first_seen = float(pending.get("first_seen_at") or now)
                    if (now - first_seen) < eff_debounce:
                        if not dry_run:
                            pending["count"] = int(pending.get("count") or 1) + 1
                            dirty = True
                        continue
                    if not dry_run:
                        w.pop("pending_transition", None)
                        dirty = True
                else:
                    if not dry_run:
                        w["pending_transition"] = {
                            "event": event_type,
                            "status": meta.status,
                            "first_seen_at": now,
                            "count": 1,
                        }
                        dirty = True
                    continue
            elif w.get("pending_transition") and not dry_run:
                w.pop("pending_transition", None)
                dirty = True

            if event_type is not None:
                snippet, context_messages = self._recent_context(adapter, sid)
                combined_parts = [meta.preview or "", snippet] + [
                    str(m.get("content") or "") for m in context_messages
                ]
                bg_info = detect_background_job("\n".join(p for p in combined_parts if p))
                if bg_info["matched"] and event_type in ("completed", "stalled"):
                    effective_label = str(bg_info["label"])
                else:
                    effective_label = _EVENT_LABELS.get(event_type, event_type)

                watch_verify = w.get("verify_agent")
                use_verify = bool(watch_verify) if watch_verify is not None else self.verify_agent
                watch_verify_events = (
                    tuple(parse_verify_events(w.get("verify_events")))
                    if w.get("verify_events")
                    else self.verify_events
                )

                verdict: dict[str, Any] | None = None
                suppressed = False
                if use_verify and event_type in watch_verify_events:
                    ctx = {
                        "watch": dict(w),
                        "event_type": event_type,
                        "live_meta": meta,
                        "reason": reason,
                        "snippet": snippet,
                        "context_messages": context_messages,
                        "background_job": bg_info,
                    }
                    try:
                        verdict = self.verifier(ctx)
                        if not isinstance(verdict, dict):
                            verdict = {"ok": False, "error": "invalid verifier return type"}
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Level-2 verifier failed for %s:%s (%s): %s", app, sid, event_type, exc)
                        verdict = {"ok": False, "error": str(exc)}

                    if verdict.get("ok"):
                        if str(verdict.get("label") or "").strip():
                            effective_label = str(verdict["label"]).strip()
                        watch_suppress = w.get("suppress_routine")
                        use_suppress = bool(watch_suppress) if watch_suppress is not None else self.suppress_routine
                        if verdict.get("classification") == "transient_error":
                            suppressed = True
                        elif use_suppress and verdict.get("needs_attention") is False:
                            suppressed = True

                text = format_watch_notification(
                    w,
                    event_type,
                    meta,
                    reason=reason,
                    snippet=snippet,
                    label_override=effective_label,
                    verdict=verdict,
                )
                targets = parse_notify_targets(w.get("notify_targets"))
                deliveries: list[dict[str, Any]] = []
                if not dry_run:
                    if not suppressed:
                        for tgt in targets:
                            ok, info = self.notifier(tgt, text)
                            deliveries.append({"target": tgt, "ok": ok, "detail": info})
                        w["last_notified_event"] = event_type
                        w["last_notified_at"] = now
                        sent_map = w.get("last_notified_by_event")
                        if not isinstance(sent_map, dict):
                            sent_map = {}
                        sent_map[event_type] = now
                        w["last_notified_by_event"] = sent_map
                    else:
                        logger.info(
                            "Suppressed %s alert for %s:%s (%s)",
                            event_type,
                            app,
                            sid,
                            verdict.get("summary", "") if isinstance(verdict, dict) else "",
                        )
                        sup_map = w.get("last_suppressed_by_event")
                        if not isinstance(sup_map, dict):
                            sup_map = {}
                        sup_map[event_type] = now
                        w["last_suppressed_by_event"] = sup_map
                    if event_type == "stalled":
                        w["last_stalled_updated_at_ms"] = meta.updated_at_ms
                    if verdict is not None:
                        w["last_verification"] = {
                            "at": datetime.fromtimestamp(now, timezone.utc).astimezone().isoformat(timespec="seconds"),
                            "event": event_type,
                            "ok": bool(verdict.get("ok")),
                            "classification": verdict.get("classification", ""),
                            "needs_attention": verdict.get("needs_attention"),
                            "suppressed": suppressed,
                            "summary": verdict.get("summary", ""),
                            **({"error": verdict["error"]} if verdict.get("error") else {}),
                        }
                    dirty = True

                ev_entry: dict[str, Any] = {
                    "watch_id": w.get("watch_id"),
                    "alias": w.get("alias", ""),
                    "app": app,
                    "session_id": sid,
                    "title": meta.title,
                    "event": event_type,
                    "label": effective_label,
                    "reason": reason,
                    "targets": targets,
                    "dry_run": dry_run,
                    "suppressed": suppressed,
                    "background_job": bg_info,
                    "message": text,
                    "deliveries": deliveries,
                }
                if verdict is not None:
                    ev_entry["verification"] = verdict
                events.append(ev_entry)

            if not dry_run:
                if (
                    w.get("last_status") != meta.status
                    or w.get("last_status_detail") != meta.status_detail
                    or w.get("last_updated_at_ms") != meta.updated_at_ms
                    or w.get("title") != meta.title
                ):
                    w["last_status"] = meta.status
                    w["last_status_detail"] = meta.status_detail
                    w["last_updated_at_ms"] = meta.updated_at_ms
                    w["title"] = meta.title
                    dirty = True

        if dirty and not dry_run:
            self.store.save_all(watches)

        return {
            "checked_count": checked,
            "event_count": len(events),
            "events": events,
            "errors": errors,
        }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    env_verify = os.environ.get("AGENT_HUB_WATCHER_VERIFY_AGENT", "").strip().lower() in ("1", "true", "yes", "on")
    env_suppress = os.environ.get("AGENT_HUB_WATCHER_SUPPRESS_ROUTINE", "").strip().lower() in ("1", "true", "yes", "on")
    ap = argparse.ArgumentParser(description="Zero-token Agent Hub session watchdog daemon")
    ap.add_argument("--interval", type=int, default=25, help="Poll interval in seconds (default 25)")
    ap.add_argument("--once", action="store_true", help="Run one poll cycle and exit")
    ap.add_argument("--dry-run", action="store_true", help="Evaluate watches without sending or mutating state")
    ap.add_argument(
        "--verify-agent",
        action="store_true",
        default=env_verify,
        help="Enable Level-2 on-demand `hermes -z` verification on key state transitions",
    )
    ap.add_argument(
        "--verify-events",
        default=os.environ.get("AGENT_HUB_WATCHER_VERIFY_EVENTS", ",".join(DEFAULT_VERIFY_EVENTS)),
        help="Comma-separated events that trigger Level-2 verification (default: completed,stalled,error)",
    )
    ap.add_argument(
        "--verify-model",
        default=os.environ.get("AGENT_HUB_WATCHER_VERIFY_MODEL", ""),
        help="Optional model override passed to `hermes -m ... -z` during Level-2 verification",
    )
    ap.add_argument(
        "--verify-reasoning",
        default=os.environ.get("AGENT_HUB_WATCHER_VERIFY_REASONING", "low"),
        help="Reasoning level for Level-2 `hermes -z` verification (default: low)",
    )
    ap.add_argument(
        "--verify-timeout",
        type=int,
        default=int(os.environ.get("AGENT_HUB_WATCHER_VERIFY_TIMEOUT", "45") or 45),
        help="Timeout in seconds for Level-2 `hermes -z` verification (default: 45)",
    )
    ap.add_argument(
        "--suppress-routine",
        action="store_true",
        default=env_suppress,
        help="Suppress push notifications when Level-2 verifier reports needs_attention=false",
    )
    ap.add_argument(
        "--debounce-seconds",
        type=int,
        default=int(os.environ.get("AGENT_HUB_WATCHER_DEBOUNCE_SECONDS", str(DEFAULT_DAEMON_DEBOUNCE_SECONDS)) or DEFAULT_DAEMON_DEBOUNCE_SECONDS),
        help="Seconds a transient error/completed state must persist across consecutive polls (default: 25)",
    )
    args = ap.parse_args(argv)

    runner = WatchdogRunner(
        verify_agent=bool(args.verify_agent),
        verify_events=parse_verify_events(args.verify_events),
        suppress_routine=bool(args.suppress_routine),
        verify_model=args.verify_model,
        verify_reasoning=args.verify_reasoning,
        verify_timeout=args.verify_timeout,
        debounce_seconds=args.debounce_seconds,
    )
    if args.once:
        res = runner.poll_once(dry_run=args.dry_run)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    stop_requested = False

    def _handle_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        logger.info("Received signal %s; shutting down watcher...", signum)
        stop_requested = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    interval = max(5, int(args.interval))
    logger.info(
        "Agent Hub watchdog started (interval=%ss, debounce=%ss, verify_agent=%s, suppress_routine=%s, store=%s)",
        interval,
        runner.debounce_seconds,
        runner.verify_agent,
        runner.suppress_routine,
        runner.store.path,
    )
    while not stop_requested:
        try:
            res = runner.poll_once(dry_run=args.dry_run)
            if res["events"]:
                logger.info("Triggered %d watch event(s): %s", len(res["events"]), [e["event"] for e in res["events"]])
            if res["errors"]:
                logger.warning("Watch poll errors: %s", res["errors"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected watcher cycle failure: %s", exc)

        # Sleep in short increments for prompt signal exit
        slept = 0
        while not stop_requested and slept < interval:
            time.sleep(min(1, interval - slept))
            slept += 1

    logger.info("Agent Hub watchdog stopped cleanly.")


if __name__ == "__main__":
    main()
