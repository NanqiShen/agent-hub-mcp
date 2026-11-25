# Agent Hub MCP (`agent-hub`)

> 📖 中文项目总结（含架构图、工具清单、安全设计）：[`docs/PROJECT_SUMMARY.zh.md`](docs/PROJECT_SUMMARY.zh.md) ｜ 事故复盘：[`docs/INCIDENT_CROSS_TALK_AND_RELAY_ARCHITECTURE.md`](docs/INCIDENT_CROSS_TALK_AND_RELAY_ARCHITECTURE.md)

Local MCP server, native bridges, and zero-token LaunchAgent watchdog that let **Hermes Agent** inspect sessions, manage task aliases, send messages into real desktop app conversations, relay prompts verbatim across agents, and push phone notifications on status transitions:

- **Codex** (read via `~/.codex/state_5.sqlite`, `thread_history_1.sqlite`, `goals_1.sqlite`; opt-in CDP write/create/model-switch & live ChatGPT Desktop DOM read via `~/.hermes/scripts/launch-codex-cdp.sh`, protected by cross-process app mutex + React Fiber routing + Two-Phase session locking)
- **DeepSeek Harness / DSH Desktop** (read via `session_projcache` + `session.v4.jsonl.zstd` content search; **native write/create/model-select/approve/cancel** via the `dsh-agent-hub-bridge` Cordis plugin over authenticated `127.0.0.1:43917`)
- **Antigravity App** (read via `conversation_summaries.db` + `conversations_fts` + `transcript*.jsonl`; opt-in CDP write/create/model-switch via `~/.hermes/scripts/launch-antigravity-cdp.sh`, protected by cross-process app mutex + Two-Phase session locking)
- **VS Code Cline** (read via `~/.cline/data/db/sessions.db`, `session-search.db`, task `messages.json` + VS Code `globalStorage`; **native write/new-task/approve/cancel** via the `local.agent-hub-cline-bridge` VS Code extension on `127.0.0.1:43918`, with Cline Hub WS fallback)

## Exposed MCP Tools (24)

### Read, Summary & Binding
1. `hub_overview(limit_per_app=5)`
2. `hub_list_sessions(app="all", active_only=False, include_archived=False, include_subagents=False, limit=20)`
3. `hub_search_sessions(query, app="all", limit=15)` — Full-text & message-body search across all 4 apps.
4. `hub_read_session(app, session_id, tail_messages=10, max_chars_per_message=6000, include_tools=False)` — Supports exact IDs, bound aliases, or fuzzy phrase matching.
5. `hub_summarize_session(app, session_id, tail_messages=16)` — Extract structured completion summary cards.
6. `hub_handover_session(app, session_id, target_hint="", tail_messages=16)` — Build lossless cross-agent handover packets.
7. `hub_bind_task(alias, app, session_id, note="")`
8. `hub_list_bindings()`
9. `hub_unbind_task(alias)`

### Write, Approval & Cross-Agent Relay
10. `hub_writer_status()` — Check live write-bridge availability for `dsh`, `codex`, `antigravity`, and `cline`.
11. `hub_send_message(app, session_id, message, sender_tag="[via Hermes]", mode="queue", model="", provider="", reasoning_effort="", dry_run=False, allow_duplicate=False, watch=False, ...)` — Inject a message directly into a verified desktop session with app-level mutex, Two-Phase routing verification, audit logging, and 60s duplicate protection.
12. `hub_relay_message(source_app, source_session_id, target_app, target_session_id, source_message_index=None, source_role="assistant", extract_code_block=False, override_text="", prefix_instruction="", mode="queue", model="", provider="", reasoning_effort="", dry_run=True, allow_duplicate=False, watch=False, ...)` — One-shot verbatim relay across apps (defaults to `dry_run=True`).
13. `hub_create_session(app, title="", workspace="", initial_message="", sender_tag="[via Hermes]", model="", provider="", reasoning_effort="", agent_preset="", dry_run=False, watch=False, ...)` — Create a new real session across all 4 apps (`dsh`, `cline`, `codex`, `antigravity`).
14. `hub_respond_approval(app, session_id, decision="yes", feedback="", dry_run=False)` — Approve or reject pending tool/command confirmations.
15. `hub_cancel_session(app, session_id)` — Interrupt an active running turn.
16. `hub_audit_recent(limit=20, app="all", action="")` — Inspect recent send/relay/approval/cancel/create audit records.

### Centralized Native Bi-Directional Relay (`relay_states.sqlite`)
17. `hub_relay_start(source_app, source_session_id, target_app, target_session_id, name="", bidirectional=True, settle_seconds=20, min_hop_interval_seconds=15, max_hops=20, notify_targets="weixin", notify_on_hop=False, notify_on_milestone=True, ...)` — Register an automated bi-directional relay loop with debounce, echo suppression, idempotent SQLite transaction tracking, and milestone-only alerts.
18. `hub_relay_list(include_disabled=False, include_recent_tx=5)` — List registered relay links and recent transaction logs.
19. `hub_relay_stop(relay_id_or_name, delete=False)` — Pause or delete a relay link.
20. `hub_relay_tick(relay_id="", dry_run=True)` — Run an immediate relay evaluation/dispatch pass.

### Zero-Token Watchdog (`ai.hermes.agent-hub-watcher`)
21. `hub_watch_add(app, session_id, alias="", notify_targets="qqbot", stall_minutes=15, notify_on="all", verify_agent=False, suppress_routine=False)`
22. `hub_watch_list()`
23. `hub_watch_remove(watch_id_or_alias)`
24. `hub_watch_check_now(dry_run=True, verify_agent=False, suppress_routine=False)`

## Running Tests & Verifying Writers

```bash
/opt/homebrew/opt/hermes-agent/libexec/bin/python -m unittest discover -s tests -v
/opt/homebrew/opt/hermes-agent/libexec/bin/python scripts/verify_writers.py --status
```
