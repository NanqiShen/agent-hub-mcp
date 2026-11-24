# Agent Hub MCP (`agent-hub`)

Local MCP server, native bridges, and zero-token LaunchAgent watchdog that let **Hermes Agent** inspect sessions, manage task aliases, send messages into real desktop app conversations, relay prompts verbatim across agents, and push phone notifications on status transitions:

- **Codex** (read via `~/.codex/state_5.sqlite`, `thread_history_1.sqlite`, `goals_1.sqlite`; optional opt-in CDP write/create/model-switch via `~/.hermes/scripts/launch-codex-cdp.sh`)
- **DeepSeek Harness / DSH Desktop** (read via `session_projcache` + `session.v4.jsonl.zstd` content search; **native write/create/model-select** via the `dsh-agent-hub-bridge` Cordis plugin over authenticated `127.0.0.1:43917`)
- **Antigravity App** (read via `conversation_summaries.db` + `conversations_fts` + `transcript*.jsonl`; optional opt-in CDP write/create/model-switch via `~/.hermes/scripts/launch-antigravity-cdp.sh`)
- **VS Code Cline** (read via `~/.cline/data/db/sessions.db`, `session-search.db`, task `messages.json` + VS Code `globalStorage`; **native write/new-task** via the `local.agent-hub-cline-bridge` VS Code extension on `127.0.0.1:43918`, with Cline Hub WS fallback)

## Exposed MCP Tools (15)

### Read & Binding
1. `hub_overview(limit_per_app=5)`
2. `hub_list_sessions(app="all", active_only=False, include_archived=False, include_subagents=False, limit=20)`
3. `hub_search_sessions(query, app="all", limit=15)` — Full-text & message-body search across all 4 apps.
4. `hub_read_session(app, session_id, tail_messages=10, max_chars_per_message=6000, include_tools=False)` — Supports exact IDs, bound aliases, or fuzzy phrase matching.
5. `hub_bind_task(alias, app, session_id, note="")`
6. `hub_list_bindings()`
7. `hub_unbind_task(alias)`

### Write & Cross-Agent Relay
8. `hub_writer_status()` — Check live write-bridge availability for `dsh`, `codex`, `antigravity`, and `cline`.
9. `hub_send_message(app, session_id, message, sender_tag="[via Hermes]", mode="queue", model="", provider="", reasoning_effort="", dry_run=False, allow_duplicate=False)` — Inject a message directly into a real desktop session (with optional model/provider/thinking-depth passthrough, fuzzy target resolution, audit logging, and 60s duplicate protection).
10. `hub_relay_message(source_app, source_session_id, target_app, target_session_id, source_message_index=None, source_role="assistant", extract_code_block=False, override_text="", prefix_instruction="", dry_run=True, allow_duplicate=False)` — Relay verbatim prompts across apps (defaults to `dry_run=True`).
11. `hub_create_session(app, title="", workspace="", initial_message="", sender_tag="[via Hermes]", model="", provider="", reasoning_effort="", agent_preset="", dry_run=False)` — Create a new real session across all 4 apps (`dsh`, `cline`, `codex`, `antigravity`) with optional model/provider/reasoning/preset configuration and initial prompt.

### Zero-Token Watchdog (`ai.hermes.agent-hub-watcher`)
12. `hub_watch_add(app, session_id, alias="", notify_targets="qqbot", stall_minutes=15, notify_on="all")`
13. `hub_watch_list()`
14. `hub_watch_remove(watch_id_or_alias)`
15. `hub_watch_check_now(dry_run=True)`

## Running Tests & Verifying Writers

```bash
/opt/homebrew/opt/hermes-agent/libexec/bin/python -m unittest discover -s tests -v
/opt/homebrew/opt/hermes-agent/libexec/bin/python scripts/verify_writers.py --status
```
