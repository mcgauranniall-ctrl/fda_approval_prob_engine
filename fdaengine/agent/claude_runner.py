"""Headless Claude Code runs (`claude -p`), the APEX `run-review.ps1` pattern ported to Python.

Differences from APEX, because these runs read untrusted downloaded documents:
- tools are restricted to Read/Glob/Grep (`--tools`), so there is no shell, no web and no file writes;
- results come back as schema-validated structured output (`--json-schema`), and Python writes the files.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fdaengine import config

# Failure signatures carried over from APEX/scripts/run-review.ps1.
_AUTH = re.compile(r"Failed to authenticate|Invalid authentication|OAuth token .*(expired|revoked)|Please run /login",
                   re.IGNORECASE)
_LIMIT = re.compile(r"You've hit your (session|usage) limit|usage limit reached|rate_limit", re.IGNORECASE)


class ClaudeRunError(RuntimeError):
    def __init__(self, kind: str, message: str, hint: str):
        super().__init__(message)
        self.kind, self.hint = kind, hint


@dataclass
class RunResult:
    structured: dict | None
    text: str
    cost_usd: float | None
    turns: int | None
    duration_s: float
    events: list[dict] = field(default_factory=list)


def _describe(event: dict) -> str | None:
    """One-line progress text for a stream-json event."""
    if event.get("type") != "assistant":
        return None
    for block in event.get("message", {}).get("content", []):
        if block.get("type") == "tool_use":
            inp = block.get("input", {})
            target = inp.get("file_path") or inp.get("pattern") or inp.get("path") or ""
            return f"{block.get('name')}: {Path(str(target)).name if block.get('name') == 'Read' else target}"[:120]
    return None


class _Lock:
    """Atomic lock file per case, so the UI and a manual run can't both run Claude on one case."""

    def __init__(self, path: Path):
        self.path = path
        self.fd = None

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pid = self.path.read_text().strip() if self.path.exists() else ""
            if pid.isdigit() and not _pid_alive(int(pid)):
                self.path.unlink(missing_ok=True)       # stale lock from a dead process
                return self.__enter__()
            raise ClaudeRunError("locked", f"Another Claude run is active on this case (PID {pid}).",
                                 "Wait for it to finish, or delete the lock file if that process is gone.")
        os.write(self.fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, timeout=10,
                             check=False)
        return str(pid) in out.stdout
    except (OSError, subprocess.SubprocessError):
        return True     # can't tell: assume alive, so we never break a live run's lock


def run(prompt: str, cwd: Path, schema: dict | None = None, model: str = "opus",
        tools: str = "Read,Glob,Grep", on_progress: Callable[[str], None] | None = None,
        log_path: Path | None = None, timeout_s: int = 1800) -> RunResult:
    exe = config.claude_exe()
    if not exe:
        raise ClaudeRunError("missing", "Claude Code CLI not found.",
                             "Install Claude Code, or set CLAUDE_EXE in .env to the path of claude.exe.")
    cmd = [exe, "-p", "--model", model, "--tools", tools, "--permission-mode", "dontAsk",
           "--no-session-persistence", "--output-format", "stream-json", "--verbose"]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema, separators=(",", ":"))]

    start = time.monotonic()
    events: list[dict] = []
    stderr_lines: list[str] = []
    result_event: dict | None = None
    with _Lock(cwd / ".claude_run.lock"):
        log = open(log_path, "w", encoding="utf-8") if log_path else None
        try:
            # The prompt goes through stdin (keeps the command line short); closing stdin avoids APEX's 3 s stall.
            proc = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            proc.stdin.write(prompt)
            proc.stdin.close()
            for line in proc.stdout:
                if log:
                    log.write(line)
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    stderr_lines.append(line)
                    continue
                events.append(ev)
                if ev.get("type") == "result":
                    result_event = ev
                elif on_progress and (msg := _describe(ev)):
                    on_progress(msg)
                if time.monotonic() - start > timeout_s:
                    proc.kill()
                    raise ClaudeRunError("timeout", f"Claude run exceeded {timeout_s}s.", "Re-run, or raise timeout_s.")
            stderr_lines += proc.stderr.read().splitlines()
            code = proc.wait()
        finally:
            if log:
                log.close()

    blob = "\n".join(stderr_lines) + json.dumps(result_event or {})
    if _AUTH.search(blob):
        raise ClaudeRunError("auth", "Claude CLI authentication failed.", "Run `claude login` in a terminal, then retry.")
    if _LIMIT.search(blob):
        raise ClaudeRunError("limit", "Claude usage limit reached.", "Retry after your usage limit resets.")
    if result_event is None or code != 0 or result_event.get("is_error"):
        detail = (result_event or {}).get("result") or "\n".join(stderr_lines[-5:]) or f"exit code {code}"
        raise ClaudeRunError("exit", f"Claude run failed: {str(detail)[:400]}", "Check logs/claude_extract.jsonl and retry.")
    structured = result_event.get("structured_output")
    if schema is not None and not structured:
        raise ClaudeRunError("empty", "Claude finished without structured output.", "Retry; if it repeats, check the log.")
    return RunResult(structured=structured, text=result_event.get("result", ""),
                     cost_usd=result_event.get("total_cost_usd"), turns=result_event.get("num_turns"),
                     duration_s=time.monotonic() - start, events=events)
