"""Structured, machine-readable JSON-lines logging for SHAPER.

Every event carries the canonical schema fields required by the protocol:

    {"timestamp": ..., "run_id": ..., "stage": ..., "dataset": ...,
     "seed": ..., "coalition": [...], "policy": ..., "event": ...,
     "status": ...}

Extra fields are appended per event. Events are written as JSON lines so
they can be parsed with standard tooling.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from typing import Any, Dict, Optional

_LOG_LOCK = threading.Lock()

EVENT_FIELDS = (
    "timestamp",
    "run_id",
    "stage",
    "dataset",
    "seed",
    "coalition",
    "policy",
    "event",
    "status",
)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def make_event(
    *,
    run_id: str,
    stage: str,
    event: str,
    status: str = "info",
    dataset: Optional[str] = None,
    seed: Optional[int] = None,
    coalition: Optional[Any] = None,
    policy: Optional[str] = None,
    timestamp: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a canonical event record. Coalition may be a list of players,
    a string, or None."""
    rec: Dict[str, Any] = {
        "timestamp": timestamp or now_iso(),
        "run_id": run_id,
        "stage": stage,
        "dataset": dataset,
        "seed": seed,
        "coalition": _coerce_coalition(coalition),
        "policy": policy,
        "event": event,
        "status": status,
    }
    rec.update(extra)
    return rec


def _coerce_coalition(coalition: Any) -> Any:
    if coalition is None:
        return None
    if isinstance(coalition, (list, tuple, set, frozenset)):
        return sorted(str(c) for c in coalition)
    return str(coalition)


def render_console(rec: Dict[str, Any]) -> str:
    """Human-readable single-line rendering of one structured event.

    Layout: [HH:MM:SS] STAGE | event | status | key=value ...  (fields like
    `coalition` are joined with '+' for readability).
    """
    ts = rec.get("timestamp", "")[11:19] or "??:??:??"
    stage = str(rec.get("stage") or "-")
    event = str(rec.get("event") or "-")
    status = str(rec.get("status") or "info")
    parts = [f"[{ts}] {stage:<20} | {event:<28} | {status}"]
    for key, value in rec.items():
        if key in ("timestamp", "run_id", "stage", "event", "status"):
            continue
        if value is None:
            continue
        if isinstance(value, list):
            value = "+".join(str(v) for v in value)
        if isinstance(value, float):
            value = f"{value:.6g}"
        parts.append(f"{key}={value}")
    return " | ".join(parts)


class StructuredLogger:
    """Append-only JSON-lines logger with a stable event schema.

    Writes machine-readable events to `<log_dir>/<name>.jsonl` AND (unless
    `console=False`) a human-readable line to stdout, so every stage's
    activity is visible while it runs.
    """

    def __init__(self, log_dir: str, run_id: str, name: str = "events", console: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{name}.jsonl")
        self.run_id = run_id
        self.console = console

    def log(self, *, stage: str, event: str, status: str = "info", **fields: Any) -> Dict[str, Any]:
        rec = make_event(run_id=self.run_id, stage=stage, event=event, status=status, **fields)
        line = json.dumps(rec, sort_keys=True, default=str)
        with _LOG_LOCK:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        if self.console:
            with _LOG_LOCK:
                print(render_console(rec), flush=True)
        return rec

    # Convenience shorthands -------------------------------------------------
    def info(self, stage: str, event: str, **fields: Any) -> Dict[str, Any]:
        return self.log(stage=stage, event=event, status="info", **fields)

    def warning(self, stage: str, event: str, **fields: Any) -> Dict[str, Any]:
        return self.log(stage=stage, event=event, status="warning", **fields)

    def error(self, stage: str, event: str, **fields: Any) -> Dict[str, Any]:
        return self.log(stage=stage, event=event, status="error", **fields)

    def stage_start(self, stage: str, **fields: Any) -> Dict[str, Any]:
        return self.log(stage=stage, event="stage_start", status="started", **fields)

    def stage_end(self, stage: str, **fields: Any) -> Dict[str, Any]:
        return self.log(stage=stage, event="stage_end", status="completed", **fields)

    def stage_fail(self, stage: str, error: str, **fields: Any) -> Dict[str, Any]:
        return self.log(stage=stage, event="stage_end", status="failed", error=error, **fields)


__all__ = ["StructuredLogger", "make_event", "now_iso", "EVENT_FIELDS"]
