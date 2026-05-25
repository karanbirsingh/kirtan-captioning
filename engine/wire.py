"""WebSocket wire-protocol contract between engine and any client (mic.html
desktop, web /mic/, benchmark client, tests).

Single source of truth for:
  - the `PROTOCOL_VERSION` integer (bumped on any breaking change),
  - the event type names and their payload shapes,
  - the JSON commands the client may send.

The engine emits `protocol_version` in every `connected` event; clients warn
on mismatch. This file is import-only — it has no behavior, just types and
constants — so it's safe to import from anywhere without circular-import risk.

Versioning policy:
  - PATCH-equivalent (add an optional field): no bump
  - MINOR-equivalent (add a new event type or command): no bump but document
  - MAJOR-equivalent (rename/remove a field, change semantics): BUMP

If you add or remove an event type, also update web/matcher-state.js and the
JS conformance fixtures in tests/state_machine/ (or document why it's
server-only and the JS port doesn't need to know).
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

# Bumped to 1 on the 2026-05-24 cleanup: removed `set_show_all_candidates`
# command + `show_all_candidates` echo event + multi-window stability filter.
# Clients that still send the removed command get a logged warning but no
# error response (engine ignores unknown commands silently for back-compat).
PROTOCOL_VERSION: int = 1


# ─── Server → Client events ─────────────────────────────────────────────

class ConnectedEvent(TypedDict):
    type: Literal["connected"]
    session: str          # session id (uuid4 hex, 8 chars)
    protocol_version: int # always equals PROTOCOL_VERSION at send time
    server_version: str   # human-readable, e.g. "engine 0.2.0"


class StatusEvent(TypedDict, total=False):
    type: Literal["status"]
    phase: Literal["identifying", "tracking"]
    message: str
    reason: Literal["silence", "instrumental", "no-matches", "tracking", ""]
    duration_seconds: float
    raw_asr: str          # ASR transcript (may be empty)
    asr_ms: int           # last ASR call duration, milliseconds
    current_line: int     # tracking-phase only, 1-indexed


class CandidatesEvent(TypedDict, total=False):
    type: Literal["candidates"]
    phase: Literal["identifying"]
    duration_seconds: float
    candidates: list[dict[str, Any]]  # see engine/event_types.py ShabadCandidate.to_dict
    confidence: float
    locked: bool          # always False on this event; locked-state goes via LockedEvent
    raw_asr: str
    previous_locked_shabad_id: int | None


class LockedEvent(TypedDict, total=False):
    type: Literal["locked"]
    shabad_id: int
    shabad_name: str
    total_lines: int
    lines: list[dict[str, Any]]
    first_line: str
    verse_id: int | None


class LineUpdateEvent(TypedDict, total=False):
    type: Literal["line_update"]
    current_line: int     # 1-indexed for client display (engine uses 0-indexed internally)
    display_text: str
    line_text: str
    verse_id: int | None
    asr_ms: int
    raw_asr: str


class TrackChangeEvent(TypedDict, total=False):
    """Server informs client that a peer (admin) changed the active track.

    Resets the session to identifying. Currently emitted by the hosted
    server when SikhNet metadata changes; not used in desktop mode.
    """
    type: Literal["track_change"]
    phase: Literal["identifying"]
    title: str | None
    raagi: str | None
    track: str | None
    started_at: float


class TrackingOverridesEvent(TypedDict):
    """Confirmation echo of a `set_tracking_overrides` command."""
    type: Literal["tracking_overrides"]
    effective: dict[str, float]


class BenchmarkDoneEvent(TypedDict):
    """Emitted in benchmark-burst mode when we run out of pre-loaded audio."""
    type: Literal["benchmark_done"]
    duration_seconds: float
    phase: str
    locked_shabad_id: int | None


class PongEvent(TypedDict):
    type: Literal["pong"]
    t: float              # echo of client's `t` timestamp


class ErrorEvent(TypedDict):
    type: Literal["error"]
    message: str
    code: str             # e.g. "rate_limit", "session_full", "invalid_command"


# ─── Client → Server commands ───────────────────────────────────────────

class LockCommand(TypedDict):
    command: Literal["lock"]
    shabad_id: int


class ResetCommand(TypedDict):
    command: Literal["reset"]


class PingCommand(TypedDict, total=False):
    command: Literal["ping"]
    t: float              # echoed in PongEvent


class SetTrackingOverridesCommand(TypedDict, total=False):
    command: Literal["set_tracking_overrides"]
    tracking_window: float
    tracking_step: float
    hysteresis_margin: float


class TrackChangeCommand(TypedDict, total=False):
    """Admin-only: force the session into identifying phase with new metadata."""
    command: Literal["track_change"]
    title: str
    raagi: str
    track: str


# Removed in protocol_version=1: `set_show_all_candidates`. Engine silently
# ignores it for back-compat; clients should stop sending.


__all__ = [
    "PROTOCOL_VERSION",
    # Events
    "ConnectedEvent",
    "StatusEvent",
    "CandidatesEvent",
    "LockedEvent",
    "LineUpdateEvent",
    "TrackChangeEvent",
    "TrackingOverridesEvent",
    "BenchmarkDoneEvent",
    "PongEvent",
    "ErrorEvent",
    # Commands
    "LockCommand",
    "ResetCommand",
    "PingCommand",
    "SetTrackingOverridesCommand",
    "TrackChangeCommand",
]
