"""
matcher_state.py — Shared shabad ID + line tracking state machine.

This is the canonical implementation that BOTH the server engine
and the edge client (JS port) delegate to. Keeping a single class avoids
the silent drift problem where every server-side improvement to ranking,
hysteresis, auto-unlock, etc. required a parallel hand-port to JS.

Design constraints:
  - Pure: no async, no I/O, no audio, no websockets, no globals.
  - Inputs: ASR transcripts + duration_seconds (caller does ASR).
  - Outputs: list[dict] events (same shape the server already sends to
    the websocket; trivially JSON-serializable for the wire / fixtures).
  - State: every per-session field that influences a future tick lives
    on the instance.

What lives OUTSIDE this class:
  - audio buffer management (caller's responsibility)
  - actually running ASR / hybrid CTC (caller decides what transcript to feed)
  - websocket sends, peer-action-seq staleness guards (server only)
  - trie building (server-only optimization, doesn't affect logic)

Trace fixtures (tests/state_machine/fixtures/*.json) feed canned transcripts
through this class and snapshot the event stream. The JS port replays the
same fixtures and must produce byte-identical events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .config import Config


@dataclass
class StateSnapshot:
    """Serializable view of every field that affects a future tick.

    Used for fixture generation/replay. Equal snapshots after the same
    inputs prove Python and JS implementations are identical.
    """
    phase: str
    locked_shabad_id: Optional[int]
    current_line: int
    recent_match_scores: list[float]
    last_top1_shabad: Optional[int]
    consecutive_top1_count: int
    rank_evidence: dict[int, float]
    previous_locked_shabad_id: Optional[int]
    previous_locked_name: Optional[str]
    previous_locked_until: float
    last_track_change_duration: float
    last_sanity_check_duration: float
    sanity_fail_count: int
    last_identification_duration: float
    last_tracking_duration: float


class MatcherStateMachine:
    """Pure shabad-ID + line-tracking state machine.

    Wire it up:
        sm = MatcherStateMachine(matcher=shabad_matcher, config=Config())
        events = sm.tick_identification(transcript, duration_seconds)
        for ev in events:
            send_to_client(ev)

    The matcher object must expose the methods we depend on:
        find_candidates(text, top_k) -> list[ShabadCandidate]
        match_line(text, shabad_id, current_line) -> LineMatch | None
        fuzzy_keyword_bidir(window_norm, line_norm) -> float
        normalize_text(text) -> str
        get_verse_text(verse_dict) -> str
        shabad_line_texts: dict[int, list[str]]      (server build)
        corpus.get_lines(shabad_id) -> list[dict]
        corpus.shabad_names: dict[int, str]

    The SM is matcher-agnostic in principle but in practice we use the
    server's ShabadMatcher in Python and a JS-port mirror on the edge.
    """

    def __init__(self, matcher: Any, config: Optional[Config] = None):
        self.matcher = matcher
        self.config: Config = config or Config()

        # ----- Per-session override for hysteresis_margin (advanced
        # settings UI; clients can tune line-flicker dampening at
        # runtime without rebuilding). None = fall back to config. -----
        self.hysteresis_margin_override: Optional[float] = None

        # ----- Phase / lock state -----
        self.phase: str = "identifying"
        self.locked_shabad_id: Optional[int] = None
        self.current_line: int = 0  # 0-indexed; events emit 1-indexed

        # ----- Auto-unlock sliding window (exp 067) -----
        self._recent_match_scores: list[float] = []

        # ----- Consecutive-wins lock gate (exp 020) -----
        self._last_top1_shabad: Optional[int] = None
        self._consecutive_top1_count: int = 0

        # ----- Cross-window rank evidence (exp 069) -----
        self._rank_evidence: dict[int, float] = {}

        # ----- Previous-lock bias (exp 068) -----
        self._previous_locked_shabad_id: Optional[int] = None
        self._previous_locked_name: Optional[str] = None
        self._previous_locked_until: float = 0.0

        # ----- Grace + sanity-check timers -----
        self._last_track_change_duration: float = 0.0
        self._last_sanity_check_duration: float = 0.0
        self._sanity_fail_count: int = 0

        # ----- Tick-rate gating (caller checks should_run_*) -----
        self._last_identification_duration: float = 0.0
        self._last_tracking_duration: float = 0.0

    # ──────────────────────────────────────────────────────────────────
    #  Tick-gating helpers (so caller doesn't duplicate this logic)
    # ──────────────────────────────────────────────────────────────────
    def should_run_identification(self, duration: float) -> bool:
        if self.phase != "identifying":
            return False
        return (duration - self._last_identification_duration) >= self.config.identification_interval

    def should_run_tracking(self, duration: float) -> bool:
        if self.phase != "tracking":
            return False
        return (duration - self._last_tracking_duration) >= self.config.tracking_interval

    def should_run_sanity_check(self, duration: float) -> bool:
        if self.phase != "tracking" or self.locked_shabad_id is None:
            return False
        return (duration - self._last_sanity_check_duration) >= self.config.sanity_check_interval

    # Pre-await stamp: caller (server) bumps the tick timestamp BEFORE
    # awaiting ASR so that a concurrent on_track_change / reset that lands
    # during the await can't be clobbered by a post-await write. tick_*
    # methods are still idempotent w.r.t. the stamp for fixtures that call
    # them directly without an ASR await.
    def mark_identification_tick(self, duration: float) -> None:
        self._last_identification_duration = duration

    def mark_tracking_tick(self, duration: float) -> None:
        self._last_tracking_duration = duration

    def reset_tick_anchors(self) -> None:
        """Wipe the identification / tracking tick-rate anchors back to 0.

        Used by ``session.reattach()`` when a new WebSocket connects with
        an existing ``cid``: the audio buffer and virtual time cursor are
        reset to 0 too, and without resetting these anchors
        ``should_run_tracking(duration=0)`` would stay False until
        ``duration`` exceeded the stale anchor (potentially many minutes
        in the future) and tracking would freeze. This is the *only*
        SM-internal piece reattach has to touch; everything else (phase,
        locked_shabad_id, current_line) is exactly what we want to keep.
        """
        self._last_identification_duration = 0.0
        self._last_tracking_duration = 0.0

    def mark_sanity_tick(self, duration: float) -> None:
        self._last_sanity_check_duration = duration

    # ──────────────────────────────────────────────────────────────────
    #  Snapshot for fixture comparison
    # ──────────────────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable view of all state fields.

        Used by fixture generator and conformance test to verify Python
        and JS arrive at identical state after the same input sequence.
        """
        return {
            "phase": self.phase,
            "locked_shabad_id": self.locked_shabad_id,
            "current_line": self.current_line,
            "recent_match_scores": [round(s, 3) for s in self._recent_match_scores],
            "last_top1_shabad": self._last_top1_shabad,
            "consecutive_top1_count": self._consecutive_top1_count,
            "rank_evidence": {str(k): round(v, 4) for k, v in sorted(self._rank_evidence.items())},
            "previous_locked_shabad_id": self._previous_locked_shabad_id,
            "previous_locked_name": self._previous_locked_name,
            "previous_locked_until": round(self._previous_locked_until, 2),
            "last_track_change_duration": round(self._last_track_change_duration, 2),
            "last_sanity_check_duration": round(self._last_sanity_check_duration, 2),
            "sanity_fail_count": self._sanity_fail_count,
            "last_identification_duration": round(self._last_identification_duration, 2),
            "last_tracking_duration": round(self._last_tracking_duration, 2),
        }

    # ──────────────────────────────────────────────────────────────────
    #  Identification tick
    # ──────────────────────────────────────────────────────────────────
    def tick_identification(self, transcript: str, duration: float) -> list[dict[str, Any]]:
        """Process one identification window.

        transcript: ASR output for the most recent ~60s window
        duration:   audio duration available (capped at wall-clock)

        Emits some combination of:
          - status (silence / instrumental / no-matches / generic)
          - candidates (ranked top-N for UI)
          - locked (auto-lock fired)
        """
        events: list[dict[str, Any]] = []

        # Stale-tick guard: if a concurrent action (manual_lock, reset,
        # track_change) flipped us out of identifying while the caller was
        # awaiting ASR, this tick is stale. Drop silently.
        if self.phase != "identifying":
            return events

        self._last_identification_duration = duration

        # Silence / quiet gate is upstream (it needs raw audio) — we only
        # see transcripts, so empty/too-short means caller already trimmed.
        if not transcript or len(transcript.strip()) < 5:
            events.append({
                "type": "status",
                "phase": "identifying",
                "message": "Instrumental passage — waiting for vocals…",
                "reason": "instrumental",
                "duration_seconds": round(duration, 1),
                "raw_asr": transcript,
            })
            return events

        # 1. Find candidates via multi-line agreement (exp 020).
        candidates = self.matcher.find_candidates(transcript, top_k=10)

        # 2. Previous-lock bias (exp 068).
        prev_id = self._previous_locked_shabad_id
        if prev_id is not None and duration < self._previous_locked_until:
            bias = self.config.previous_lock_bias
            found = False
            for c in candidates:
                if c.shabad_id == prev_id:
                    c.score += bias
                    found = True
                    break
            if not found:
                # Re-run a wider search so the previous shabad is visible.
                extra = self.matcher.find_candidates(transcript, top_k=500)
                for c in extra:
                    if c.shabad_id == prev_id:
                        c.score += bias
                        candidates.append(c)
                        break
            candidates.sort(key=lambda x: x.score, reverse=True)
            for i, c in enumerate(candidates):
                c.rank = i + 1
        elif prev_id is not None and duration >= self._previous_locked_until:
            self._previous_locked_shabad_id = None
            self._previous_locked_name = None

        # 3. Cross-window rank evidence (exp 069).
        if self.config.rank_evidence_enabled and candidates:
            self._apply_rank_evidence(candidates)

        if not candidates:
            events.append({
                "type": "status",
                "phase": "identifying",
                "message": "No matches found yet…",
                "duration_seconds": round(duration, 1),
                "raw_asr": transcript,
            })
            return events

        # 4. Attach matched-line info for UI display on top-5.
        for c in candidates[:self.config.candidates_to_show]:
            m = self.matcher.match_line(transcript, c.shabad_id, current_line=0)
            if m:
                c.matched_line_index = m.line_index
                c.matched_line_text = m.text
                c.matched_line_translation = m.translation
                c.matched_line_transliteration = m.transliteration
                c.matched_line_score = m.score

        # 5. Consecutive-wins counter.
        top1_id = candidates[0].shabad_id
        if top1_id == self._last_top1_shabad:
            self._consecutive_top1_count += 1
        else:
            self._last_top1_shabad = top1_id
            self._consecutive_top1_count = 1

        # 6. Confidence = gap between #1 and #2.
        if len(candidates) >= 2:
            confidence = min((candidates[0].score - candidates[1].score) / 10.0, 1.0)
        else:
            confidence = 1.0

        # 7. Lock decision.
        if self._should_lock(candidates, duration, confidence):
            self.locked_shabad_id = candidates[0].shabad_id
            self.phase = "tracking"
            self.current_line = 0
            self._last_track_change_duration = duration
            events.append(self.build_locked_event(
                shabad_id=candidates[0].shabad_id,
                shabad_name=candidates[0].name,
                confidence=confidence,
                duration=duration,
                manual=False,
            ))
            return events

        # 8. Otherwise emit a candidates update.
        #
        # Stability filter: a shabad must have appeared in N ID windows
        # before it's shown to the user. Counts decay by 1 on absence
        # (capped at max_count when present) so transient pop-ins fade
        # out cleanly while previously-confident matches survive a few
        # noisy ticks. The lock decision above used the unfiltered
        # candidate set, so this is purely a display dampener against
        # the 2s churn. If the previous tick was slow (ASR taking >3s)
        # the threshold drops to 1 so a slow box isn't staring at an
        # Always show the raw top-5 by score. The multi-window stability filter
        # (candidate_seen_counts + min_consecutive_windows) was an experiment
        # that didn't pay off in practice: at the slower 5s identification
        # cadence (matched to web/edge.js for smoother UX) the list already
        # changes infrequently enough that hiding rank-N candidates until they
        # appear N times in a row was net-negative — users want to see the
        # current top-5 immediately, even if one slot is noisy.

        top5 = list(candidates[:self.config.candidates_to_show])
        if prev_id is not None and not any(c.shabad_id == prev_id for c in top5):
            for c in candidates:
                if c.shabad_id == prev_id:
                    if c.matched_line_index is None:
                        m = self.matcher.match_line(transcript, c.shabad_id, current_line=0)
                        if m:
                            c.matched_line_index = m.line_index
                            c.matched_line_text = m.text
                            c.matched_line_translation = m.translation
                            c.matched_line_score = m.score
                    top5.append(c)
                    break

        cand_dicts = [c.to_dict() for c in top5]
        if prev_id is not None:
            for d in cand_dicts:
                if d.get("shabad_id") == prev_id:
                    d["previous_locked"] = True

        events.append({
            "type": "candidates",
            "phase": "identifying",
            "duration_seconds": round(duration, 1),
            "candidates": cand_dicts,
            "previous_locked_shabad_id": prev_id,
            "confidence": round(confidence, 3),
            "locked": False,
            "raw_asr": transcript,
        })
        return events

    # ──────────────────────────────────────────────────────────────────
    #  Tracking tick
    # ──────────────────────────────────────────────────────────────────
    def tick_tracking(self, transcript: str, duration: float) -> list[dict[str, Any]]:
        """Process one tracking window. Emits status, line_update, or
        an unlock-trigger that morphs the SM back into identifying."""
        events: list[dict[str, Any]] = []

        # Stale-tick guard: if a concurrent action unlocked us while the
        # caller awaited ASR, this tick is stale.
        if self.phase != "tracking" or self.locked_shabad_id is None:
            return events

        self._last_tracking_duration = duration

        # Always emit a status so the UI's raw-ASR panel stays fresh.
        tstep = self.config.tracking_interval
        events.append({
            "type": "status",
            "phase": "tracking",
            "reason": "tracking",
            "raw_asr": transcript,
            "current_line": self.current_line + 1,  # 1-indexed
            "duration_seconds": round(duration, 1),
            "content_cursor": round(max(0.0, duration - tstep), 3),
        })

        if not transcript.strip() or len(transcript.strip()) < 5:
            return events

        match = self.matcher.match_line(transcript, self.locked_shabad_id, self.current_line)
        if not match:
            return events

        # Auto-unlock sliding window (only when auto_lock is on, i.e. benchmark
        # mode).  Desktop users lock manually and expect it to stay locked.
        if self.config.auto_lock_enabled:
            self._recent_match_scores.append(match.score)
            if len(self._recent_match_scores) > self.config.lost_window:
                self._recent_match_scores.pop(0)

            recent = sorted(self._recent_match_scores)
            median_score = recent[len(recent) // 2] if recent else 0.0
            dyn_thr = max(self.config.min_match_score, median_score * self.config.unlock_ratio)
            low_frames = sum(1 for s in self._recent_match_scores if s < dyn_thr)
            in_grace = (duration - self._last_track_change_duration) < self.config.track_change_grace

            if low_frames >= self.config.lost_threshold and not in_grace:
                events.extend(self._restart_identification(duration))
                return events

        # Hysteresis on line switch (exp 029). Per-session override
        # (set via session.set_tracking_overrides) shadows the config
        # default when present — lets the Advanced settings UI tune
        # line-flicker dampening without a rebuild.
        hysteresis = (
            self.hysteresis_margin_override
            if self.hysteresis_margin_override is not None
            else self.config.hysteresis_margin
        )
        displayed_line = self.current_line
        if match.line_index != displayed_line and match.margin >= hysteresis:
            displayed_line = match.line_index

        if displayed_line != self.current_line:
            confidence = min(match.score / 50.0, 1.0)
            lines = self.matcher.corpus.get_lines(self.locked_shabad_id)
            display_line_obj = lines[displayed_line] if displayed_line < len(lines) else None
            if display_line_obj:
                display_text = self.matcher.get_verse_text(display_line_obj)
                display_translation = display_line_obj.get("translation_english", "")
            else:
                display_text = match.text
                display_translation = match.translation

            events.append({
                "type": "line_update",
                "phase": "tracking",
                "shabad_id": self.locked_shabad_id,
                "current_line": displayed_line + 1,  # 1-indexed for client
                "verse_id": display_line_obj.get("verse_id") if display_line_obj else None,
                "total_lines": len(lines),
                "line_text": display_text,
                "line_translation": display_translation,
                "confidence": round(confidence, 3),
                "duration_seconds": round(duration, 1),
                "raw_asr": transcript,
                "content_cursor": round(max(0.0, duration - tstep), 3),
            })
            self.current_line = displayed_line

        return events

    # ──────────────────────────────────────────────────────────────────
    #  Sanity check (server-only — needs a 60s ASR pass while tracking)
    # ──────────────────────────────────────────────────────────────────
    def tick_sanity_check(self, id_transcript: str, duration: float) -> list[dict[str, Any]]:
        """Caller transcribed a fresh 60s window; verify the locked
        shabad still appears in top-K. Repeated failure forces unlock."""
        self._last_sanity_check_duration = duration
        events: list[dict[str, Any]] = []
        if self.phase != "tracking" or self.locked_shabad_id is None:
            return events
        if not id_transcript or len(id_transcript.strip()) < 10:
            return events

        cands = self.matcher.find_candidates(id_transcript, top_k=self.config.sanity_check_top_k)
        cand_ids = [c.shabad_id for c in cands]
        if self.locked_shabad_id not in cand_ids:
            self._sanity_fail_count += 1
            if self._sanity_fail_count >= self.config.sanity_check_fail_threshold:
                events.extend(self._restart_identification(duration, reason="sanity_check"))
        else:
            self._sanity_fail_count = 0
        return events

    # ──────────────────────────────────────────────────────────────────
    #  External actions
    # ──────────────────────────────────────────────────────────────────
    def manual_lock(self, shabad_id: int, start_line: int, duration: float) -> list[dict[str, Any]]:
        """User confirmed a candidate. Lock immediately, regardless of
        consecutive-wins / confidence gates."""
        events: list[dict[str, Any]] = []
        lines = self.matcher.corpus.get_lines(shabad_id)
        if not lines:
            events.append({
                "type": "error",
                "message": f"Shabad {shabad_id} not found",
            })
            return events

        self.locked_shabad_id = shabad_id
        self.phase = "tracking"
        self.current_line = start_line
        self._recent_match_scores = []
        self._last_track_change_duration = duration

        first_line = lines[0]
        gurmukhi = first_line.get("gurmukhi", {})
        if isinstance(gurmukhi, dict):
            shabad_name = gurmukhi.get("unicode", first_line.get("unicode", ""))[:50]
        else:
            shabad_name = first_line.get("unicode", "")[:50]

        events.append(self.build_locked_event(
            shabad_id=shabad_id,
            shabad_name=shabad_name,
            confidence=1.0,
            duration=duration,
            manual=True,
        ))

        if start_line < len(lines):
            line = lines[start_line]
            line_text = self.matcher.get_verse_text(line)
            events.append({
                "type": "line_update",
                "phase": "tracking",
                "shabad_id": shabad_id,
                "current_line": start_line + 1,
                "verse_id": line.get("verse_id"),
                "total_lines": len(lines),
                "line_text": line_text,
                "line_translation": line.get("translation_english", ""),
                "confidence": 1.0,
            })
        return events

    def reset(self, duration: float) -> list[dict[str, Any]]:
        """Wipe tracking state. Caller is responsible for trimming the
        audio buffer to a short tail (5s) so re-ID can bootstrap fast."""
        self.phase = "identifying"
        self.locked_shabad_id = None
        self.current_line = 0
        self._recent_match_scores = []
        self._last_sanity_check_duration = 0.0
        self._sanity_fail_count = 0
        self._last_top1_shabad = None
        self._consecutive_top1_count = 0
        self._rank_evidence = {}
        # Let next ID tick fire immediately.
        self._last_identification_duration = max(0.0, duration - self.config.identification_interval)
        self._last_tracking_duration = 0.0
        return [{
            "type": "status",
            "phase": "identifying",
            "message": "Reset — listening…",
            "reason": "reset",
        }]

    def on_track_change(
        self,
        duration: float,
        title: str,
        raagi: Optional[str] = None,
        track: Optional[str] = None,
        started_at: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """ICY metadata says the audio source switched. Reset state."""
        self.phase = "identifying"
        self.locked_shabad_id = None
        self.current_line = 0
        self._recent_match_scores = []
        self._last_sanity_check_duration = 0.0
        self._sanity_fail_count = 0
        self._last_track_change_duration = duration
        self._last_top1_shabad = None
        self._consecutive_top1_count = 0
        self._rank_evidence = {}
        self._last_identification_duration = 0.0
        self._last_tracking_duration = 0.0
        return [{
            "type": "track_change",
            "phase": "identifying",
            "title": title,
            "raagi": raagi,
            "track": track,
            "started_at": started_at,
        }]

    # ──────────────────────────────────────────────────────────────────
    #  Internals
    # ──────────────────────────────────────────────────────────────────
    def _should_lock(self, candidates: list, duration: float, confidence: float) -> bool:
        """exp 020: NO force-lock; consecutive_wins=2 + confidence threshold."""
        if not candidates:
            return False
        if not self.config.auto_lock_enabled:
            return False
        if duration < self.config.lock_min_duration:
            return False
        required = self.config.consecutive_wins
        if self._consecutive_top1_count < required:
            return False
        if confidence >= self.config.lock_confidence_threshold:
            return True
        if duration >= self.config.lock_mid_duration and confidence >= self.config.lock_mid_confidence:
            return True
        return False

    def _apply_rank_evidence(self, candidates: list) -> None:
        """Decay accumulator + add this window's reciprocal-rank credits +
        blend into raw scores. Mutates `candidates` in place."""
        decay = self.config.rank_evidence_decay
        topk = self.config.rank_evidence_topk
        k0 = self.config.rank_evidence_k0
        min_score = self.config.rank_evidence_min_score
        alpha = self.config.rank_evidence_alpha

        for sid in list(self._rank_evidence):
            self._rank_evidence[sid] *= decay
            if self._rank_evidence[sid] < 0.01:
                del self._rank_evidence[sid]

        for rank_idx, c in enumerate(candidates[:topk], start=1):
            if c.score < min_score:
                continue
            contrib = (c.score / 100.0) / (rank_idx + k0)
            self._rank_evidence[c.shabad_id] = self._rank_evidence.get(c.shabad_id, 0.0) + contrib

        if self._rank_evidence:
            max_ev = max(self._rank_evidence.values())
            if max_ev > 0:
                for c in candidates:
                    ev_norm = self._rank_evidence.get(c.shabad_id, 0.0) / max_ev
                    c.score = (1.0 - alpha) * c.score + alpha * 100.0 * ev_norm
                candidates.sort(key=lambda x: x.score, reverse=True)
                for i, c in enumerate(candidates):
                    c.rank = i + 1

    def _restart_identification(self, duration: float, reason: str = "lost_tracking") -> list[dict[str, Any]]:
        """Internal: fall back to identification, remembering the unlocked
        shabad for previous-lock bias."""
        if self.locked_shabad_id is not None:
            self._previous_locked_shabad_id = self.locked_shabad_id
            self._previous_locked_name = self.matcher.corpus.shabad_names.get(
                self.locked_shabad_id, f"Shabad {self.locked_shabad_id}"
            )
            self._previous_locked_until = duration + self.config.previous_lock_ttl

        self.phase = "identifying"
        self.locked_shabad_id = None
        self._recent_match_scores = []
        self._last_sanity_check_duration = 0.0
        self._sanity_fail_count = 0
        self._last_top1_shabad = None
        self._consecutive_top1_count = 0
        self._rank_evidence = {}
        self._last_identification_duration = 0.0

        return [{
            "type": "status",
            "phase": "identifying",
            "message": "Lost tracking, re-identifying shabad…",
            "reason": reason,
            "duration_seconds": round(duration, 1),
        }]

    def build_locked_event(
        self,
        *,
        shabad_id: int,
        shabad_name: str,
        confidence: float,
        duration: float,
        manual: bool,
    ) -> dict[str, Any]:
        """Build the verbose 'locked' event with all lines for client render.

        Called by ``tick_identification`` and ``manual_lock`` on a fresh
        lock, and by ``session.send_resume_snapshot`` to push a snapshot
        of the current locked state to a freshly-reattached WebSocket.
        """
        lines = self.matcher.corpus.get_lines(shabad_id)
        lines_data = []
        for i, line in enumerate(lines):
            lines_data.append({
                "index": i + 1,
                "text": self.matcher.get_verse_text(line),
                "translation": line.get("translation_english", ""),
                "transliteration": line.get("transliteration_english", ""),
            })
        ev: dict[str, Any] = {
            "type": "locked",
            "phase": "tracking",
            "shabad_id": shabad_id,
            "shabad_name": shabad_name,
            "total_lines": len(lines),
            "lines": lines_data,
            "confidence": round(confidence, 3),
            "duration_seconds": round(duration, 1),
        }
        if manual:
            ev["manual"] = True
        return ev
