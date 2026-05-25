"""LiveDetectionSession: per-WebSocket-connection session.

Owns:
  - audio buffer (incoming PCM, trimmed)
  - virtual-time cursor (for offline burst mode)
  - ASR scheduling (run_asr_loop background task)
  - hard-CTC trie (rebuilt on lock)

Delegates to:
  - ShabadMatcher for find_candidates() and match_line()
  - MatcherStateMachine for all phase / lock / line-tracking decisions

Emits wire events via the send_callback supplied at construction. The
shape of those events is owned by matcher_state.py; nothing here knows
about JSON.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import numpy as np

from .corpus import SAMPLE_RATE
from ._internal.hard_ctc import (
    build_shabad_lexicon,
    build_trie,
    greedy_decode_with_timestamps,
    hard_constrained_decode,
    load_shabad_lines as _load_shabad_lines,
)
from ._internal.banidb_phrases import fetch_phrases as _fetch_bias_phrases

# Type-only imports — pulled into a TYPE_CHECKING block so session.py
# doesn't pull in the package __init__ side-effects (logging configuration,
# etc.) at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Engine


SendCallback = Callable[[dict[str, Any]], Awaitable[None]]


logger = logging.getLogger("live_detection")


class LiveDetectionSession:
    """Manages a single live detection session.

    Handles the two-phase approach:
      1. Shabad identification (progressive narrowing)
      2. Line tracking (once locked)
    """

    def __init__(
        self,
        session_id: str,
        engine: "Engine",
        send_callback: Callable[[dict[str, Any]], Awaitable[None]],
    ):
        self.session_id = session_id
        self.engine = engine
        self.asr = engine.asr
        self._default_asr = engine.asr  # keep reference for reverting
        self.matcher = engine.matcher
        self.corpus = engine.corpus
        self.send = send_callback  # async function to send JSON to client

        # Audio buffer (session-only; SM is pure logic and never touches audio).
        self.audio_buffer: list[np.ndarray] = []
        self.total_samples = 0

        # State machine — single source of truth for all shabad-ID +
        # line-tracking decisions. Session keeps only audio/ASR/websocket
        # I/O concerns. Config is taken from the engine so env-var tweaks
        # flow through without a second knob to forget.
        self.sm = engine.make_state_machine(
            matcher=self.matcher,
            config=engine.config,
        )

        # Adaptive tracking step: starts at config.tracking_step. If ASR
        # inference consistently exceeds it, _run_tracking bumps this up so
        # ticks don't stack back-to-back on slow hardware. Per-session
        # mutable state — NOT config, which is frozen.
        self._tracking_step: float = engine.config.tracking_step
        self._slow_asr_count: int = 0

        # ─── Per-session tracking thresholds (Advanced settings UI) ──
        # Clients can override tracking_window / tracking_step at runtime
        # via the WS `set_tracking_overrides` command. None = fall back to
        # engine.config defaults. The hysteresis override lives on the SM
        # itself (sm.hysteresis_margin_override); these two are session
        # concerns because the audio buffer + asr cadence are session-scoped.
        self._tracking_window_override: Optional[float] = None
        self._tracking_step_override: Optional[float] = None

        # Hard CTC trie (built on lock, exp 030) — server-side optimization
        # that the SM is intentionally agnostic about (edge can't afford it).
        self._trie = None
        self._pa_vocab: Optional[list[str]] = None

        # Speech adaptation phrases for cloud ASR backends (Chirp 2).
        # Fetched from BaniDB once on lock; passed to transcribe_async on
        # every tracking call so Chirp biases toward the locked shabad's
        # actual lyrics. Empty list = no biasing applied.
        self._bias_phrases: list[dict] = []
        self._bias_fetch_task: Optional[asyncio.Task] = None

        # Timing
        self.start_time = time.time()
        self.last_audio_time = time.time()  # Wall-clock: last time audio arrived (for idle timeout)
        self._virtual_time = 0.0
        self._input_ended = False
        self._benchmark_done_sent = False

        # Serializes phase-transition mutations across concurrent peer
        # actions. manual_lock has multiple awaits between state writes and
        # client sends; without this lock, two concurrent confirms (or a
        # confirm racing a reset) can interleave and leave listeners seeing
        # inconsistent events. Scope: peer actions only — the audio loop's
        # own identification / tracking flow is naturally serialized.
        self._state_lock = asyncio.Lock()

        # Monotonic counter bumped by any peer action (manual_lock / reset).
        # _run_identification captures its value before the ~5s ASR await
        # and re-checks afterward: if a peer confirm/reset landed during the
        # await, the ASR's auto-lock is stale and is dropped rather than
        # overwriting the human's choice.
        self._peer_action_seq: int = 0

        # ─── Reattachment handles ─────────────────────────────────────
        # The WebSocket route handler stashes the current `ws`, `asr_task`,
        # and `_idle_watchdog` task here so a subsequent connect with the
        # same client_id (Engine.sessions_by_client_id) can cleanly tear
        # them down before swapping in its own. Without these handles a
        # superseding handler can't wait for the old asr_task to actually
        # stop, which would let two ASR loops race on a single audio
        # buffer for a tick or two. None when no WebSocket is attached.
        self._current_ws: Any = None
        self._asr_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None

    # ─── Properties exposed to caller (HTTP handler, tests) ───────────
    @property
    def duration_seconds(self) -> float:
        """Effective audio duration available for analysis.

        Capped at wall-clock elapsed since session start (+1s slack). This
        prevents upstream sources that pre-buffer audio (e.g. ICY radio
        proxies, file replay) from making identification fire several ticks
        earlier than a real live listener would experience.

        Bypass with allow_audio_burst=True for offline benchmarks.
        """
        audio_dur = self.total_samples / SAMPLE_RATE
        if self.engine.config.allow_audio_burst:
            return audio_dur
        wall_dur = time.time() - self.start_time + 1.0
        return min(audio_dur, wall_dur)

    @property
    def audio_duration_seconds(self) -> float:
        """Raw audio duration in buffer (uncapped). For internal windowing only."""
        return self.total_samples / SAMPLE_RATE

    # Property delegators into the state machine — pre-SM code reads these
    # as plain fields. Surface them as read-only properties so SM remains
    # the sole owner.
    @property
    def phase(self) -> str:
        return self.sm.phase

    @property
    def locked_shabad_id(self) -> Optional[int]:
        return self.sm.locked_shabad_id

    @property
    def current_line(self) -> int:
        return self.sm.current_line

    @property
    def _previous_locked_shabad_id(self) -> Optional[int]:
        return self.sm._previous_locked_shabad_id

    @property
    def _previous_locked_name(self) -> Optional[str]:
        return self.sm._previous_locked_name

    @property
    def last_identification_duration(self) -> float:
        return self.sm._last_identification_duration

    @property
    def last_tracking_duration(self) -> float:
        return self.sm._last_tracking_duration

    # ─── Audio buffer ─────────────────────────────────────────────────
    def _trim_buffer(self) -> None:
        """Cap audio buffer to config.max_buffer_s to prevent unbounded memory growth.

        Identification uses last 60s and tracking uses last 15s, so trimming
        is safe. `total_samples` is kept as the true running count for
        `duration_seconds`.
        """
        # In benchmark burst mode, run_asr_loop intentionally walks a
        # virtual cursor from t=0 through the full recording. Trimming
        # would discard the start of the track before that cursor reaches
        # it, making --no-realtime score the tail of the file over and over
        # instead of the real timeline.
        if self.engine.config.allow_audio_burst:
            return

        max_samples = int(self.engine.config.max_buffer_s * SAMPLE_RATE)
        buf_samples = sum(len(a) for a in self.audio_buffer)
        if buf_samples <= max_samples:
            return
        excess = buf_samples - max_samples
        while self.audio_buffer and excess > 0:
            first = self.audio_buffer[0]
            if len(first) <= excess:
                excess -= len(first)
                self.audio_buffer.pop(0)
            else:
                self.audio_buffer[0] = first[excess:]
                excess = 0

    def ingest_audio(self, audio_data: bytes) -> None:
        """Buffer incoming audio. MUST be fast and non-blocking — called
        from the WebSocket read loop which must not be held up by ASR.

        Args:
            audio_data: Raw PCM bytes (float32, 16kHz, mono)
        """
        samples = np.frombuffer(audio_data, dtype=np.float32).copy()
        # Defensive sanitisation: mic clients occasionally send NaN/Inf
        # samples (ScriptProcessorNode buffers can be detached/reused on
        # Safari). Replace non-finite values with 0 so downstream numpy ops
        # don't propagate NaN through the entire buffer (which would cause
        # ASR to emit only ⁇ tokens).
        if samples.size and not np.isfinite(samples).all():
            np.nan_to_num(samples, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Pre-buffered-source trim: if a client (radio ICY proxy, file
        # replay, etc.) sends several seconds of audio in one burst,
        # accepting it all would fire identification much earlier than a
        # real live listener would experience. Drop any samples that would
        # put us more than 1s ahead of wall-clock elapsed since session
        # start.
        # Bench bypass: config.allow_audio_burst lets clients stream audio
        # faster than real-time (used by benchmark_client.py --no-realtime).
        if not self.engine.config.allow_audio_burst:
            wall_elapsed = time.time() - self.start_time
            max_samples_so_far = int((wall_elapsed + 1.0) * SAMPLE_RATE)
            if self.total_samples + len(samples) > max_samples_so_far:
                allowed = max(0, max_samples_so_far - self.total_samples)
                if allowed <= 0:
                    return  # fully ahead of wall-clock, drop chunk entirely
                samples = samples[:allowed]
                if len(samples) == 0:
                    return

        self.audio_buffer.append(samples)
        self.total_samples += len(samples)
        self.last_audio_time = time.time()
        self._trim_buffer()

    # Backward-compat alias for external integrations that historically used
    # on_audio_chunk. Internal code uses ingest_audio directly.
    async def on_audio_chunk(self, audio_data: bytes) -> None:
        self.ingest_audio(audio_data)

    def mark_input_ended(self) -> None:
        """Signal that a benchmark/offline client has finished sending audio."""
        self._input_ended = True

    # ─── Main ASR loop ────────────────────────────────────────────────
    async def run_asr_loop(self) -> None:
        """Background task: polls the audio buffer and dispatches ASR when
        enough new audio has arrived. Runs independently of the WebSocket
        read loop so audio ingestion is never blocked by slow inference.

        Uses a virtual time cursor that advances at the identification /
        tracking interval pace. In real-time mode this naturally tracks the
        audio duration. In burst mode (benchmark --no-realtime) it steps
        through pre-buffered audio sequentially instead of jumping to the end.
        """
        while True:
            try:
                audio_dur = self.audio_duration_seconds

                if not self.audio_buffer:
                    await asyncio.sleep(0.1)
                    continue

                # Advance virtual time to match available audio, but don't
                # jump ahead — cap at the next interval boundary so ASR
                # processes sequentially.
                cad = getattr(self.asr, "cadence", None)
                if self.phase == "identifying":
                    interval = (cad.id_tick if cad and cad.id_tick else None) \
                        or self.engine.config.identification_interval
                elif self.phase == "tracking":
                    interval = (cad.track_tick if cad and cad.track_tick else None) \
                        or self._tracking_step
                else:
                    interval = 5.0

                # Virtual time can advance up to audio_dur but only in
                # interval-sized steps (one per loop iteration).
                if self._virtual_time + interval <= audio_dur:
                    self._virtual_time += interval
                elif self._virtual_time < audio_dur:
                    # Fractional final step
                    self._virtual_time = audio_dur

                duration = self._virtual_time

                if self.phase == "identifying":
                    if self.sm.should_run_identification(duration):
                        await self._run_identification()
                        self.sm.mark_identification_tick(duration)

                elif self.phase == "tracking":
                    if self.sm.should_run_tracking(duration):
                        await self._run_tracking()
                        self.sm.mark_tracking_tick(duration)

                # If virtual time has caught up to audio, wait for more
                if self._virtual_time >= audio_dur:
                    if self._input_ended and not self._benchmark_done_sent:
                        self._benchmark_done_sent = True
                        await self.send({
                            "type": "benchmark_done",
                            "duration_seconds": round(audio_dur, 1),
                            "phase": self.phase,
                            "locked_shabad_id": self.locked_shabad_id,
                        })
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(f"[{self.session_id}] run_asr_loop error, continuing")

            await asyncio.sleep(0.05)

    async def _run_identification(self) -> None:
        """Run shabad identification on accumulated audio.

        Audio handling and ASR live here; the actual matching/scoring/lock
        decisions are owned by self.sm (MatcherStateMachine). This keeps
        the wire-event semantics identical to the JS edge port.
        """
        duration = self._virtual_time
        logger.info(f"[{self.session_id}] Running identification at {duration:.1f}s")

        peer_seq_at_start = self._peer_action_seq

        full_audio = np.concatenate(self.audio_buffer)
        # Window up to virtual_time, not the full buffer
        vt_samples = int(self._virtual_time * SAMPLE_RATE)
        if vt_samples < len(full_audio):
            full_audio = full_audio[:vt_samples]
        # Drop the first ~3s — on macOS/WKWebView the CoreAudio input AudioUnit
        # ramps gain from near-zero over 2-3s after permission grant, so those
        # samples are dead silence or near-silent transients. Including them
        # makes CTC normalize against junk and collapse the whole window to <unk>.
        # Skip the trim if the whole buffer is short (benchmarks, first call).
        WARMUP_TRIM_SAMPLES = int(3.0 * SAMPLE_RATE)
        if len(full_audio) > WARMUP_TRIM_SAMPLES + int(2.0 * SAMPLE_RATE):
            full_audio = full_audio[WARMUP_TRIM_SAMPLES:]
        audio_max = np.abs(full_audio).max()
        audio_mean = np.abs(full_audio).mean()
        logger.info(
            f"[{self.session_id}] Audio stats: len={len(full_audio)}, "
            f"max={audio_max:.4f}, mean={audio_mean:.6f}"
        )

        # Silence gate (audio-domain, lives in caller — SM never sees raw
        # audio). Thresholds are low enough that a laptop mic across the
        # room from a speaker still passes — the ASR normalizer
        # (transcribe()) will boost the signal to peak=0.8 before inference
        # anyway, so we only need to reject true silence / noise floor here.
        if audio_max < 0.005 or audio_mean < 0.0003:
            logger.info(f"[{self.session_id}] Audio too quiet, skipping identification")
            await self.send({
                "type": "status",
                "phase": "identifying",
                "message": "Awaiting audio…",
                "reason": "silence",
                "duration_seconds": duration,
            })
            return

        # Limit tail audio for identification ASR (default 60s, see Config).
        # Backend cadence can override (e.g. paid Google batch uses 12s).
        cad = getattr(self.asr, "cadence", None)
        id_window = (cad.id_window if cad and cad.id_window else None) \
            or self.engine.config.max_id_seconds
        max_samples = int(id_window * SAMPLE_RATE)
        if len(full_audio) > max_samples:
            full_audio = full_audio[-max_samples:]

        start = time.time()
        transcription = await self.asr.transcribe_async(full_audio)
        asr_time = time.time() - start
        logger.info(f"[{self.session_id}] ASR took {asr_time:.2f}s, text: '{transcription[:100]}'")

        # Mirror the auto-lock kill-switch into the SM (cheap; one tick).
        # auto_lock_enabled is on Config (set once at engine build, possibly
        # overridden by BANI_AUTO_LOCK env var). SM reads it from self.config
        # directly; no per-tick mutation needed.

        # Hand off to the state machine
        events = self.sm.tick_identification(transcription, duration)

        # Stale-result guard: if a peer action landed during the ASR await,
        # tick_identification already saw the post-action phase and emitted
        # nothing (or its events were calculated against fresh state). The
        # remaining concern is the case where the SM's auto-lock fired with
        # state that pre-dated the peer action — handled by SM's phase
        # guard at the top of tick_identification; here we just log and
        # forward.
        if self._peer_action_seq != peer_seq_at_start:
            filtered: list[dict] = []
            for ev in events:
                if ev.get("type") == "locked":
                    logger.info(
                        f"[{self.session_id}] Dropping stale auto-lock: peer action "
                        f"(seq {peer_seq_at_start} → {self._peer_action_seq}) superseded"
                    )
                    continue
                if ev.get("type") == "candidates" and self.sm.phase != "identifying":
                    logger.info(
                        f"[{self.session_id}] Skipping stale candidates "
                        f"(phase now {self.sm.phase})"
                    )
                    continue
                filtered.append(ev)
            events = filtered

        # Side-effect the SM is intentionally agnostic about: hard-CTC trie
        # build (server-only optimization, exp 030).
        for ev in events:
            if ev.get("type") == "locked":
                self._build_trie(ev["shabad_id"])
                self._kickoff_bias_fetch(ev["shabad_id"])
                logger.info(f"[{self.session_id}] Locked onto shabad {ev['shabad_id']}")

        # Top-3 diagnostic log (exp 066) — reproduce the original DEBUG line
        # by inspecting the candidates event.
        for ev in events:
            if ev.get("type") == "candidates":
                cs = ev.get("candidates", [])
                if cs:
                    top_str = " | ".join(
                        f"#{i+1} id={c['shabad_id']} score={c.get('score', 0):.1f} "
                        f"{c.get('name', '')[:40]}"
                        for i, c in enumerate(cs[:3])
                    )
                    margin = (cs[0].get("score", 0) - cs[1].get("score", 0)) if len(cs) >= 2 else 0.0
                    logger.debug(
                        f"[{self.session_id}] Top candidates (margin={margin:.1f}): {top_str}"
                    )

        # Stamp ASR timing onto outgoing events (session-level concern,
        # not SM's — the SM is intentionally pure and timing-unaware).
        asr_ms = round(asr_time * 1000)
        for ev in events:
            ev["asr_ms"] = asr_ms

        for ev in events:
            await self.send(ev)

    # ─── Hard CTC trie (server-only optimization) ────────────────────
    def _build_trie(self, shabad_id: int) -> None:
        """Build the hard CTC trie for a locked shabad (exp 030).

        Only activates if the ASR backend supports CTC (exposes logprobs +
        vocab). Backends like Google Chirp or Whisper that only return text
        will skip this — _hybrid_transcribe falls back to greedy.
        """
        if not self.asr.supports_ctc:
            self._trie = None
            return
        try:
            verses = self.engine.corpus.get_lines(shabad_id)
            lines = _load_shabad_lines(verses)
            if len(lines) < 2:
                logger.warning(f"[{self.session_id}] Shabad {shabad_id}: <2 lines, no trie")
                self._trie = None
                return
            lexicon = build_shabad_lexicon(lines)
            self._pa_vocab = self.asr.get_vocab()
            self._trie = build_trie(lexicon, self._pa_vocab)
            logger.info(
                f"[{self.session_id}] Built trie for shabad {shabad_id}: "
                f"{len(lexicon)} words"
            )
        except Exception as e:
            logger.error(f"[{self.session_id}] Failed to build trie: {e}")
            self._trie = None

    # ─── Speech adaptation phrases for cloud ASR (Chirp 2) ───────────
    def _kickoff_bias_fetch(self, shabad_id: int) -> None:
        """Launch a background BaniDB fetch to build the phrase set for
        the locked shabad. Result lands in `self._bias_phrases` and is
        consumed by `_run_tracking()` on every Chirp call.

        Best-effort: failures (offline, BaniDB down, unknown shabad) just
        leave `_bias_phrases` empty and tracking proceeds unbiased.
        """
        # Cancel any in-flight fetch from a previous lock
        if self._bias_fetch_task and not self._bias_fetch_task.done():
            self._bias_fetch_task.cancel()
        self._bias_phrases = []

        async def _runner() -> None:
            try:
                phrases = await _fetch_bias_phrases(shabad_id)
                if phrases:
                    self._bias_phrases = phrases
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"[{self.session_id}] BaniDB phrase fetch failed: {e}"
                )

        try:
            loop = asyncio.get_running_loop()
            self._bias_fetch_task = loop.create_task(_runner())
        except RuntimeError:
            # No running loop (shouldn't happen inside session calls but
            # be defensive). Skip biasing for this lock.
            self._bias_fetch_task = None

    def _hybrid_transcribe(self, audio: np.ndarray) -> str:
        """Hybrid transcription: hard CTC if >= min_hard_words, else greedy
        fallback (exp 030). If trie is unavailable, falls back to plain
        greedy ASR.
        """
        if self._trie is None or self._pa_vocab is None:
            _t0 = time.time()
            result = self.asr.transcribe(audio)
            logger.info(f"[{self.session_id}] _hybrid(greedy-only) took {time.time()-_t0:.2f}s")
            return result

        _lp_t0 = time.time()
        logprobs = self.asr.extract_logprobs(audio)
        _lp_elapsed = time.time() - _lp_t0
        logger.info(f"[{self.session_id}] _hybrid extract_logprobs took {_lp_elapsed:.2f}s")
        if logprobs is None:
            return self.asr.transcribe(audio)

        # Greedy decode (always needed for fallback)
        g_text, _ = greedy_decode_with_timestamps(logprobs, self._pa_vocab)

        # Hard CTC decode
        h_text, _ = hard_constrained_decode(logprobs, self._pa_vocab, self._trie)

        # Hybrid: use hard CTC if enough words, else greedy
        min_words = self.engine.config.min_hard_words
        h_nw = len(h_text.split()) if h_text else 0
        if h_nw >= min_words:
            logger.info(f"[{self.session_id}] Hybrid→hard CTC ({h_nw} words): {h_text[:60]}")
            return h_text
        else:
            logger.info(
                f"[{self.session_id}] Hybrid→greedy fallback (hard had {h_nw} words): "
                f"{g_text[:60]}"
            )
            return g_text

    # ─── Line tracking tick ───────────────────────────────────────────
    async def _run_tracking(self) -> None:
        """Run line tracking on recent audio (15s window).

        Audio + ASR live here; matching/hysteresis/auto-unlock decisions
        are delegated to self.sm. The optional periodic sanity check (which
        needs a separate 60s ASR pass) is also driven from here but feeds
        the SM via tick_sanity_check.
        """
        if not self.locked_shabad_id:
            return

        logger.info(f"[{self.session_id}] Running line tracking")

        full_audio = np.concatenate(self.audio_buffer)
        # Window up to virtual_time, not the full buffer
        vt_samples = int(self._virtual_time * SAMPLE_RATE)
        if vt_samples < len(full_audio):
            full_audio = full_audio[:vt_samples]
        duration = self._virtual_time

        # Periodic sanity check (exp 067): re-identify on a 60s window and
        # confirm the locked shabad is still in top-K. DISABLED by default:
        # the 60s inference (11-14s on slow hardware) freezes line tracking
        # completely and fires every 45s, wasting ~30% of tracking time.
        # When the user manually confirmed a shabad, sanity checking adds
        # no value. Unattended auto-lock servers can re-enable by setting
        # SANITY_CHECK_ENABLED=1 (or BANI_SANITY_CHECK=true legacy alias).
        if self.engine.config.sanity_check_enabled and self.sm.should_run_sanity_check(duration):
            self.sm.mark_sanity_tick(duration)
            id_window = int(60.0 * SAMPLE_RATE)
            id_audio = full_audio[-id_window:] if len(full_audio) > id_window else full_audio
            logger.info(
                f"[{self.session_id}] SANITY CHECK starting "
                f"({len(id_audio)/SAMPLE_RATE:.1f}s audio)"
            )
            try:
                _sc_t0 = time.time()
                id_text_raw = await self.asr.transcribe_async(id_audio)
                _sc_elapsed = time.time() - _sc_t0
                logger.info(f"[{self.session_id}] SANITY CHECK ASR took {_sc_elapsed:.2f}s")
                id_text = id_text_raw[0] if isinstance(id_text_raw, list) else id_text_raw
                if id_text and len(id_text.strip()) >= 10:
                    sanity_events = self.sm.tick_sanity_check(id_text, duration)
                    for ev in sanity_events:
                        await self.send(ev)
                    if self.sm.phase == "identifying":
                        return
            except Exception as e:
                logger.warning(f"[{self.session_id}] Sanity check error (non-fatal): {e}")

        # Sliding window for the tracking ASR (exp 024). Backend cadence
        # can override (e.g. paid Google batch uses 4s window).
        cad = getattr(self.asr, "cadence", None)
        tracking_window = (
            self._tracking_window_override
            or (cad.track_window if cad and cad.track_window else None)
            or self.engine.config.tracking_window
        )
        window_samples = int(tracking_window * SAMPLE_RATE)
        recent_audio = full_audio[-window_samples:] if len(full_audio) > window_samples else full_audio
        _trk_audio_secs = len(recent_audio) / SAMPLE_RATE
        logger.info(f"[{self.session_id}] Transcribing {_trk_audio_secs:.1f}s for tracking")

        _trk_t0 = time.time()
        # Hybrid CTC path (greedy + hard constrained) needs sync access to
        # extract_logprobs — only ONNX supports it. Non-CTC backends like
        # GoogleCloudASR don't implement sync transcribe(); call their
        # async path directly. Cloud backends receive the locked-shabad
        # phrase set for speech adaptation biasing (no-op for ONNX).
        if self.asr.supports_ctc:
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(None, self._hybrid_transcribe, recent_audio)
        else:
            transcription = await self.asr.transcribe_async(
                recent_audio,
                bias_phrases=self._bias_phrases or None,
            )
        _trk_elapsed = time.time() - _trk_t0
        logger.info(
            f"[{self.session_id}] Tracking ASR took {_trk_elapsed:.2f}s "
            f"({_trk_audio_secs:.0f}s audio), result: '{transcription[:60]}'"
        )

        # Skip noise transcripts — single characters or very short text
        # from near-silence audio (e.g. after mic is turned off). Without
        # this guard, random hallucinated characters cause line jumps.
        stripped = transcription.strip() if transcription else ""
        if len(stripped) < 5:
            logger.info(f"[{self.session_id}] Tracking: transcript too short ({len(stripped)} chars), skipping")
            return

        # Adaptive tracking step: if inference consistently exceeds the
        # configured step, bump self._tracking_step so we don't fire
        # back-to-back. On fast hardware (0.2s inference, 2.0s step) this
        # never triggers. On slow hardware (2.5s inference), step
        # auto-adjusts to ~3.1s so each tick has breathing room.
        # User-set override (Advanced settings) takes precedence over the
        # config default as the baseline; adaptive bumping still applies
        # on top of it (we never go BELOW the user's choice).
        configured_step = self._tracking_step_override or self.engine.config.tracking_step
        if _trk_elapsed > configured_step:
            self._slow_asr_count += 1
            if self._slow_asr_count >= 3:
                new_step = round(_trk_elapsed * 1.25, 1)
                if new_step != self._tracking_step:
                    logger.info(
                        f"[{self.session_id}] Adaptive step: {self._tracking_step}s -> "
                        f"{new_step}s (inference avg ~{_trk_elapsed:.1f}s)"
                    )
                    self._tracking_step = new_step
        else:
            self._slow_asr_count = 0
            # Restore original step if hardware catches up
            if self._tracking_step != configured_step:
                logger.info(f"[{self.session_id}] Adaptive step: restored to {configured_step}s")
                self._tracking_step = configured_step

        # Hand off to the state machine
        events = self.sm.tick_tracking(transcription, duration)
        asr_ms = round(_trk_elapsed * 1000)
        for ev in events:
            ev["asr_ms"] = asr_ms
            await self.send(ev)

    async def _restart_identification(self) -> None:
        """Restart identification phase after losing tracking. Caller-side
        wrapper around SM's restart logic — handles trie cleanup + send."""
        logger.info(f"[{self.session_id}] Lost tracking, restarting identification")
        events = self.sm._restart_identification(self.duration_seconds)
        self._trie = None
        self._pa_vocab = None
        self._bias_phrases = []
        if self._bias_fetch_task and not self._bias_fetch_task.done():
            self._bias_fetch_task.cancel()
        for ev in events:
            await self.send(ev)

    # ─── External actions ─────────────────────────────────────────────
    def set_asr_backend(self, backend) -> None:
        """Swap ASR backend at runtime (e.g. local ONNX → Google Cloud).
        Pass None to revert to the default engine ASR."""
        if backend is None:
            self.asr = self._default_asr
        else:
            self.asr = backend
        logger.info(f"[{self.session_id}] ASR backend: {type(self.asr).__name__}")

    def reattach(self, send_callback: SendCallback) -> None:
        """Re-bind this session to a fresh WebSocket connection.

        Called by the route handler when a client reconnects with the same
        ``cid`` (see Engine.sessions_by_client_id). The contract:

        Preserved (the whole point of reattaching):
          - SM state: phase, locked_shabad_id, current_line, hysteresis
          - Hard-CTC trie + Pa vocab (built on lock; expensive to rebuild)
          - BaniDB bias phrases (fetched async on lock)
          - Adaptive tracking step + slow-ASR counter

        Reset (the OLD WebSocket's timeline is dead, we get a fresh one):
          - Audio buffer + total_samples (stale PCM from prior socket)
          - Virtual time cursor (anchors on audio_duration which we just
            wiped — leaving it at the prior value would freeze the ASR
            loop until new audio caught up to that timestamp)
          - SM tick anchors (via ``sm.reset_tick_anchors()``):
            should_run_tracking() compares against virtual_time, so a
            stale anchor would mean the first tick fires N minutes late.
          - start_time / last_audio_time (wall-clock baseline for
            audio-burst protection in duration_seconds)
          - send callback (the OLD socket is gone or being closed)

        Caller is responsible for cancelling the prior asr_task /
        watchdog_task before invoking this and starting fresh ones after.
        See engine/routes.py::handle_websocket_aiohttp for the dance —
        or better, call ``attach_or_swap_ws()`` which encapsulates it.
        """
        self.send = send_callback
        self.audio_buffer = []
        self.total_samples = 0
        self.start_time = time.time()
        self.last_audio_time = time.time()
        self._virtual_time = 0.0
        self._input_ended = False
        self._benchmark_done_sent = False
        self.sm.reset_tick_anchors()

    async def attach_or_swap_ws(
        self,
        new_ws: Any,
        send_callback: SendCallback,
    ) -> None:
        """Bind this session to a fresh WebSocket, tearing down any prior
        one cleanly.

        For a freshly-constructed session ``_current_ws / _asr_task /
        _watchdog_task`` are all None and this just sets ``_current_ws``
        and calls ``reattach()`` (which is idempotent in that case — the
        audio buffer is already empty, virtual_time is already 0, etc.).

        For a session that's being reattached to a new socket (same cid,
        new /ws connect — see ``Engine.sessions_by_client_id``) this:

          1. Closes the prior WebSocket with code 1000 / "superseded"
             so its ``async for msg in ws`` loop exits, which is what
             ultimately stops the prior route handler from feeding stale
             audio into the shared session.
          2. Cancels and *awaits* the prior asr_task / watchdog_task.
             Awaiting is critical: without it the old asr_task could
             fire one more tick on what's about to become an empty
             buffer between us nulling it and the caller spinning up
             the new asr_task — a classic teardown race.
          3. Calls ``reattach()`` to wipe per-WS state while preserving
             SM state, trie, and bias phrases.
          4. Records the new ``_current_ws`` so a subsequent swap can
             find it.

        Caller (route handler) is still responsible for *creating* the
        new asr_task and watchdog_task after this returns and recording
        them on the session — those callables close over the new ws
        and send_json, which live in the handler's scope.
        """
        prev_ws = self._current_ws
        prev_asr = self._asr_task
        prev_wd = self._watchdog_task
        if prev_ws is not None and not prev_ws.closed:
            try:
                await prev_ws.close(code=1000, message=b"superseded")
            except Exception:
                pass
        for task in (prev_asr, prev_wd):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self.reattach(send_callback=send_callback)
        self._current_ws = new_ws

    async def detach_ws_handler(
        self,
        ws: Any,
        asr_task: "asyncio.Task[Any]",
        watchdog_task: "asyncio.Task[Any]",
    ) -> None:
        """Tear down a WS handler's background tasks. Symmetric with
        ``attach_or_swap_ws`` — call this from the route handler's
        ``finally`` block.

        The ``is``-checks before nulling are deliberate: if a superseding
        ``/ws`` connect with the same ``cid`` already swapped this
        session over to its own ws + tasks, our slots no longer point at
        the locals we were given, and we must NOT clobber the
        successor's plumbing. We still await our own (already-cancelled)
        tasks to release them from the event loop.
        """
        asr_task.cancel()
        watchdog_task.cancel()
        for t in (asr_task, watchdog_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        if self._asr_task is asr_task:
            self._asr_task = None
        if self._watchdog_task is watchdog_task:
            self._watchdog_task = None
        if self._current_ws is ws:
            self._current_ws = None

    async def send_resume_snapshot(self) -> None:
        """Emit a `locked` event reflecting the session's current tracking
        state so a freshly-attached WebSocket can render the right UI
        immediately.

        Called by the route handler right after reattach when the session
        is already in tracking phase. Without this, a client that lost
        its JS state to a page reload would sit in identification UI for
        up to one tracking_interval (~4-12s) until the next normal status
        tick happens to mention the lock — a confusing flash from
        "candidates" back to "tracking <shabad>".

        Reuses the same `locked` event shape the SM emits on initial lock
        (matcher_state.build_locked_event), with two additions:
          - current_line: lets the UI land on the right line instead of
            defaulting to line 1
          - resumed: True so telemetry can distinguish resumes from real
            new locks (and so the client could choose different copy)
        """
        if self.phase != "tracking" or self.locked_shabad_id is None:
            return
        lines = self.engine.corpus.get_lines(self.locked_shabad_id)
        if not lines:
            return
        # shabad_name derivation matches manual_lock() — keep it in sync if
        # the locked-event schema for shabad_name ever changes.
        first_line = lines[0]
        gurmukhi = first_line.get("gurmukhi", {})
        if isinstance(gurmukhi, dict):
            shabad_name = gurmukhi.get("unicode", first_line.get("unicode", ""))[:50]
        else:
            shabad_name = first_line.get("unicode", "")[:50]
        ev = self.sm.build_locked_event(
            shabad_id=self.locked_shabad_id,
            shabad_name=shabad_name,
            confidence=1.0,
            duration=self.duration_seconds,
            manual=True,
        )
        ev["current_line"] = self.current_line + 1  # SM 0-indexed → wire 1-indexed
        ev["resumed"] = True
        await self.send(ev)

    def set_tracking_overrides(
        self,
        *,
        window: Optional[float] = None,
        step: Optional[float] = None,
        hysteresis_margin: Optional[float] = None,
    ) -> dict[str, Any]:
        """Apply per-session overrides for the three tracking-phase knobs
        the Advanced settings UI exposes. Pass `None` for any value to
        clear that override (revert to ``engine.config`` default).

        Bounds are enforced server-side as safety rails — a malicious or
        confused client sending ``tracking_window=0.001`` shouldn't be
        able to thrash the ASR loop or wedge the matcher.

        Returns a dict of the *effective* values now in force (after
        clamping + override resolution), so the client can echo them
        back in its UI.
        """
        if window is None:
            self._tracking_window_override = None
        else:
            self._tracking_window_override = max(3.0, min(60.0, float(window)))

        if step is None:
            self._tracking_step_override = None
            # Also reset adaptive state so the loop resnaps to the
            # config default cleanly.
            self._slow_asr_count = 0
        else:
            self._tracking_step_override = max(1.0, min(10.0, float(step)))

        if hysteresis_margin is None:
            self.sm.hysteresis_margin_override = None
        else:
            self.sm.hysteresis_margin_override = max(0.0, min(30.0, float(hysteresis_margin)))

        effective = {
            "tracking_window": self._tracking_window_override or self.engine.config.tracking_window,
            "tracking_step": self._tracking_step_override or self.engine.config.tracking_step,
            "hysteresis_margin": (
                self.sm.hysteresis_margin_override
                if self.sm.hysteresis_margin_override is not None
                else self.sm.config.hysteresis_margin
            ),
        }
        logger.info(
            f"[{self.session_id}] Tracking overrides set: "
            f"window={self._tracking_window_override}, "
            f"step={self._tracking_step_override}, "
            f"hysteresis={self.sm.hysteresis_margin_override} → effective={effective}"
        )
        return effective

    def reset(self) -> None:
        """Reset session state. Keeps a short tail of recent audio so the
        next identification can bootstrap immediately."""
        tail_samples = int(5.0 * SAMPLE_RATE)
        kept: list[np.ndarray] = []
        have = 0
        for chunk in reversed(self.audio_buffer):
            kept.append(chunk)
            have += len(chunk)
            if have >= tail_samples:
                break
        kept.reverse()
        self.audio_buffer = kept
        self.total_samples = sum(len(c) for c in kept)
        self._trie = None
        self._pa_vocab = None
        self._bias_phrases = []
        if self._bias_fetch_task and not self._bias_fetch_task.done():
            self._bias_fetch_task.cancel()
        self._peer_action_seq += 1
        # Reset virtual time to match the trimmed buffer so the ASR loop
        # doesn't stall waiting for audio_duration to catch up.
        self._virtual_time = self.audio_duration_seconds
        # SM owns all matching/lock state and the tick-rate timestamps.
        self.sm.reset(self.duration_seconds)

    async def on_track_change(
        self,
        title: str,
        raagi: Optional[str] = None,
        track: Optional[str] = None,
    ) -> None:
        """External signal that the audio source switched to a new piece.
        Collapse state back to identification so we don't carry over a stale
        locked shabad — but keep a short tail of audio so ID can bootstrap
        without a gap.
        """
        logger.info(f"[{self.session_id}] Track change → {title!r}; resetting tracker")

        tail_samples = int(5.0 * SAMPLE_RATE)
        kept: list[np.ndarray] = []
        have = 0
        for chunk in reversed(self.audio_buffer):
            kept.append(chunk)
            have += len(chunk)
            if have >= tail_samples:
                break
        kept.reverse()
        self.audio_buffer = kept
        self.total_samples = sum(len(c) for c in kept)

        self._trie = None
        self._pa_vocab = None
        self._bias_phrases = []
        if self._bias_fetch_task and not self._bias_fetch_task.done():
            self._bias_fetch_task.cancel()
        events = self.sm.on_track_change(
            duration=self.duration_seconds,
            title=title,
            raagi=raagi,
            track=track,
            started_at=time.time(),
        )
        for ev in events:
            await self.send(ev)

    async def manual_lock(self, shabad_id: int, start_line: int = 0) -> None:
        """Manually lock onto a specific shabad (user confirmation).

        If the shabad isn't in the local corpus (e.g. Dasam Granth, Bhai
        Gurdas), fetches it from api.banidb.com and injects it before
        locking. This enables the search → lock flow for any shabad in
        BaniDB, not just SGGS.

        Serialized under `_state_lock` so concurrent peer confirms (or a
        confirm racing a peer reset) can't interleave their state writes
        and `send()` calls.
        """
        async with self._state_lock:
            # If shabad not in corpus, try fetching from BaniDB
            if not self.engine.corpus.get_lines(shabad_id):
                await self._fetch_and_inject_shabad(shabad_id)

            logger.info(
                f"[{self.session_id}] Manual lock onto shabad {shabad_id} at line {start_line}"
            )

            self._peer_action_seq += 1
            events = self.sm.manual_lock(
                shabad_id=shabad_id,
                start_line=start_line,
                duration=self.duration_seconds,
            )

            # Send events to the client FIRST — the UI should react
            # immediately. Trie build (below) takes ~50-200ms which
            # caused visible lag when it ran before the send.
            for ev in events:
                await self.send(ev)

            # Server-only side effect: build hard CTC trie for hybrid
            # decoding. Runs after the client already has the lock
            # confirmation so there's no user-visible delay.
            for ev in events:
                if ev.get("type") == "locked":
                    self._build_trie(shabad_id)
                    self._kickoff_bias_fetch(shabad_id)
                    break

    async def _fetch_and_inject_shabad(self, shabad_id: int) -> None:
        """Fetch a shabad from api.banidb.com and inject into the corpus.

        Called when the user searches for and locks a shabad that isn't in
        the local sggs_corpus.json (e.g. Dasam Granth, Bhai Gurdas, Nitnem
        banis not in SGGS). The fetched data is normalized to match the
        corpus schema so the matcher, SM, and CTC trie all work as usual.

        Note: this mutates engine.corpus from the session layer, which
        violates the pure-corpus boundary. Acceptable because: (1) it only
        fires on manual lock of out-of-corpus shabads (rare), (2) it only
        adds new keys (no mutation of existing data), (3) Python's GIL
        makes dict insertion atomic, (4) it runs under _state_lock so
        no concurrent manual_lock can race.
        """
        import aiohttp as _aiohttp
        url = f"https://api.banidb.com/v2/shabads/{shabad_id}"
        logger.info(f"[{self.session_id}] Shabad {shabad_id} not in corpus, fetching from BaniDB…")
        try:
            async with _aiohttp.ClientSession() as client:
                async with client.get(url, timeout=_aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning(f"[{self.session_id}] BaniDB returned {resp.status} for shabad {shabad_id}")
                        return
                    data = await resp.json()
            raw_verses = data.get("verses", [])
            if not raw_verses:
                logger.warning(f"[{self.session_id}] BaniDB returned empty verses for shabad {shabad_id}")
                return
            # Normalize to match sggs_corpus.json schema
            verses = []
            for v in raw_verses:
                verse_data = v.get("verse", {})
                translation = v.get("translation", {}).get("en", {})
                tr_text = translation.get("bdb") or translation.get("ms") or translation.get("ssk") or ""
                verses.append({
                    "unicode": verse_data.get("unicode", ""),
                    "gurmukhi": {"unicode": verse_data.get("unicode", "")},
                    "verse_id": v.get("verseId"),
                    "translation_english": tr_text,
                })
            shabad_data = {
                "shabad_id": shabad_id,
                "verses": verses,
            }
            self.engine.corpus._index_shabad(shabad_data)
            logger.info(
                f"[{self.session_id}] Injected BaniDB shabad {shabad_id}: "
                f"{len(verses)} verses"
            )
        except Exception as e:
            logger.warning(f"[{self.session_id}] Failed to fetch shabad {shabad_id} from BaniDB: {e}")


__all__ = ["LiveDetectionSession"]
