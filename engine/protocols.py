"""Engine extension points.

If you want to plug in a different ASR backend, matcher, or state
machine, implement one of these Protocols and pass your instance to
`engine.build_engine(...)`.

These are runtime-checkable Protocols (PEP 544) — you don't need to
inherit; matching the shape is enough. Inheriting from the Protocol is
allowed and gives you `isinstance(x, MyProto)` for free.

The shipped default impls are in:
    asr.OnnxBackend             ASRBackend
    matcher.ShabadMatcher       Matcher
    matcher_state.MatcherStateMachine   StateMachine
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

from .event_types import LineMatch, ShabadCandidate


@runtime_checkable
class Corpus(Protocol):
    """Shabad corpus: read-only mapping of shabad_id → verses.

    Implementations: `corpus.ShabadCorpus` (loads from JSON files on disk).
    """

    shabads: dict[int, dict]
    shabad_names: dict[int, str]

    def get_lines(self, shabad_id: int) -> list[dict]: ...


@runtime_checkable
class ASRBackendProto(Protocol):
    """Audio-to-Gurmukhi-text backend.

    The default `asr.OnnxBackend` runs IndicConformer v4 int8 on CPU. To
    plug in Whisper / Google Chirp / your own model, implement
    `transcribe()`. CTC-aware backends (used for trie-constrained decoding
    when a shabad is locked) additionally implement `extract_logprobs()`
    and `get_vocab()`; non-CTC backends return None from both and the
    engine transparently falls back to plain greedy ASR.

    `transcribe_async()` is the canonical entry point from async code (the
    session loop). The base ASRBackend provides a default that offloads
    `transcribe()` to a thread pool, so simple sync impls only need to
    define `transcribe()`. Override `transcribe_async()` directly for
    inherently async backends (streaming network APIs, etc.).
    """

    def transcribe(self, audio: np.ndarray) -> str: ...

    async def transcribe_async(self, audio: np.ndarray) -> str: ...

    def extract_logprobs(self, audio: np.ndarray) -> Optional[np.ndarray]: ...

    def get_vocab(self) -> Optional[list[str]]: ...

    @property
    def supports_ctc(self) -> bool: ...


class Matcher(Protocol):
    """Shabad matching: identification (Phase 1) + line tracking (Phase 2).

    Implementations: `matcher.ShabadMatcher` (multi-line agreement +
    fuzzy_kw_bidir). The state machine drives this via the methods
    below — it never touches the matcher's internal indices, so swapping
    in a TF-IDF or embedding matcher is purely a question of returning
    the same shapes.

    Internal helpers (`normalize_text`, `fuzzy_keyword_bidir`, etc.) are
    NOT part of the Protocol — they're free to differ per implementation.
    Only what the StateMachine consumes is contractually required.
    """

    corpus: Corpus

    def find_candidates(
        self, transcription: str, top_k: int = 10
    ) -> list[ShabadCandidate]: ...

    def match_line(
        self,
        transcription: str,
        shabad_id: int,
        current_line: int = 0,
    ) -> Optional[LineMatch]: ...

    def get_verse_text(self, verse: dict) -> str: ...


@runtime_checkable
class StateMachine(Protocol):
    """Phase / lock / line-tracking state machine.

    Implementations: `matcher_state.MatcherStateMachine` (also the canonical
    JS-port spec).

    Pure: no async, no I/O, no audio. Caller (session.py) feeds ASR
    transcripts in and forwards the returned event dicts to the WebSocket.
    """

    phase: str
    locked_shabad_id: Optional[int]
    current_line: int

    def should_run_identification(self, duration: float) -> bool: ...
    def should_run_tracking(self, duration: float) -> bool: ...
    def should_run_sanity_check(self, duration: float) -> bool: ...

    def mark_identification_tick(self, duration: float) -> None: ...
    def mark_tracking_tick(self, duration: float) -> None: ...
    def mark_sanity_tick(self, duration: float) -> None: ...

    def tick_identification(self, transcription: str, duration: float) -> list[dict]: ...
    def tick_tracking(self, transcription: str, duration: float) -> list[dict]: ...
    def tick_sanity_check(self, transcription: str, duration: float) -> list[dict]: ...

    def manual_lock(self, shabad_id: int, start_line: int, duration: float) -> list[dict]: ...
    def reset(self, duration: float) -> None: ...
    def on_track_change(
        self,
        duration: float,
        title: str,
        raagi: Optional[str],
        track: Optional[str],
        started_at: float,
    ) -> list[dict]: ...


# Factory type for constructing a state machine from a matcher + config.
# Lets callers pass any callable with the right shape — class constructor,
# functools.partial, lambda, whatever.
StateMachineFactory = Callable[..., StateMachine]


__all__ = [
    "Corpus",
    "ASRBackendProto",
    "Matcher",
    "StateMachine",
    "StateMachineFactory",
]
