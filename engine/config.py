"""Engine configuration: a single frozen dataclass.

All algorithm / server tuning knobs live on `Config`. Construct via
`Config()` for defaults, `Config.from_env()` to pick up env-var
overrides, or pass any field as a kwarg to `engine.build_engine(config=...)`.

The dataclass is frozen — once constructed, a Config is immutable. Use
`cfg.with_overrides(foo=...)` to derive a new instance with changes.

Env-var overrides are uppercase versions of the field names, e.g.
`LOCK_CONFIDENCE_THRESHOLD=0.7 CONSECUTIVE_WINS=1 python server.py`.
Two env vars use legacy names for back-compat:
  - `BANI_AUTO_LOCK`   → `auto_lock_enabled`
  - `IDLE_TIMEOUT_S`   → `idle_timeout_s`  (no rename, kept for clarity)

Adaptive runtime values (the tracking step that auto-grows on slow
hardware) are NOT on Config — they live on `LiveDetectionSession`
since they're per-connection mutable state, not config.

Default values come from iterative tuning; each threshold is annotated
with its experiment number for traceability.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass, fields


logger = logging.getLogger("live_detection")


# Env-var name → Config field name. Empty when the env var IS the
# uppercased field name (the common case, handled automatically).
_ENV_ALIASES: dict[str, str] = {
    "BANI_AUTO_LOCK": "auto_lock_enabled",
    "ALLOW_AUDIO_BURST": "allow_audio_burst",
    "MAX_SESSIONS": "max_sessions",
    "IDLE_TIMEOUT_S": "idle_timeout_s",
    "RATE_LIMIT_PER_IP": "rate_limit_per_ip",
    "RATE_WINDOW_S": "rate_window_s",
    "MAX_BUFFER_S": "max_buffer_s",
    "BANI_MODE": "mode",
    "BANI_SANITY_CHECK": "sanity_check_enabled",
    "MAX_ID_SECONDS": "max_id_seconds",
}


def _coerce(raw: str, type_str: str):
    """Coerce env-var string to the field's type. Returns (value, ok)."""
    try:
        if "bool" in type_str:
            return raw.lower() in ("1", "true", "yes", "on"), True
        if "int" in type_str:
            return int(raw), True
        if "float" in type_str:
            return float(raw), True
        return raw, True
    except ValueError:
        return None, False


@dataclass(frozen=True)
class Config:
    """All engine tuning constants. Immutable.

    Hand into `build_engine(config=Config(...))` or rely on the default
    `Config.from_env()` wiring that `build_engine()` does for you.
    """

    # ─── Phase 1: Shabad identification (growing window) ───────────────
    identification_interval: float = 5.0   # Seconds between ID updates (5s matches web/edge.js
                                           # ID_INTERVAL_SEC; reduces candidate-list thrash. The
                                           # 2s value was an experiment that produced 2.5× more
                                           # repaints with no measurable lock-accuracy win and
                                           # noisier UX.)
    lock_confidence_threshold: float = 0.8 # Confidence gap to auto-lock (exp 020: 0.8 optimal)
    lock_min_duration: float = 15.0        # Min seconds before lock allowed
    lock_mid_duration: float = 30.0        # After this, accept lower confidence
    lock_mid_confidence: float = 0.7       # Lower confidence accepted after mid_duration
    consecutive_wins: int = 2              # Same shabad must be #1 for N windows in a row
    candidates_to_show: int = 5            # Number of candidates sent to client (matches web)

    # ─── Phase 2: Line tracking (15s sliding window, exp 024-025) ──────
    tracking_window: float = 15.0          # Sliding window — matches eval
    tracking_step: float = 2.0             # Transcribe every Ns (session may adapt up)
    tracking_interval: float = 2.0         # Seconds between line updates

    # ─── Auto-unlock tuning (exp 067 → exp 068) ────────────────────────
    lost_threshold: int = 5                # How many of last lost_window frames must be low
    lost_window: int = 8                   # Sliding window size (frames) for lost detection
    min_match_score: float = 22.0          # Absolute floor for "low" frame
    unlock_ratio: float = 0.5              # Frame low if score < max(min_match_score,
                                           #   median(recent) * unlock_ratio)
    track_change_grace: float = 20.0       # Seconds after lock during which low scores
                                           #   don't force unlock

    # ─── Sanity check (every N seconds while locked) ───────────────────
    sanity_check_interval: float = 45.0
    sanity_check_top_k: int = 25
    sanity_check_fail_threshold: int = 2

    # ─── Previous-lock memory (exp 068) ────────────────────────────────
    previous_lock_bias: float = 6.0
    previous_lock_ttl: float = 120.0

    # ─── Matching ──────────────────────────────────────────────────────
    min_content_lines: int = 3             # Shabads w/ fewer lines: no agreement bonus
    good_line_threshold: int = 60          # Score threshold for "good" line

    # ─── Hard CTC hybrid (exp 030) ─────────────────────────────────────
    min_hard_words: int = 3                # Use hard CTC if >= this many words

    # ─── False-switch reduction (exp 029) ──────────────────────────────
    hysteresis_margin: float = 7.0

    # ─── Cross-window rank evidence (exp 069) ────────────────────────
    # Decaying accumulator of which shabads have been in top-K across
    # recent windows; blends into raw scores so the candidate order
    # doesn't flicker between near-tied alternatives. Web has had this
    # on by default since exp 069 and feels noticeably more stable;
    # desktop was off by oversight — aligning the defaults here.
    #
    # Temporarily OFF while we A/B test the simpler candidate-stability
    # filter (below). If the filter alone holds up, we'll delete this
    # whole rerank machinery — five tuning knobs and opaque blend math
    # vs three knobs and a counter.
    rank_evidence_enabled: bool = False
    rank_evidence_alpha: float = 0.4       # Blend: new = (1-a)*raw + a*100*evidence_norm
    rank_evidence_decay: float = 0.75      # Per-window decay (~17s half-life at 5s tick)
    rank_evidence_topk: int = 10
    rank_evidence_k0: float = 1.0          # Rank smoothing
    rank_evidence_min_score: float = 25.0  # Noise floor for evidence credit

    # ─── Candidate stability filter — REMOVED 2026-05-24 ────────
    # The multi-window seen-count filter was an experiment that hid useful
    # info. At the 5s identification cadence the raw top-5 by score is
    # already smooth enough. See git history if we ever want it back.


    # ─── Admin ─────────────────────────────────────────────────────────
    auto_lock_enabled: bool = False      # Default off; enable via BANI_AUTO_LOCK=true or allow_audio_burst

    # ─── Server hardening ──────────────────────────────────────────────
    max_sessions: int = 5                  # Hard cap on concurrent sessions
    idle_timeout_s: float = 3600.0         # Disconnect after N seconds of no audio
                                           # (1 h: long enough that backgrounded
                                           #  Chrome tabs with throttled audio
                                           #  don't trigger a phantom restart)
    rate_limit_per_ip: int = 20
    rate_window_s: float = 60.0
    max_buffer_s: float = 120.0
    allow_audio_burst: bool = False        # Benchmark/CI: skip wall-clock audio clamp
    # ─── Deployment mode + misc gates ──────────────────────────────────
    mode: str = ""                         # "" (server) or "desktop" (sidecar)
    sanity_check_enabled: bool = False     # Periodic top-K re-verify while locked
    max_id_seconds: int = 30               # Cap on tail audio for identification ASR (local ONNX)
    onnx_force_cpu: bool = False           # Force CPU provider (prod hosting; DML/CUDA may be unsafe under concurrency)
    onnx_num_threads: int = 0              # ORT intra_op threads per inference (0 = auto). Lower = more parallel sessions per box.
    onnx_path: str = ""                    # Path to .onnx model; required if no `asr=...` is passed to build_engine().
    corpus_path: str = ""                  # Path to sggs_corpus.json; defaults to <repo>/data/sggs_corpus.json
    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Build a Config from defaults + env vars + explicit overrides.

        Field types are read from the dataclass to coerce env-var strings.
        Unknown / un-coercible env vars are ignored (warnings printed).
        Explicit kwargs win over env vars.
        """
        env_overrides: dict[str, object] = {}
        for f in fields(cls):
            # Standard name: ENV_VAR matches FIELD_NAME.upper()
            raw = os.environ.get(f.name.upper())
            if raw is not None:
                val, ok = _coerce(raw, str(f.type))
                if ok:
                    env_overrides[f.name] = val
                    logger.info("[CONFIG override] %s = %r (from $%s)", f.name, val, f.name.upper())
        # Legacy aliases (e.g. BANI_AUTO_LOCK -> auto_lock_enabled)
        for env_name, field_name in _ENV_ALIASES.items():
            if field_name in env_overrides:
                continue  # already set via the standard name
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            f = next((x for x in fields(cls) if x.name == field_name), None)
            if f is None:
                continue
            val, ok = _coerce(raw, str(f.type))
            if ok:
                env_overrides[field_name] = val
                logger.info("[CONFIG override] %s = %r (from $%s)", field_name, val, env_name)
        cfg = cls(**{**env_overrides, **overrides})
        # Benchmark/burst mode implies auto-lock (no human to click candidates)
        if cfg.allow_audio_burst and "auto_lock_enabled" not in overrides and "auto_lock_enabled" not in env_overrides:
            cfg = dataclasses.replace(cfg, auto_lock_enabled=True)
            logger.info("[CONFIG] auto_lock_enabled=True (implied by allow_audio_burst)")
        return cfg

    def with_overrides(self, **kwargs) -> "Config":
        """Return a new Config with the given fields replaced."""
        return dataclasses.replace(self, **kwargs)


__all__ = [
    "Config",
]
