"""bani engine package.

Quick start:

    from engine import build_engine, Config
    cfg = Config.from_env(onnx_path="models/v4.int8.onnx")
    engine = await build_engine(config=cfg)
    app["engine"] = engine
    app.router.add_get("/ws", handle_websocket_aiohttp)

Or just rely on env vars (ONNX_PATH=...):

    engine = await build_engine()

Swap any of the four layers by passing an instance:

    from engine import build_engine, ShabadMatcher
    from my_custom import MyChirpBackend

    engine = await build_engine(
        asr=MyChirpBackend(api_key=...),
        matcher=ShabadMatcher(corpus, config=cfg),
        config=cfg,
    )

For low-level access (custom backends, fixtures, tests):

    from engine.protocols import ASRBackendProto, Matcher, StateMachine
    from engine.event_types import ShabadCandidate, LineMatch
"""

from __future__ import annotations

import logging
import os
import sys as _sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Public API surface — re-exported so callers can `from engine import X`.
from .asr import ASRBackend, OnnxBackend
from .config import Config
from .corpus import (
    BYTES_PER_SAMPLE,
    SAMPLE_RATE,
    SYNTHETIC_SHABAD_ID_MIN,
    ShabadCorpus,
)
from .event_types import LineMatch, ShabadCandidate
from .matcher import ShabadMatcher
from .matcher_state import MatcherStateMachine
from .protocols import (
    ASRBackendProto,
    Corpus,
    Matcher,
    StateMachine,
    StateMachineFactory,
)

logger = logging.getLogger("live_detection")


# Repo root resolver. In normal runs this is the directory two levels up
# from this file. In a PyInstaller bundle, `sys._MEIPASS` is the extraction
# dir holding the bundled data/ and engine/ at the same layout.
_REPO_ROOT = Path(getattr(_sys, "_MEIPASS", None) or Path(__file__).resolve().parent.parent)

# Default corpus artifact — single consolidated JSON. The per-shabad source
# directories live in the private bani/ repo and are NOT shipped here.
DEFAULT_CORPUS_PATH = _REPO_ROOT / "data" / "sggs_corpus.json"


def _configure_root_logging() -> None:
    """Set up root logging once. Safe to call multiple times."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # Optional persistent file logging. Set LOG_DIR to a writable path
    # to enable a rotating handler so the volume never fills up.
    log_dir = os.environ.get("LOG_DIR")
    if log_dir and not any(getattr(h, "_bani_persistent", False) for h in root.handlers):
        from logging.handlers import RotatingFileHandler
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                Path(log_dir) / "server.log",
                maxBytes=50 * 1024 * 1024,
                backupCount=10,
                encoding="utf-8",
            )
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            fh._bani_persistent = True  # type: ignore[attr-defined]
            root.addHandler(fh)
            logger.info(f"Persistent logging enabled: {log_dir}/server.log")
        except Exception as e:
            logger.warning(f"Could not enable persistent logging: {e}")


@dataclass
class Engine:
    """All long-lived state a server / sidecar holds in memory.

    Construct via `build_engine()` (recommended) or directly if you have
    your own corpus/asr/matcher instances (useful for tests). Pass into
    your aiohttp app as `app["engine"] = engine` — the handlers in
    `engine.routes` read it from there.

    Sessions are independent of each other: each `LiveDetectionSession`
    has its own state machine, audio buffer, and locked shabad — two
    concurrent clients never share. They share only the read-only
    `corpus`, `asr`, and `matcher` instances. ASR runs in the default
    ThreadPoolExecutor; ONNX Runtime releases the GIL so concurrent
    sessions get true parallelism on multi-core hardware.

    Sessions are keyed by client identity (the `cid` query param on
    `/ws`), NOT by WebSocket connection. A single session can outlive
    its current socket: when a client reconnects with the same `cid`
    we reattach to its prior session (see `sessions_by_client_id` and
    `engine.session.attach_or_swap_ws`) instead of starting over in
    identifying phase.
    """

    corpus: Corpus
    asr: ASRBackendProto
    matcher: Matcher
    config: Config = field(default_factory=Config)
    make_state_machine: StateMachineFactory = MatcherStateMachine

    # Per-process counters mutated by the WebSocket route. Live on Engine
    # so multiple sub-apps (rare) get the same view.
    active_sessions: int = 0
    recent_connections: dict[str, Any] = field(default_factory=dict)

    # Sessions kept alive across WebSocket reconnects, keyed by the
    # `cid` query param the client sends on /ws?cid=<uuid>. The client
    # generates a UUID once and persists it in localStorage, so the same
    # browser/app instance always reconnects with the same cid. On reconnect
    # we look up an existing session here and `reattach()` to it instead
    # of constructing a fresh one in identifying phase — that's how
    # Stop/Start mic, network blips, page reloads, and idle-timeout
    # reaping all preserve the locked shabad + line position.
    #
    # No TTL eviction is wired in yet. For single-user desktop this is
    # fine (~1 entry, dropped when the process exits). The hosted server
    # will eventually want an idle-eviction policy here — see
    # docs/architecture_review.md when that lands.
    # Bounded by `max_sessions_by_client_id` (default 1024). When the cap
    # is hit we evict the least-recently-used entry. Desktop never trips
    # this (one client_id per machine); the hosted variant relies on it to
    # avoid an unbounded leak per unique visitor.
    sessions_by_client_id: dict[str, Any] = field(default_factory=dict)
    max_sessions_by_client_id: int = 1024


async def build_engine(
    *,
    corpus: Optional[Corpus] = None,
    asr: Optional[ASRBackendProto] = None,
    matcher: Optional[Matcher] = None,
    state_machine_cls: type[Any] = MatcherStateMachine,
    config: Optional[Config] = None,
) -> Engine:
    """Build a fully-wired Engine.

    Common case — nothing required, reads everything from env:

        engine = await build_engine()       # uses Config.from_env() → cfg.onnx_path / cfg.corpus_path

    Override any layer by passing an instance:

        engine = await build_engine(asr=MyChirpBackend(), config=cfg)
        engine = await build_engine(matcher=MyMatcher(corpus, cfg))

    Defaults if not passed:
      - corpus     = ShabadCorpus(cfg.corpus_path or DEFAULT_CORPUS_PATH).load()
      - asr        = OnnxBackend(cfg.onnx_path, force_cpu=cfg.onnx_force_cpu, ...)
      - matcher    = ShabadMatcher(corpus, config=cfg)
      - state_machine_cls = MatcherStateMachine (a class, called per session)
      - config     = Config.from_env()

    The four layers (corpus, asr, matcher, state_machine_cls) are the only
    extension points. Everything else is plumbing.
    """
    _configure_root_logging()
    logger.info("Initializing engine...")

    cfg = config if config is not None else Config.from_env()

    if corpus is None:
        corpus_path = Path(cfg.corpus_path) if cfg.corpus_path else DEFAULT_CORPUS_PATH
        corpus = ShabadCorpus(corpus_path)
        corpus.load()  # type: ignore[attr-defined]

    if asr is None:
        if not cfg.onnx_path:
            raise ValueError(
                "build_engine: pass `asr=...` or set Config.onnx_path "
                "(env var ONNX_PATH)."
            )
        asr = OnnxBackend(
            cfg.onnx_path,
            force_cpu=cfg.onnx_force_cpu,
            num_threads=cfg.onnx_num_threads,
        )

    if matcher is None:
        matcher = ShabadMatcher(corpus, config=cfg)

    engine = Engine(
        corpus=corpus,
        asr=asr,
        matcher=matcher,
        config=cfg,
        make_state_machine=state_machine_cls,
    )
    logger.info("Engine ready")
    return engine


__all__ = [
    "build_engine",
    "Engine",
    # Types
    "ShabadCandidate",
    "LineMatch",
    # Default impls (named so callers can subclass)
    "ShabadCorpus",
    "ShabadMatcher",
    "MatcherStateMachine",
    "ASRBackend",
    "OnnxBackend",
    # Protocols (named so custom impls can `isinstance` against them)
    "Corpus",
    "ASRBackendProto",
    "Matcher",
    "StateMachine",
    "StateMachineFactory",
    # Config
    "Config",
    # Audio constants
    "SAMPLE_RATE",
    "BYTES_PER_SAMPLE",
    "SYNTHETIC_SHABAD_ID_MIN",
    # Default paths
    "DEFAULT_CORPUS_PATH",
]
