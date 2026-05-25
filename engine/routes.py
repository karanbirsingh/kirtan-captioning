"""HTTP / WebSocket route handlers.

These functions are aiohttp request handlers. The actual route wiring
lives at the application's entry point (e.g. `server.py`) which does
`app.router.add_get(...)` for each one.

Handlers access the shared engine instance via `request.app["engine"]`.
The application MUST set this before the first request is served — see
`engine.build_engine()` for the recommended construction path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys as _sys
import time
import uuid
from collections import deque
from functools import lru_cache
from pathlib import Path

import aiohttp
import aiohttp.web

from typing import Any

from .corpus import SAMPLE_RATE
from .session import LiveDetectionSession
from .wire import PROTOCOL_VERSION


logger = logging.getLogger("live_detection")

# Repo root resolver. In normal runs this is two levels up from this file
# (engine/ -> repo root). In a PyInstaller bundle, `sys._MEIPASS` is the
# extraction dir holding bundled data/ at the same relative layout.
_REPO_ROOT = Path(getattr(_sys, "_MEIPASS", None) or Path(__file__).resolve().parent.parent)


def _engine(request):
    """Fetch the shared engine instance from the aiohttp app.

    Raises RuntimeError with an actionable message if the app didn't
    register it — much friendlier than the bare KeyError you'd otherwise
    get on every request.
    """
    engine = request.app.get("engine")
    if engine is None:
        raise RuntimeError(
            "engine is not registered on app. Call "
            "`app['engine'] = await build_engine(...)` before starting "
            "the server."
        )
    return engine


def _is_ios_webkit_ua(request) -> bool:
    """True if the request looks like iOS Safari / iOS Chrome (CriOS).

    iOS WebKit has a longstanding bug (microsoft/onnxruntime#11679, four
    years of confirmed reports through iOS 17.6+) where multi-threaded
    WASM in onnxruntime-web silently hangs `InferenceSession.create` and
    `session.run` under sustained inference. The trigger is COOP/COEP
    headers → cross-origin-isolated context → SharedArrayBuffer enabled →
    ORT auto-enables the multi-thread WASM bundle.

    On iOS we therefore *skip* COOP/COEP entirely so the page stays in a
    non-isolated context, SAB stays unavailable, and ORT picks the
    single-thread WASM build. Our edge-worker.js also forces numThreads=1
    as a belt-and-suspenders measure.

    Desktop browsers keep COOP/COEP and threading because:
      - they primarily use WebGPU which doesn't care about COOP/COEP
      - if they fall back to WASM, threading on Chrome/Firefox/macOS
        Safari is reliable; only iOS WebKit hangs.

    Detection covers iPhone/iPad/iPod (native UA tokens) plus iOS Chrome
    which carries CriOS. iPadOS 13+ desktop-mode disguise (Mac UA +
    multi-touch) can't be detected server-side; those iPads fall through
    to the desktop COOP/COEP path — the worker-side numThreads=1 forcing
    catches them anyway.
    """
    ua = request.headers.get("User-Agent", "")
    return ("iPhone" in ua) or ("iPad" in ua) or ("iPod" in ua) or ("CriOS" in ua)


@lru_cache(maxsize=1)
def _mic_build_id() -> str:
    """Cache-busting token for /mic/ asset URLs.

    The HTML carries `__V__` literals next to each same-origin asset URL;
    we replace them with this token so any CDN in front of the server
    caches each deploy under a distinct key (next request after a new
    deploy is a guaranteed MISS).

    The token is the max mtime across the user-facing JS bundle. Each
    deploy ships new files, so the token rotates. Memoized via lru_cache.
    """
    mic_html = _REPO_ROOT / "desktop" / "frontend-stub" / "mic.html"
    if mic_html.exists():
        build_id = f"{int(mic_html.stat().st_mtime):x}"
    else:
        build_id = f"{int(time.time()):x}"
    logger.info("[mic] build_id=%s (cache-buster for /mic/ assets)", build_id)
    return build_id


# ─── WebSocket command handlers ─────────────────────────────────────────
# One async function per client command, registered in _CMD_HANDLERS below.
# Each takes (session, engine, data, send_json) and is wrapped in a per-call
# try/except by the dispatcher so a bug in one handler can't drop the WS for
# unrelated commands. See engine/wire.py for the canonical command schemas.

async def _cmd_reset(session, engine, data, send_json):
    session.reset()
    await send_json({
        "type": "status",
        "message": "Session reset",
        "phase": "identifying",
    })


async def _cmd_lock(session, engine, data, send_json):
    shabad_id = data.get("shabad_id")
    start_line = data.get("start_line", 0)
    if shabad_id:
        await session.manual_lock(shabad_id, start_line)


async def _cmd_ping(session, engine, data, send_json):
    await send_json({
        "type": "pong",
        "duration_seconds": session.duration_seconds,
        "phase": session.phase,
    })


async def _cmd_jump(session, engine, data, send_json):
    try:
        line = int(data.get("line", 0))
    except (TypeError, ValueError):
        line = 0
    total = len(engine.corpus.get_lines(session.locked_shabad_id or 0))
    if session.phase == "tracking" and 1 <= line <= total:
        session.sm.current_line = line - 1  # SM uses 0-indexed
        logger.info(f"[{session.session_id}] Jump to line {line}")


async def _cmd_end(session, engine, data, send_json):
    session.mark_input_ended()


async def _cmd_set_google_asr(session, engine, data, send_json):
    project_id = data.get("project_id", "").strip()
    if project_id:
        from .asr import GoogleCloudASR
        session.set_asr_backend(GoogleCloudASR(project_id))
        logger.info(f"[{session.session_id}] Switched to Google Chirp (batch)")
        await send_json({"type": "status", "message": "Switched to Google Chirp (batch)", "phase": session.phase})
    else:
        session.set_asr_backend(None)  # revert to default ONNX
        logger.info(f"[{session.session_id}] Reverted to local ONNX ASR")
        await send_json({"type": "status", "message": "Switched to local model", "phase": session.phase})


async def _cmd_set_tracking_overrides(session, engine, data, send_json):
    # Advanced settings UI passes any subset of
    # {tracking_window, tracking_step, hysteresis_margin}.
    # Missing keys are treated as "clear that override" so a partial
    # update is unambiguous — caller sends the full triple on every Apply.
    effective = session.set_tracking_overrides(
        window=data.get("tracking_window"),
        step=data.get("tracking_step"),
        hysteresis_margin=data.get("hysteresis_margin"),
    )
    await send_json({
        "type": "tracking_overrides",
        "effective": effective,
    })


_CMD_HANDLERS = {
    "reset": _cmd_reset,
    "lock": _cmd_lock,
    "ping": _cmd_ping,
    "jump": _cmd_jump,
    "end": _cmd_end,
    "set_google_asr": _cmd_set_google_asr,
    "set_tracking_overrides": _cmd_set_tracking_overrides,
}


# ─── WebSocket: /ws ───────────────────────────────────────────────────
async def handle_websocket_aiohttp(request):
    """Handle a WebSocket connection via aiohttp."""
    engine = _engine(request)
    cfg = engine.config
    from aiohttp import web, WSMsgType

    # --- Per-IP rate limit (before accepting connection) ---
    peer_ip = (
        request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        or (request.remote or "unknown")
    )
    now = time.time()
    dq = engine.recent_connections.setdefault(peer_ip, deque())
    while dq and now - dq[0] > cfg.rate_window_s:
        dq.popleft()
    if len(dq) >= cfg.rate_limit_per_ip:
        logger.warning(f"Rate limit hit for {peer_ip} ({len(dq)} in {cfg.rate_window_s}s)")
        return web.Response(status=429, text="Too many connections from this IP")
    dq.append(now)

    # --- Session cap ---
    if engine.active_sessions >= cfg.max_sessions:
        logger.warning(
            f"Session cap reached ({engine.active_sessions}/{cfg.max_sessions}), rejecting {peer_ip}"
        )
        return web.Response(status=503, text="Server at capacity, try again shortly")

    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)

    # --- Client identity & session reuse ---
    # The client sends ?cid=<uuid> (generated once, persisted in
    # localStorage). We look up an existing session in the engine pool
    # and reattach instead of constructing a fresh one — that's how
    # Stop/Start mic, page reload, network blip, and idle-timeout reaping
    # all preserve the locked shabad + line position. Cap the length so
    # a malicious caller can't pump arbitrary bytes into a dict key.
    client_id = request.query.get("cid", "").strip()[:64]

    async def send_json(data: dict) -> None:
        if ws.closed:
            return
        try:
            await ws.send_str(json.dumps(data, ensure_ascii=False))
        except (ConnectionResetError, ConnectionError):
            pass

    engine.active_sessions += 1
    existing = engine.sessions_by_client_id.get(client_id) if client_id else None
    if existing is not None:
        # Touch this cid as most-recently-used so the LRU eviction below
        # doesn't kill an active reattach mid-session. Plain dict is
        # insertion-ordered; popping + reinserting moves the key to the end.
        engine.sessions_by_client_id.pop(client_id, None)
        engine.sessions_by_client_id[client_id] = existing
        session = existing
        was_reattach = True
    else:
        session_id = str(uuid.uuid4())[:8]
        session = LiveDetectionSession(
            session_id=session_id,
            engine=engine,
            send_callback=send_json,
        )
        was_reattach = False  # default for both cid + no-cid first connect
        if client_id:
            # Bounded insertion: if we've hit the cap, evict the LRU entry.
            # dict preserves insertion order in Python 3.7+ so the first key
            # is the oldest. The hosted server can accumulate one entry per
            # unique visitor over its lifetime; without this it leaks.
            if len(engine.sessions_by_client_id) >= engine.max_sessions_by_client_id:
                oldest_cid = next(iter(engine.sessions_by_client_id))
                evicted = engine.sessions_by_client_id.pop(oldest_cid)
                logger.info(f"[engine] evicted LRU session cid={oldest_cid[:8]} (cap={engine.max_sessions_by_client_id})")
                # Best-effort teardown if the session still has any tasks.
                try:
                    await evicted.attach_or_swap_ws(None, None, None)
                except Exception:
                    logger.exception("[engine] error evicting session")
            engine.sessions_by_client_id[client_id] = session

    # Single entry point for both fresh-attach and reattach. The session
    # encapsulates the prev-ws-close / task-cancel / state-reset dance —
    # see LiveDetectionSession.attach_or_swap_ws for why ordering and
    # awaiting matter here.
    await session.attach_or_swap_ws(new_ws=ws, send_callback=send_json)
    session_id = session.session_id

    if was_reattach:
        logger.info(
            f"[{session_id}] Reattached WebSocket from {peer_ip} "
            f"(cid={client_id[:8]}, phase={session.phase}, "
            f"locked={session.locked_shabad_id}, "
            f"active: {engine.active_sessions}/{cfg.max_sessions})"
        )
    else:
        logger.info(
            f"[{session_id}] New WebSocket connection from {peer_ip} "
            f"(cid={client_id[:8] if client_id else 'none'}, "
            f"active: {engine.active_sessions}/{cfg.max_sessions})"
        )

    await send_json({
        "type": "connected",
        "session_id": session_id,
        "protocol_version": PROTOCOL_VERSION,
        "message": "Connected to live shabad detection server",
        "config": {
            "sample_rate": SAMPLE_RATE,
            "format": "float32",
            "channels": 1,
        }
    })

    # If we just reattached to an in-flight tracking session, push a
    # `locked` snapshot so the client UI lands directly in tracking
    # instead of flashing the identification view until the next tick.
    if existing is not None:
        await session.send_resume_snapshot()

    # --- Idle timeout watchdog ---
    async def _idle_watchdog():
        while not ws.closed:
            await asyncio.sleep(10)
            idle = time.time() - session.last_audio_time
            if idle > cfg.idle_timeout_s:
                logger.info(f"[{session_id}] Idle for {idle:.0f}s, closing")
                try:
                    await send_json({
                        "type": "status",
                        "message": "Idle timeout",
                        "phase": session.phase,
                    })
                except (ConnectionResetError, asyncio.CancelledError):
                    pass  # WS already dead; closing below
                await ws.close(code=1000, message=b"idle timeout")
                return

    watchdog_task = asyncio.create_task(_idle_watchdog())
    asr_task = asyncio.create_task(session.run_asr_loop())
    # Hand the tasks to the session so a future attach_or_swap_ws can
    # tear them down deterministically before re-binding to a new ws.
    session._asr_task = asr_task
    session._watchdog_task = watchdog_task

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                # Audio data — fast buffer append, never blocks on ASR
                session.ingest_audio(msg.data)
            elif msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning(f"[{session_id}] Invalid JSON: {msg.data[:100]}")
                    continue
                cmd = data.get("command")
                handler = _CMD_HANDLERS.get(cmd)
                if handler is None:
                    # Unknown command. Removed commands (set_show_all_candidates)
                    # land here too — we tolerate them silently for back-compat
                    # so older clients on newer servers don't break.
                    logger.debug(f"[{session_id}] Unknown command: {cmd!r}")
                    continue
                try:
                    await handler(session, engine, data, send_json)
                except Exception:
                    # Per-command isolation: a bug in one command (e.g. the
                    # manual_lock _build_locked_event typo, 2026-05-24) shouldn't
                    # drop the whole WebSocket. Log + send an error event + keep
                    # the connection alive. The outer except below still catches
                    # truly catastrophic exceptions (protocol corruption, etc).
                    logger.exception(f"[{session_id}] Command {cmd!r} failed")
                    try:
                        await send_json({
                            "type": "error",
                            "code": "command_failed",
                            "message": f"Command {cmd!r} failed; see server logs.",
                        })
                    except Exception:
                        pass  # send_json already swallows ConnectionError
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"[{session_id}] WebSocket error: {ws.exception()}")
    except Exception as e:
        # logger.exception() includes the traceback — critical for catching
        # bugs like the manual_lock _build_locked_event typo (2026-05-24)
        # that were being silently swallowed and dropping the WS.
        logger.exception(f"[{session_id}] Error: {e}")
    finally:
        # Hand our locals to the session; it'll cancel + await them and
        # null the matching slots iff *this* handler still owns them.
        # If a superseding /ws connect with the same cid already swapped
        # in its own ws + tasks, those `is`-checks fail and we don't
        # disturb the successor. See session.detach_ws_handler().
        await session.detach_ws_handler(ws, asr_task, watchdog_task)
        engine.active_sessions = max(0, engine.active_sessions - 1)
        logger.info(
            f"[{session_id}] Connection closed (active: {engine.active_sessions}/{cfg.max_sessions})"
        )

    return ws


# ─── /mic/ HTML + assets ──────────────────────────────────────────────────
async def handle_mic_page(request):
    """Serve the in-browser mic ASR page at /mic/.

    Architecture is edge-inference (onnxruntime-web + JS matcher) when
    served standalone, OR a thin WebSocket client when BANI_MODE=desktop
    (the Tauri sidecar mode — ASR + matching happen native-side).
    """
    if not request.path.endswith("/"):
        raise aiohttp.web.HTTPFound("/mic/")
    html_path = _REPO_ROOT / "desktop" / "frontend-stub" / "mic.html"
    if html_path.exists():
        html_text = html_path.read_text(encoding="utf-8").replace(
            "__V__", _mic_build_id()
        )

        # Desktop-app boot mode marker. When server.py --desktop starts,
        # it sets mode="desktop" on the engine Config before importing.
        # We surface that to the browser via a meta tag the page reads at
        # boot:
        #   - bani-mode absent: edge-worker.js + matcher-state.js + corpus
        #     fetch — the in-browser inference path.
        #   - bani-mode=desktop: skip the worker + corpus fetch, open a
        #     WebSocket to /ws and let the native sidecar do ASR + matching.
        if _engine(request).config.mode == "desktop":
            html_text = html_text.replace(
                "<head>",
                '<head>\n  <meta name="bani-mode" content="desktop">',
                1,
            )

        # COOP/COEP make the page "cross-origin isolated", which unlocks
        # SharedArrayBuffer → multi-threaded WASM in onnxruntime-web (2-4×
        # speed-up). On iOS WebKit this triggers a four-year-old hang bug
        # (microsoft/onnxruntime#11679), so we skip those headers for iOS.
        # COEP=credentialless lets cross-origin imports (CDN
        # onnxruntime-web) work without pinning specific CDN response
        # headers.
        headers = {"Cache-Control": "public, max-age=60, must-revalidate"}
        if not _is_ios_webkit_ua(request):
            headers["Cross-Origin-Opener-Policy"] = "same-origin"
            headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        # Vary on User-Agent so any CDN in front doesn't serve a cached
        # desktop response (with COOP/COEP) to an iOS client or vice versa.
        headers["Vary"] = "User-Agent"
        return aiohttp.web.Response(
            text=html_text,
            content_type="text/html",
            charset="utf-8",
            headers=headers,
        )
    return aiohttp.web.Response(text="mic.html not found", status=404)


async def handle_mic_asset(request):
    """Serve mic-page assets (model, vocab, corpus, matcher JS)."""
    filename = request.match_info["filename"]
    allowed = {
        "v4.int8.onnx", "edge-vocab.json", "edge-corpus.json",
        "edge-corpus.json.gz", "edge-matcher.js", "edge.js",
        "edge-worker.js",
        "matcher-state.js", "shabad-matcher.js",
        "edge-matcher-adapter.js", "hard-ctc.js",
        "icon-192.png", "manifest.json",
    }
    if filename not in allowed:
        return aiohttp.web.Response(text="Not found", status=404)
    # Check models/ for ONNX/tokenizer, web/ for JS assets, frontend-stub/ for icons
    filepath = _REPO_ROOT / "models" / "shabad-id-models" / filename
    if not filepath.exists():
        alt = _REPO_ROOT / "web" / filename
        if alt.exists():
            filepath = alt
    if not filepath.exists():
        alt2 = _REPO_ROOT / "desktop" / "frontend-stub" / filename
        if alt2.exists():
            filepath = alt2
    if not filepath.exists():
        return aiohttp.web.Response(text="Not found", status=404)
    # Short TTL on JS modules / JSON corpus so browser caches don't lock
    # users on a stale build. The ONNX model gets a long TTL (rarely
    # changes; rev the filename when swapping).
    if filename in {"v4.int8.onnx"}:
        cache = "public, max-age=86400"
    else:
        cache = "public, max-age=60, must-revalidate"
    headers = {"Cache-Control": cache}
    if not _is_ios_webkit_ua(request):
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        headers["Cross-Origin-Resource-Policy"] = "same-origin"
    headers["Vary"] = "User-Agent"
    return aiohttp.web.FileResponse(filepath, headers=headers)


# ─── /api/corpus ──────────────────────────────────────────────────────────
async def handle_corpus_api(request):
    """Serve edge-corpus.json from /api/corpus (hosted mode only).

    This endpoint serves the browser-side TF-IDF index used by the
    edge-inference JS worker. Desktop-only installs don't have this file.
    """
    filepath = _REPO_ROOT / "web" / "edge-corpus.json"
    if not filepath.exists():
        # Desktop-only installs don't have web/; this endpoint is for
        # the hosted edge-inference path.
        return aiohttp.web.json_response({"error": "edge-corpus.json not available"}, status=404)
    headers = {
        "Cache-Control": "public, max-age=60, must-revalidate",
        "CDN-Cache-Control": "no-store",
    }
    if not _is_ios_webkit_ua(request):
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return aiohttp.web.FileResponse(filepath, headers=headers)


# ─── /healthz ─────────────────────────────────────────────────────────────
async def handle_healthz(request):
    """Liveness probe. Returns {ok: true} if the engine is registered."""
    engine = request.app.get("engine")
    ok = (
        engine is not None
        and engine.corpus is not None
        and engine.asr is not None
    )
    return aiohttp.web.json_response({"ok": ok}, status=200 if ok else 503)


# ─── /api/google/* — service account credential management ───────────────
# End-customer flow: upload a service account JSON key (one-time setup)
# instead of installing gcloud CLI. The key is persisted to the sidecar's
# app-data dir via engine.google_auth (see that module's docstring for the
# why, the where, and the security posture).

async def handle_google_status(request):
    """GET /api/google/status

    Returns whether a service account key is currently saved and, if so,
    the safe-to-display metadata (service account email + GCP project_id).
    Never echoes the private key.
    """
    from .google_auth import credentials_status
    return aiohttp.web.json_response(credentials_status())


async def handle_google_upload(request):
    """POST /api/google/credentials

    Accepts a service account JSON key in the request body. The body can
    be either the raw JSON text or a parsed object — we handle both so
    the client can be small (`fetch(..., {body: fileText})` works).
    Validates structure (must be type=service_account with required
    fields) before persisting; rejects with 400 + an actionable error
    message otherwise.

    Note: this endpoint is reachable on localhost only — same threat
    model as the rest of the sidecar API. The uploaded key file lives
    under the user's per-user app-data dir with 0o600 perms on POSIX.
    """
    from .google_auth import save_credentials, InvalidServiceAccountKey
    try:
        # Tolerate either {"credentials": ...} envelope or a bare object.
        raw = await request.text()
        try:
            outer = json.loads(raw)
            if isinstance(outer, dict) and "credentials" in outer:
                data: Any = outer["credentials"]
            else:
                data = outer
        except json.JSONDecodeError:
            # Treat the entire body as the key file contents.
            data = raw
        result = save_credentials(data)
    except InvalidServiceAccountKey as e:
        return aiohttp.web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Failed to save Google credentials")
        return aiohttp.web.json_response({"error": f"Save failed: {e}"}, status=500)
    return aiohttp.web.json_response({"connected": True, **result})


async def handle_google_disconnect(request):
    """DELETE /api/google/credentials

    Removes the saved service account key. Active GoogleCloudASR backends
    will fail their *next* token refresh and surface the standard
    "credentials not configured" error to the WebSocket client — that's
    the same path as a fresh install. We deliberately don't try to flip
    every active session back to local ONNX here; the client already
    handles that by sending `set_google_asr` with an empty project_id
    when the user clicks Disconnect.
    """
    from .google_auth import clear_credentials
    removed = clear_credentials()
    return aiohttp.web.json_response({"connected": False, "removed": removed})


__all__ = [
    "handle_websocket_aiohttp",
    "handle_mic_page",
    "handle_mic_asset",
    "handle_corpus_api",
    "handle_healthz",
    "handle_google_status",
    "handle_google_upload",
    "handle_google_disconnect",
]
