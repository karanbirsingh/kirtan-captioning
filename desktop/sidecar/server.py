#!/usr/bin/env python3
"""
desktop/sidecar/server.py — minimal localhost server for the desktop app.

Serves the same /mic/ UX as the hosted version but runs ASR natively (5-30×
faster than browser WASM on Windows) and writes telemetry to a local JSONL
file instead of fly logs.

Architecture:
    Tauri shell (later)
       │  spawns this process, parses "BANI_READY port=<n>" from stdout
       ▼
    aiohttp server on 127.0.0.1:<random_port>
       │  serves desktop/frontend-stub/mic.html
       ▼
    Browser-class WebView (Edge WebView2 inside Tauri)
       │  loads localhost:<port>/mic/, opens mic, ticks transcribe, etc.
       ▼
    /ws websocket → handle_websocket_aiohttp (engine/routes.py)
       │  drives MatcherStateMachine (matcher_state.py — canonical)
       ▼
    Native ONNX Runtime decodes audio windows in ~10× the speed of
    onnxruntime-web WASM. Same v4.int8 model, same SM, same UX.

What's NOT here vs. the hosted server:
    × GitHub OAuth admin
    × Radio feeders
    × Cross-device peer-confirm
    × Server-side telemetry (replaced with local JSONL)

The 7 routes registered below are the exact ones the /mic/ UX consumes —
the desktop subset of the full server. Handlers are imported from
engine/routes.py so we inherit any bug fixes / feature work that lands there.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# engine/ holds the core package (asr, matcher, session, etc.).
# PyInstaller bundles engine/ + data/ next to this script under
# sys._MEIPASS (the extraction dir for --onefile, the _internal/ dir for
# --onedir). In dev we walk up from this file: desktop/sidecar/server.py
# -> repo root. Either way, REPO_ROOT contains data/, engine/ at
# the same relative paths.
if hasattr(sys, "_MEIPASS"):
    REPO_ROOT = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Tell the engine to inject <meta name="bani-mode" content="desktop">
# into the served HTML (desktop/frontend-stub/mic.html). The page reads
# this meta tag and skips onnxruntime-web + JS matcher, opening a WebSocket
# to /ws so ASR + matching run natively in this sidecar instead.
os.environ.setdefault("BANI_MODE", "desktop")

import logging  # noqa: E402

# ─── File logging for installed app ──────────────────────────────────────
# Tauri swallows the sidecar's stderr, so Python logging is invisible on the
# installed app. Route it to a log file next to the telemetry JSONL.
def _setup_file_logging() -> None:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    log_dir = base / "bani-mic"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "sidecar.log"
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

import logging.handlers  # noqa: E402
_setup_file_logging()

from engine import Config, build_engine  # noqa: E402
from engine.routes import (  # noqa: E402
    handle_corpus_api,
    handle_google_disconnect,
    handle_google_status,
    handle_google_upload,
    handle_healthz,
    handle_mic_asset,
    handle_mic_page,
    handle_websocket_aiohttp,
)
from aiohttp import web  # noqa: E402


# ─── Local telemetry sink ────────────────────────────────────────────────
# Production /api/t logs to stdout (captured by `fly logs`). Desktop has
# no fly logs, so we append JSON Lines to a user-writable path:
#   Windows: %LOCALAPPDATA%\bani-mic\telemetry.jsonl
#   Mac:     ~/Library/Application Support/bani-mic/telemetry.jsonl
#   Linux:   ~/.local/share/bani-mic/telemetry.jsonl
# Tauri sets BANI_TELEMETRY_FILE to the absolute path on launch. In dev
# (running this script directly) we default to ./telemetry.jsonl so the
# file shows up next to the working dir, easy to tail.
def _default_telemetry_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "bani-mic" / "telemetry.jsonl"


async def handle_telemetry_local(request: web.Request) -> web.Response:
    """Append telemetry beacon as one JSONL line. Mirrors the prod handler's
    contract (POST /api/t, returns 204) so the WebView's _t() doesn't notice
    a difference."""
    log_path = Path(os.environ.get("BANI_TELEMETRY_FILE") or _default_telemetry_path())
    try:
        body = await request.json()
        body["ts_recv"] = int(time.time() * 1000)
        body["src"] = "desktop"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(body, ensure_ascii=False) + "\n")
    except Exception:
        # Never let telemetry errors break the request — same posture as prod.
        pass
    return web.Response(status=204)


# ─── App wiring ──────────────────────────────────────────────────────────
async def main() -> None:
    # Parent-death watchdog: the Tauri shell passes its own PID via BANI_SHELL_PID.
    # We poll every 2s; if that PID no longer exists, we self-exit.
    shell_pid_str = os.environ.get("BANI_SHELL_PID")
    if shell_pid_str and shell_pid_str.isdigit():
        shell_pid = int(shell_pid_str)
        def _is_shell_alive(pid: int) -> bool:
            """Check if a PID is still alive (cross-platform)."""
            if sys.platform == "win32":
                # On Windows, os.kill(pid, 0) calls TerminateProcess — NOT safe.
                # Use ctypes OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION.
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return True
                return False
            else:
                try:
                    os.kill(pid, 0)
                    return True
                except ProcessLookupError:
                    return False
                except PermissionError:
                    return True  # exists but can't signal
        async def _watchdog() -> None:
            while True:
                await asyncio.sleep(2)
                if not _is_shell_alive(shell_pid):
                    print(f"[sidecar] shell pid {shell_pid} is gone, exiting", flush=True)
                    os._exit(0)
        asyncio.create_task(_watchdog())

    p = argparse.ArgumentParser(description="bani-mic desktop sidecar")
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("BANI_SIDECAR_PORT", "0")),
        help="Bind port (0 = let OS pick a free one). Final port is printed "
             "as 'BANI_READY port=<n>' on stdout for the Tauri shell to parse.",
    )
    p.add_argument(
        "--onnx-model",
        type=str,
        default=os.environ.get("BANI_ONNX_MODEL")
                or str(REPO_ROOT / "models" / "shabad-id-models" / "v4.int8.onnx"),
        help="Path to ONNX ASR model. Defaults to models/shabad-id-models/v4.int8.onnx "
             "repo (dev) or the bundled model in production builds.",
    )
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address. Always loopback in desktop mode — never expose "
             "this server externally; it has no auth.",
    )
    args = p.parse_args()

    # Warm up the ASR engine + corpus before binding the port. If init fails,
    # we want the Tauri shell to see a non-zero exit instead of a half-up
    # server that 500s on first /ws/mic.
    print(f"[sidecar] initializing ASR model={args.onnx_model}", flush=True)
    cfg = Config.from_env(onnx_path=str(args.onnx_model))
    engine = await build_engine(config=cfg)
    print("[sidecar] ASR ready", flush=True)

    app = web.Application()
    app["engine"] = engine
    # The 7 routes the desktop /mic/ UX consumes.
    app.router.add_get("/mic", handle_mic_page)
    app.router.add_get("/mic/", handle_mic_page)
    app.router.add_get("/mic/{filename}", handle_mic_asset)
    app.router.add_get("/api/corpus", handle_corpus_api)  # unused in
    # desktop mode (the page skips /api/corpus when bani-mode=desktop) but
    # kept for parity / debugging — `curl /api/corpus` is a quick way to
    # confirm the corpus loaded inside the sidecar.
    app.router.add_post("/api/t", handle_telemetry_local)
    app.router.add_get("/healthz", handle_healthz)
    # Google Cloud service account credential management (production
    # customer flow — see engine/google_auth.py for the storage policy).
    app.router.add_get("/api/google/status", handle_google_status)
    app.router.add_post("/api/google/credentials", handle_google_upload)
    app.router.add_delete("/api/google/credentials", handle_google_disconnect)
    app.router.add_get("/ws", handle_websocket_aiohttp)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()

    # Pull the actual bound port out of the underlying socket so port=0
    # (OS-assigned) flows back to the Tauri shell. site._server is private
    # but stable across aiohttp 3.x; if upstream renames it we'll see a
    # crash on first launch (loud failure, easy to fix).
    sock = site._server.sockets[0]  # type: ignore[union-attr]
    actual_port = sock.getsockname()[1]
    # ⚠️  This exact line format is parsed by the Tauri shell's main.rs.
    # Don't change it without updating both sides.
    print(f"BANI_READY port={actual_port}", flush=True)
    print(f"[sidecar] open http://127.0.0.1:{actual_port}/mic/ in a browser to verify", flush=True)

    # Run until killed. Tauri sends SIGTERM on app close; in dev, Ctrl-C.
    try:
        await asyncio.Future()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[sidecar] stopped", flush=True)
