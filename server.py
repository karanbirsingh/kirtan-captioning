#!/usr/bin/env python3
"""
bani — Shabad identification and line tracking engine.

Usage:
    python server.py                        # API server on 0.0.0.0:8765
    python server.py --desktop              # Desktop sidecar (localhost, serves /mic/)
    python server.py --port 9000            # API server on custom port

WebSocket protocol (ws://<host>:<port>/ws):
    Client → Server: binary PCM frames (16kHz, mono, float32)
    Server → Client: JSON events (connected, candidates, locked, line_update, status)
    Client → Server: JSON commands {"command": "lock"|"reset"|"ping", ...}

Desktop mode is what the Tauri shell spawns. It prints
`BANI_READY port=<n>` on stdout, then serves the /mic/ UX on localhost.

API mode binds to 0.0.0.0 and exposes /ws, /healthz, /api/corpus.
Any WebSocket client can connect and stream audio for identification.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Engine lives at engine/ relative to repo root.
if hasattr(sys, "_MEIPASS"):
    REPO_ROOT = Path(sys._MEIPASS)  # type: ignore[attr-defined]
else:
    REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from engine import Config, build_engine  # noqa: E402
from engine.routes import (  # noqa: E402
    handle_corpus_api,
    handle_healthz,
    handle_mic_asset,
    handle_mic_page,
    handle_websocket_aiohttp,
)
from aiohttp import web  # noqa: E402

VERSION = "0.1.0"


# ─── Telemetry ───────────────────────────────────────────────────────────
def _default_telemetry_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "bani" / "telemetry.jsonl"


async def handle_telemetry(request: web.Request) -> web.Response:
    """Append telemetry beacon as one JSONL line (POST /api/t → 204)."""
    log_path = Path(os.environ.get("BANI_TELEMETRY_FILE") or _default_telemetry_path())
    try:
        body = await request.json()
        body["ts_recv"] = int(time.time() * 1000)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(body, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return web.Response(status=204)


async def handle_version(request: web.Request) -> web.Response:
    """GET /api/version — returns engine version and model info."""
    return web.json_response({
        "version": VERSION,
        "model": os.environ.get("BANI_ONNX_MODEL", "v4.int8"),
    })


# ─── App wiring ──────────────────────────────────────────────────────────
def _find_model() -> str:
    """Find the ONNX model, checking common locations. Auto-downloads from
    HuggingFace if not found locally."""
    candidates = [
        os.environ.get("BANI_ONNX_MODEL", ""),
        str(REPO_ROOT / "models" / "shabad-id-models" / "v4.int8.onnx"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            return path
    # Not found — auto-download
    print("[bani] model not found locally, downloading from HuggingFace...", flush=True)
    try:
        import subprocess
        subprocess.check_call(
            [sys.executable, str(REPO_ROOT / "scripts" / "download_models.py")],
            cwd=str(REPO_ROOT),
        )
        default = candidates[1]
        if Path(default).exists():
            return default
    except Exception as e:
        print(f"[bani] auto-download failed: {e}", flush=True)
    return candidates[1]  # return default path, will fail with clear error at load time


async def main() -> None:
    p = argparse.ArgumentParser(
        description="bani — shabad identification engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py                     Start API server on 0.0.0.0:8765
  python server.py --desktop           Desktop mode (localhost, /mic/ UI)
  python server.py --port 9000         Custom port
  python server.py --host 127.0.0.1    Localhost only
""",
    )
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("BANI_PORT", "0")),
                   help="Bind port (default: 0 = OS picks free port)")
    p.add_argument("--host", type=str, default=None,
                   help="Bind address (default: 127.0.0.1 for --desktop, 0.0.0.0 otherwise)")
    p.add_argument("--desktop", action="store_true",
                   help="Desktop mode: localhost, serve /mic/ HTML, print BANI_READY")
    p.add_argument("--onnx-model", type=str, default=None,
                   help="Path to ONNX ASR model (auto-detected if not set)")

    args = p.parse_args()

    # Desktop mode sets env vars for the engine
    if args.desktop:
        os.environ.setdefault("BANI_MODE", "desktop")

    host = args.host or ("127.0.0.1" if args.desktop else "0.0.0.0")
    model_path = args.onnx_model or _find_model()

    # Initialize the engine. onnx_path flows through Config so the
    # OnnxBackend is constructed with the right CPU/thread settings.
    print(f"[bani] initializing ASR model={model_path}", flush=True)
    cfg = Config.from_env(onnx_path=str(model_path))
    engine = await build_engine(config=cfg)
    print("[bani] engine ready", flush=True)

    app = web.Application()
    app["engine"] = engine

    # Core API — always available
    app.router.add_get("/ws", handle_websocket_aiohttp)
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/api/version", handle_version)
    app.router.add_get("/api/corpus", handle_corpus_api)
    app.router.add_post("/api/t", handle_telemetry)

    # Desktop mode: also serve the /mic/ UX
    if args.desktop:
        app.router.add_get("/mic", handle_mic_page)
        app.router.add_get("/mic/", handle_mic_page)
        app.router.add_get("/mic/{filename}", handle_mic_asset)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, args.port)
    await site.start()

    # Get actual bound port (handles port=0 case)
    sock = site._server.sockets[0]  # type: ignore[union-attr]
    actual_port = sock.getsockname()[1]

    if args.desktop:
        # Tauri shell parses this exact line — don't change format without
        # updating desktop/src-tauri/src/main.rs.
        print(f"BANI_READY port={actual_port}", flush=True)
        print(f"[bani] open http://127.0.0.1:{actual_port}/mic/", flush=True)
    else:
        print(f"[bani] listening on {host}:{actual_port}", flush=True)
        print(f"[bani] WebSocket: ws://{host}:{actual_port}/ws", flush=True)
        print(f"[bani] Health:    http://{host}:{actual_port}/healthz", flush=True)

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
        print("[bani] stopped", flush=True)
