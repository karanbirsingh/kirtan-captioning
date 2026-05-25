#!/usr/bin/env python3
"""Integration test: starts the server, connects via WebSocket, sends
synthetic audio, and verifies the full pipeline emits expected events.

Unlike regression_sttm.py (which tests matcher+SM in isolation), this
exercises the ENTIRE stack: server startup, WebSocket, audio ingestion,
ASR loop, session lifecycle, HTTP health check.

No real audio or ONNX model needed — we mock OnnxBackend to return
canned transcriptions so the test is fast and deterministic.

Usage:
    python tests/integration_server.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.asr import ASRBackend  # noqa: E402


class MockBackend(ASRBackend):
    """Fake ASR backend that returns canned Gurmukhi text."""

    def __init__(self):
        self._call_count = 0
        # Simulate progressing through shabad 1512 lines
        self._responses = [
            "ਚਰਨ ਕਮਲ ਕੀ ਆਸ ਪਿਆਰੇ",
            "ਚਰਨ ਕਮਲ ਕੀ ਆਸ ਪਿਆਰੇ ਜਮਕੰਕਰ ਨਸਿ ਗਏ ਵਿਚਾਰੇ",
            "ਚਰਨ ਕਮਲ ਕੀ ਆਸ ਪਿਆਰੇ ਜਮਕੰਕਰ ਨਸਿ ਗਏ ਵਿਚਾਰੇ",
            "ਜਮਕੰਕਰ ਨਸਿ ਗਏ ਵਿਚਾਰੇ ਤੂ ਚਿਤਿ ਆਵਹਿ ਤੇਰੀ ਮਇਆ",
            "ਸਿਮਰਤ ਨਾਮ ਸਗਲ ਰੋਗ ਖਇਆ",
            "ਦਰਸ ਤੇਰੇ ਕੀ ਪਿਆਸ ਮਨਿ ਲਾਗੀ",
        ]

    def transcribe(self, audio: np.ndarray) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    def extract_logprobs(self, audio: np.ndarray):
        return None  # no CTC

    def get_vocab(self):
        return None

    @property
    def supports_ctc(self):
        return False


async def run_test():
    import aiohttp
    from engine import (
        Config,
        DEFAULT_CORPUS_PATH,
        Engine,
        ShabadCorpus,
        ShabadMatcher,
    )
    from engine.routes import handle_healthz, handle_websocket_aiohttp

    # --- 1. Load corpus + matcher (real) ---
    print("[1/5] Loading corpus...", flush=True)
    corpus = ShabadCorpus(DEFAULT_CORPUS_PATH)
    corpus.load()
    matcher = ShabadMatcher(corpus)
    print(f"  {len(corpus.shabads)} shabads loaded", flush=True)

    # --- 2. Wire up with mock ASR backend ---
    print("[2/5] Creating mock ASR + server...", flush=True)
    mock_asr = MockBackend()

    # Build an Engine with the mock ASR. This is the new app["engine"]
    # pattern — no module globals, no monkey-patching.
    engine = Engine(
        corpus=corpus,
        asr=mock_asr,
        matcher=matcher,
        config=Config.from_env(),
    )

    # Create the app
    app = aiohttp.web.Application()
    app["engine"] = engine
    app.router.add_get("/ws", handle_websocket_aiohttp)
    app.router.add_get("/healthz", handle_healthz)

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    print(f"  Server on port {port}", flush=True)

    try:
        # --- 3. Test healthz ---
        print("[3/5] Testing /healthz...", flush=True)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://127.0.0.1:{port}/healthz") as resp:
                assert resp.status == 200, f"healthz returned {resp.status}"
                body = await resp.json()
                assert body["ok"] is True, f"healthz not ok: {body}"
                print(f"  healthz OK: {body}", flush=True)

        # --- 4. Connect WS, send audio, collect events ---
        print("[4/5] WebSocket test: connect + stream audio...", flush=True)
        events = []
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
                # Read welcome
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
                welcome = json.loads(msg.data)
                assert welcome["type"] == "connected", f"expected connected, got {welcome}"
                events.append(welcome)
                print(f"  connected: session={welcome.get('session_id')}", flush=True)

                # Send 30s of fake audio in chunks (triggers identification)
                # Each chunk: 4096 samples at 16kHz = 256ms
                chunk_samples = 4096
                total_chunks = int(30 * 16000 / chunk_samples)  # ~30s
                for i in range(total_chunks):
                    # Small random noise so silence gate doesn't reject
                    audio = np.random.randn(chunk_samples).astype(np.float32) * 0.05
                    await ws.send_bytes(audio.tobytes())
                    # Real-time pacing: 256ms per chunk but we send faster
                    # and let the burst limiter handle it
                    if i % 50 == 0:
                        await asyncio.sleep(0.01)  # yield to event loop

                # Wait for events to arrive
                # Give ASR loop time to fire + process
                deadline = time.time() + 15
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            ev = json.loads(msg.data)
                            events.append(ev)
                            etype = ev.get("type")
                            if etype == "candidates":
                                top = ev.get("candidates", [{}])[0]
                                print(f"  candidates: top={top.get('name', '?')[:40]}", flush=True)
                            elif etype == "status":
                                pass  # normal
                            else:
                                print(f"  {etype}: {json.dumps(ev, ensure_ascii=False)[:80]}", flush=True)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                    except asyncio.TimeoutError:
                        break

                # Send lock command
                await ws.send_str(json.dumps({
                    "command": "lock",
                    "shabad_id": 1512,
                    "start_line": 0,
                }))
                print("  sent lock command for shabad 1512", flush=True)

                # Collect lock + tracking events
                deadline = time.time() + 10
                while time.time() < deadline:
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            ev = json.loads(msg.data)
                            events.append(ev)
                            etype = ev.get("type")
                            if etype in ("locked", "line_update"):
                                print(f"  {etype}: {json.dumps(ev, ensure_ascii=False)[:80]}", flush=True)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                    except asyncio.TimeoutError:
                        break

                # Send reset
                await ws.send_str(json.dumps({"command": "reset"}))
                # Drain reset status event
                msg = await asyncio.wait_for(ws.receive(), timeout=5)

                # Ping
                await ws.send_str(json.dumps({"command": "ping"}))
                msg = await asyncio.wait_for(ws.receive(), timeout=5)
                pong = json.loads(msg.data)
                assert pong.get("type") == "pong", f"expected pong, got {pong}"
                print("  ping/pong OK", flush=True)

        # --- 5. Verify we got the expected event types ---
        print(f"\n[5/5] Verifying events ({len(events)} total)...", flush=True)
        types = {}
        for ev in events:
            t = ev.get("type", "?")
            types[t] = types.get(t, 0) + 1

        print(f"  Event types: {types}", flush=True)

        assert "connected" in types, "missing connected event"
        assert "locked" in types, "missing locked event"
        assert "line_update" in types, "missing line_update event"

        # Verify lock was for shabad 1512
        locked = [e for e in events if e.get("type") == "locked"]
        assert locked[0]["shabad_id"] == 1512, f"locked wrong shabad: {locked[0]}"

        print(f"\n{'='*50}")
        print("PASS: Integration test complete")
        print(f"  {len(events)} events, types: {types}")
        print(f"  Mock ASR called {mock_asr._call_count} times")
        print(f"{'='*50}")
        return True

    finally:
        await runner.cleanup()


def main():
    try:
        result = asyncio.run(run_test())
        return 0 if result else 1
    except Exception as e:
        print(f"\nFAIL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
