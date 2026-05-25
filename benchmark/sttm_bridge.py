#!/usr/bin/env python3
"""
sttm_bridge.py — Bridge between the bani WS API and the benchmark's
STTM recorder.

Streams audio to the bani server over WebSocket, receives line_update
events, and forwards them as STTM Bani Controller events to the
sttm_recorder.py from the benchmark repo.

This is the "recommended" benchmark submission path: it proves that
a system which already drives STTM Desktop can produce benchmark
submissions with zero code changes — just point it at the recorder
instead of api.sikhitothemax.org.

Usage:
    # Terminal 1: start bani server
    python server.py --port 8765

    # Terminal 2: start the benchmark recorder for one GT case
    python sttm_recorder.py --video-id zOtIpxMT9hU --out submission/zOtIpxMT9hU.json --code bench --pin 1234

    # Terminal 3: run this bridge
    python sttm_bridge.py \
        --bani ws://localhost:8765/ws \
        --recorder http://localhost:5051 \
        --code bench --pin 1234 \
        --audio benchmark/audio/zOtIpxMT9hU_16k.wav

    # Score:
    python eval.py --pred submission/ --gt test/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


async def run_bridge(
    bani_url: str,
    recorder_url: str,
    code: str,
    pin: int,
    audio_path: str,
    chunk_seconds: float = 0.5,
    realtime: bool = True,
):
    import aiohttp
    import socketio

    # Load audio
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono
    if sr != 16000:
        raise ValueError(f"Expected 16kHz, got {sr}Hz. Resample first.")
    total_duration = len(audio) / sr
    print(f"[bridge] audio: {audio_path} ({total_duration:.1f}s)")

    chunk_samples = int(chunk_seconds * sr)

    # --- Connect to STTM recorder ---
    namespace = "/" + code.lstrip("/")
    sio = socketio.AsyncClient(logger=False, engineio_logger=False)

    recorder_ready = asyncio.Event()
    recorder_error = None

    @sio.on("data", namespace=namespace)
    async def on_recorder_data(data):
        nonlocal recorder_error
        if data.get("type") == "response-control":
            if data.get("success"):
                print(f"[bridge] recorder auth OK")
                recorder_ready.set()
            else:
                recorder_error = "recorder auth failed (wrong PIN?)"
                recorder_ready.set()

    print(f"[bridge] connecting to recorder at {recorder_url}{namespace}")
    await sio.connect(recorder_url, namespaces=[namespace])
    await sio.emit("data", {
        "host": "sttm-web",
        "type": "request-control",
        "pin": pin,
    }, namespace=namespace)

    await asyncio.wait_for(recorder_ready.wait(), timeout=5)
    if recorder_error:
        raise RuntimeError(recorder_error)

    # --- Connect to bani server ---
    print(f"[bridge] connecting to bani server at {bani_url}")
    events_received = 0
    locked_shabad = None

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(bani_url) as ws:
            # Read welcome
            msg = await asyncio.wait_for(ws.receive(), timeout=5)
            welcome = json.loads(msg.data)
            assert welcome["type"] == "connected"
            print(f"[bridge] bani connected, streaming audio...")

            # Background task: read events from bani and forward to recorder
            bani_done = asyncio.Event()

            async def event_reader():
                nonlocal events_received, locked_shabad
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    ev = json.loads(msg.data)
                    etype = ev.get("type")

                    if etype == "locked":
                        locked_shabad = ev.get("shabad_id")
                        verse_id = ev.get("verse_id")
                        audio_t = ev.get("duration_seconds", 0)
                        if locked_shabad and verse_id:
                            await sio.emit("data", {
                                "host": "sttm-web",
                                "type": "shabad",
                                "shabadId": locked_shabad,
                                "verseId": verse_id,
                                "lineCount": ev.get("total_lines", 0),
                                "audio_t": audio_t,
                                "pin": pin,
                            }, namespace=namespace)
                            events_received += 1
                            print(f"  [{audio_t:.1f}s] locked sid={locked_shabad} vid={verse_id}")

                    elif etype == "line_update":
                        shabad_id = ev.get("shabad_id") or locked_shabad
                        verse_id = ev.get("verse_id")
                        audio_t = ev.get("duration_seconds", 0)
                        if shabad_id and verse_id:
                            await sio.emit("data", {
                                "host": "sttm-web",
                                "type": "shabad",
                                "shabadId": shabad_id,
                                "verseId": verse_id,
                                "lineCount": ev.get("total_lines", 0),
                                "audio_t": audio_t,
                                "pin": pin,
                            }, namespace=namespace)
                            events_received += 1
                            line = ev.get("current_line", "?")
                            print(f"  [{audio_t:.1f}s] line {line} vid={verse_id}")

                    elif etype == "candidates":
                        top = ev.get("candidates", [{}])[0]
                        print(f"  [{ev.get('duration_seconds',0):.1f}s] candidates: "
                              f"top={top.get('name','?')[:30]}")

                    elif etype == "benchmark_done":
                        print(f"  [{ev.get('duration_seconds',0):.1f}s] benchmark_done")
                        bani_done.set()

            reader_task = asyncio.create_task(event_reader())

            try:
                # Stream audio in chunks. In real-time mode, pace to match
                # playback speed. In no-realtime mode, send audio in bursts
                # that match the identification interval (~5s of audio at a
                # time), yielding between bursts so the ASR loop can process
                # each window before the next arrives. This lets the server's
                # duration_seconds advance naturally without overwhelming the
                # ASR loop with 289s of buffered audio at once.
                offset = 0
                t0 = time.time()
                id_interval_samples = int(5.0 * sr)  # 5s = identification interval

                while offset < len(audio):
                    # Send one interval's worth of audio
                    burst_end = min(offset + id_interval_samples, len(audio))
                    while offset < burst_end:
                        chunk = audio[offset:offset + chunk_samples]
                        await ws.send_bytes(chunk.astype(np.float32).tobytes())
                        offset += chunk_samples

                    audio_time = offset / sr

                    if realtime:
                        elapsed = time.time() - t0
                        if audio_time > elapsed:
                            await asyncio.sleep(audio_time - elapsed)
                    else:
                        # Let the ASR loop process this burst before sending more.
                        # The server needs ~0.2-1s per identification tick.
                        await asyncio.sleep(0.5)

                    # Print progress
                    if int(audio_time) % 30 == 0 or offset >= len(audio):
                        print(f"  [{audio_time:.0f}/{total_duration:.0f}s] "
                              f"streamed, {events_received} events so far")

                # Wait for the server to finish processing final audio
                print(f"[bridge] audio done, waiting for final ASR ticks...")
                await ws.send_str(json.dumps({"command": "end"}))
                if realtime:
                    await asyncio.sleep(10)
                else:
                    try:
                        await asyncio.wait_for(bani_done.wait(), timeout=1800)
                    except asyncio.TimeoutError:
                        print("[bridge] WARNING: timed out waiting for benchmark_done")

                # Send bench-end to recorder
                await sio.emit("data", {
                    "host": "sttm-web",
                    "type": "bench-end",
                    "audio_t": total_duration,
                    "pin": pin,
                }, namespace=namespace)
                await asyncio.sleep(1)

            finally:
                reader_task.cancel()
                try:
                    await reader_task
                except asyncio.CancelledError:
                    pass

    await sio.disconnect()
    print(f"\n[bridge] done — {events_received} STTM events forwarded")


def main():
    p = argparse.ArgumentParser(description="Bridge bani WS API → STTM recorder")
    p.add_argument("--bani", default="ws://localhost:8765/ws",
                   help="bani server WebSocket URL")
    p.add_argument("--recorder", default="http://localhost:5051",
                   help="sttm_recorder.py HTTP URL")
    p.add_argument("--code", default="bench", help="STTM sync code")
    p.add_argument("--pin", type=int, default=1234, help="STTM PIN")
    p.add_argument("--audio", required=True, help="Path to 16kHz mono WAV")
    p.add_argument("--no-realtime", action="store_true",
                   help="Stream audio as fast as possible (not real-time)")
    args = p.parse_args()

    asyncio.run(run_bridge(
        bani_url=args.bani,
        recorder_url=args.recorder,
        code=args.code,
        pin=args.pin,
        audio_path=args.audio,
        realtime=not args.no_realtime,
    ))


if __name__ == "__main__":
    main()
