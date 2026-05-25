#!/usr/bin/env python3
"""
Reference benchmark client — streams audio to our WebSocket API and
collects line predictions in the submission format.

This is our integration harness. You do not need it to submit results:
it only exists because our reference implementation happens to be a
streaming server. Any system that produces prediction JSON files in the
documented submission format can be scored directly with `eval.py`.

Protocol (reference server only):
  Client → Server: binary PCM frames (16kHz, mono, float32)
  Server → Client: JSON messages (connected, candidates, locked, line_update, tracking)

The client is protocol-agnostic over these messages: it streams audio and
records whatever line transitions the server reports. The server handles
shabad identification and line tracking internally.

Usage:
    # Blind mode (server identifies shabad automatically)
    python benchmark_client.py --api ws://localhost:8765/ws --gt test/ --audio-dir audio/

    # Oracle mode (tell server which shabad via lock command)
    python benchmark_client.py --api ws://localhost:8765/ws --gt test/ --audio-dir audio/ --oracle

    # Fast mode (don't wait real-time between chunks)
    python benchmark_client.py --api ws://localhost:8765/ws --gt test/ --audio-dir audio/ --no-realtime

    # Score results
    python eval.py --pred predictions/ --gt test/
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


async def stream_and_collect(
    ws_url: str,
    audio_path: str,
    shabad_id: int = None,
    oracle: bool = False,
    chunk_seconds: float = 2.0,
    realtime: bool = True,
    audio_offset: float = 0.0,
) -> dict:
    """
    Stream audio to a WebSocket API and collect line predictions.

    Args:
        ws_url: WebSocket URL
        audio_path: Path to 16kHz mono WAV
        shabad_id: Shabad ID (only used in oracle mode)
        oracle: If True, send lock command with shabad_id before streaming
        chunk_seconds: Audio chunk size in seconds
        realtime: If True, pace audio at real-time speed

    Returns dict with:
        segments: [{"start": float, "end": float, "line_idx": int}, ...]
        locked_shabad_id: int or None (what the server locked onto)
        lock_time: float or None (seconds into audio when lock happened)
    """
    import websockets

    audio, sr = sf.read(audio_path)
    if sr != 16000:
        raise ValueError(f"Audio must be 16kHz, got {sr}")

    # Cold-start: skip into audio to simulate joining mid-shabad
    if audio_offset > 0:
        skip_samples = int(audio_offset * sr)
        audio = audio[skip_samples:]

    total_duration = len(audio) / sr + audio_offset
    chunk_samples = int(chunk_seconds * sr)

    segments = []
    current_line = None
    current_start = None
    locked_shabad_id = None
    lock_time = None

    def handle_event(data: dict, event_time: float) -> bool:
        """Process one server event. Returns True when offline processing is done."""
        nonlocal current_line, current_start, locked_shabad_id, lock_time

        msg_type = data.get("type", "")

        if msg_type == "locked":
            # Server identified and locked a shabad
            locked_shabad_id = data.get("shabad_id")
            lock_time = event_time

        elif msg_type in ("line_update", "tracking"):
            # Server reports current line (1-indexed from server)
            new_line = data.get("current_line", 0) - 1  # Convert to 0-indexed

            # Close previous segment
            if current_line is not None and new_line != current_line:
                segments.append({
                    "start": round(current_start, 1),
                    "end": round(event_time, 1),
                    "line_idx": current_line,
                })

            if new_line != current_line:
                current_line = new_line
                current_start = event_time

        elif msg_type == "benchmark_done":
            return True

        return False

    async with websockets.connect(ws_url) as ws:
        # Wait for connected message
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)
            if data.get("type") == "connected":
                pass  # Good
        except asyncio.TimeoutError:
            print("WARNING: No connected message from server")

        # Oracle mode: tell server which shabad to track
        if oracle and shabad_id is not None:
            await ws.send(json.dumps({
                "command": "lock",
                "shabad_id": shabad_id,
            }))
            # Wait for lock confirmation
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                if data.get("type") == "locked":
                    locked_shabad_id = data.get("shabad_id")
                    lock_time = 0.0
            except asyncio.TimeoutError:
                print("WARNING: No lock confirmation from server")

        # Stream audio in chunks
        offset = 0
        while offset < len(audio):
            chunk = audio[offset:offset + chunk_samples]

            # Send as raw PCM bytes
            await ws.send(chunk.astype(np.float32).tobytes())

            audio_time = (offset + len(chunk)) / sr + audio_offset

            # Collect any responses (non-blocking)
            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    data = json.loads(msg)
                    # Use server-reported audio time when available (critical for --no-realtime)
                    event_time = (data.get("duration_seconds", audio_time - audio_offset) + audio_offset) if not realtime else audio_time
                    handle_event(data, event_time)

            except asyncio.TimeoutError:
                pass

            offset += chunk_samples

            # Simulate real-time playback speed
            if realtime:
                await asyncio.sleep(chunk_seconds)

        # Send end signal
        try:
            await ws.send(json.dumps({"command": "end"}))
        except Exception:
            pass

        # Collect final responses. In --no-realtime mode the server may still
        # be walking the virtual timeline after all audio bytes are sent, so
        # wait for its explicit completion event instead of timing out after a
        # fixed 3s gap.
        done = False
        final_timeout = 60.0 if not realtime else 3.0
        max_wait = 1800.0 if not realtime else 10.0
        final_start = time.time()
        try:
            while not done and (time.time() - final_start) < max_wait:
                msg = await asyncio.wait_for(ws.recv(), timeout=final_timeout)
                data = json.loads(msg)
                event_time = (data.get("duration_seconds", total_duration - audio_offset) + audio_offset) if not realtime else total_duration
                done = handle_event(data, event_time)

        except (asyncio.TimeoutError, Exception):
            pass

        # Close final segment
        if current_line is not None:
            segments.append({
                "start": round(current_start, 1),
                "end": round(total_duration, 1),
                "line_idx": current_line,
            })

    return {
        "segments": segments,
        "locked_shabad_id": locked_shabad_id,
        "lock_time": lock_time,
    }


async def run_benchmark(
    api_url: str,
    gt_dir: Path,
    audio_dir: Path,
    output_dir: Path,
    oracle: bool = False,
    realtime: bool = True,
):
    """Run benchmark against an API for all test videos."""

    gt_files = sorted(gt_dir.glob("*.json"))

    if not gt_files:
        print(f"ERROR: No GT files in {gt_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "oracle" if oracle else "blind"
    print(f"Mode: {mode} | API: {api_url} | GT files: {len(gt_files)}")

    total_correct_shabad = 0
    total_videos = 0

    for gt_file in gt_files:
        with open(gt_file, encoding="utf-8") as f:
            gt = json.load(f)

        video_id = gt["video_id"]
        shabad_id = gt["shabad_id"]
        cold_start = gt.get("cold_start", 0.0)

        # Find audio file
        audio_path = None
        for ext in [".wav", ".mp3", ".flac"]:
            candidate = audio_dir / f"{video_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break

        if not audio_path:
            print(f"  SKIP {video_id}: no audio file in {audio_dir}")
            continue

        cold_label = f" cold@{cold_start:.0f}s" if cold_start > 0 else ""
        print(f"  {gt_file.stem} (shabad {shabad_id}{cold_label})...", end=" ", flush=True)

        t0 = time.time()
        result = await stream_and_collect(
            ws_url=api_url,
            audio_path=str(audio_path),
            shabad_id=shabad_id if oracle else None,
            oracle=oracle,
            realtime=realtime,
            audio_offset=cold_start,
        )
        elapsed = time.time() - t0

        segments = result["segments"]
        locked_id = result["locked_shabad_id"]
        lock_t = result["lock_time"]

        # Check shabad identification (compare as int to handle str/int mismatch)
        total_videos += 1
        shabad_correct = (int(locked_id) == int(shabad_id)) if locked_id is not None else False
        if shabad_correct:
            total_correct_shabad += 1

        lock_info = ""
        if not oracle:
            if locked_id is not None:
                lock_info = f"locked S{locked_id} at {lock_t:.0f}s"
                if not shabad_correct:
                    lock_info += f" (WRONG, expected S{shabad_id})"
                else:
                    lock_info += " ✓"
            else:
                lock_info = "NEVER LOCKED"

        # Save predictions
        pred = {
            "video_id": video_id,
            "shabad_id": locked_id,
            "segments": segments,
            "lock_time": lock_t,
            "mode": mode,
        }
        pred_path = output_dir / f"{gt_file.stem}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pred, f, indent=2)

        print(f"{len(segments)} segs, {elapsed:.1f}s | {lock_info}")

    # Summary
    if not oracle:
        print(f"\nShabad ID accuracy: {total_correct_shabad}/{total_videos} "
              f"({total_correct_shabad/total_videos*100:.0f}%)")

    print(f"Predictions saved to {output_dir}/")
    print(f"Score: python benchmark/eval.py --pred {output_dir} --gt {gt_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Reference streaming client for the kirtan captioning benchmark"
    )
    parser.add_argument("--api", required=True, help="WebSocket URL (e.g. ws://localhost:8765/ws)")
    parser.add_argument("--gt", required=True, help="Ground truth directory")
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files (video_id.wav)")
    parser.add_argument("--output", default="benchmark/predictions", help="Output directory for predictions")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle mode: send shabad_id to server (skip identification)")
    parser.add_argument("--no-realtime", action="store_true",
                        help="Stream audio as fast as possible (not real-time)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        api_url=args.api,
        gt_dir=Path(args.gt),
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output),
        oracle=args.oracle,
        realtime=not args.no_realtime,
    ))


if __name__ == "__main__":
    main()
