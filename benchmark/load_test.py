#!/usr/bin/env python3
"""Concurrent load-test harness for the engine WebSocket API.

Spins up N parallel "fake clients" that stream a real WAV in real time
and records per-session ASR latency, lock time, line accuracy, plus
process-wide CPU/memory. Use it to find the breakpoint where adding
another session degrades quality or saturates the box, so we can pick
the right hosting shape (dedicated CPU box vs Cloud Run per-instance).

Usage:
    # 1 session (baseline)
    python load_test.py --sessions 1 --audio benchmark/audio/IZOsmkdmmcg.wav

    # 10 concurrent sessions, all streaming the same audio in real time
    python load_test.py --sessions 10 --audio benchmark/audio/IZOsmkdmmcg.wav

    # Sweep: try 1, 2, 5, 10, 20 — emit a CSV for cost math
    python load_test.py --sweep 1,2,5,10,20 --audio benchmark/audio/IZOsmkdmmcg.wav --csv load_results.csv

Each fake client:
1. Connects to ws://<host>:<port>/ws
2. Sends `{"command": "lock", "shabad_id": <id>}` (oracle lock, skips ID phase)
3. Streams 2s PCM chunks paced in real time
4. Records timestamps for every line_update / tracking event
5. Reports ASR latency = server's "duration_seconds" delta vs wall clock

The harness samples psutil for the server process every 500 ms so we
can plot CPU/memory vs concurrency. The server PID is read from the
listening socket (Windows: Get-NetTCPConnection; Unix: lsof or /proc).
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import socket
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import websockets

try:
    import psutil
except ImportError:
    psutil = None  # CPU/mem sampling will be disabled


# ─── Per-session stats ─────────────────────────────────────────────────
@dataclass
class SessionStats:
    session_idx: int
    connected_at: float = 0.0
    locked_at: float = 0.0
    line_updates: int = 0
    asr_latencies_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    closed_at: float = 0.0

    @property
    def asr_p50_ms(self) -> float:
        return statistics.median(self.asr_latencies_ms) if self.asr_latencies_ms else 0.0

    @property
    def asr_p95_ms(self) -> float:
        if len(self.asr_latencies_ms) < 5:
            return max(self.asr_latencies_ms or [0.0])
        s = sorted(self.asr_latencies_ms)
        return s[int(0.95 * len(s))]


async def run_session(
    *,
    session_idx: int,
    ws_url: str,
    audio: np.ndarray,
    sample_rate: int,
    shabad_id: int,
    chunk_seconds: float = 2.0,
    realtime: bool = True,
) -> SessionStats:
    """Stream `audio` to the server in real time; record server events."""
    stats = SessionStats(session_idx=session_idx)
    chunk_samples = int(chunk_seconds * sample_rate)
    wall_start = time.time()
    last_event_wall: float = 0.0

    try:
        async with websockets.connect(ws_url, max_size=None) as ws:
            stats.connected_at = time.time() - wall_start

            # Oracle lock
            await ws.send(json.dumps({"command": "lock", "shabad_id": shabad_id}))

            async def receive_loop():
                nonlocal last_event_wall
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        t = msg.get("type")
                        now = time.time() - wall_start
                        if t == "locked":
                            stats.locked_at = now
                        elif t == "line_update":
                            stats.line_updates += 1
                            # ASR latency proxy: server-reported audio
                            # position vs our wall clock. Closer to 0 =
                            # fresh; higher = server falling behind.
                            srv_t = msg.get("duration_seconds", now)
                            lag_ms = max(0.0, (now - srv_t) * 1000.0)
                            stats.asr_latencies_ms.append(lag_ms)
                        last_event_wall = now
                except websockets.ConnectionClosed:
                    pass
                except Exception as e:
                    stats.errors.append(f"recv: {e!r}")

            recv_task = asyncio.create_task(receive_loop())

            # Pace audio in real time
            offset = 0
            while offset < len(audio):
                chunk = audio[offset:offset + chunk_samples]
                await ws.send(chunk.astype(np.float32).tobytes())
                offset += chunk_samples
                if realtime:
                    await asyncio.sleep(chunk_seconds)

            # Drain final events
            await asyncio.sleep(2.0)
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass
    except Exception as e:
        stats.errors.append(f"connect: {e!r}")
    finally:
        stats.closed_at = time.time() - wall_start

    return stats


# ─── Server process sampler ────────────────────────────────────────────
class ProcessSampler:
    """Sample CPU% / RSS MB of the server process every `interval_s`."""

    def __init__(self, pid: int, interval_s: float = 0.5):
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[tuple[float, float, float]] = []  # (t, cpu%, rss_mb)
        self._task: Optional[asyncio.Task] = None
        self._stop = False
        self._proc = psutil.Process(pid) if psutil else None

    async def _run(self) -> None:
        if not self._proc:
            return
        # Prime cpu_percent so the first reading isn't 0.0
        self._proc.cpu_percent(interval=None)
        t0 = time.time()
        while not self._stop:
            try:
                cpu = self._proc.cpu_percent(interval=None)
                rss = self._proc.memory_info().rss / (1024 * 1024)
                self.samples.append((time.time() - t0, cpu, rss))
            except psutil.NoSuchProcess:
                break
            await asyncio.sleep(self.interval_s)

    def start(self) -> None:
        if self._proc:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop = True
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"cpu_avg": 0.0, "cpu_max": 0.0, "rss_max_mb": 0.0}
        cpus = [c for _, c, _ in self.samples]
        rss = [r for _, _, r in self.samples]
        return {
            "cpu_avg": statistics.mean(cpus),
            "cpu_max": max(cpus),
            "rss_max_mb": max(rss),
        }


def _find_server_pid(host: str, port: int) -> Optional[int]:
    """Best-effort: find which PID owns the listening socket. Optional."""
    if not psutil:
        return None
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr and conn.laddr.port == port and conn.status == "LISTEN":
            return conn.pid
    return None


# ─── Sweep driver ──────────────────────────────────────────────────────
async def run_one(
    *,
    n_sessions: int,
    ws_url: str,
    audio_path: Path,
    shabad_id: int,
    realtime: bool,
    sampler: Optional[ProcessSampler],
    max_seconds: float = 0.0,
) -> dict:
    audio, sr = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if max_seconds > 0:
        audio = audio[: int(max_seconds * sr)]

    print(f"\n[load] {n_sessions} sessions, {audio_path.name} ({len(audio)/sr:.0f}s @ {sr}Hz)")
    if sampler:
        sampler.samples.clear()
        sampler._stop = False

    t0 = time.time()
    coros = [
        run_session(
            session_idx=i,
            ws_url=ws_url,
            audio=audio,
            sample_rate=sr,
            shabad_id=shabad_id,
            realtime=realtime,
        )
        for i in range(n_sessions)
    ]
    all_stats: list[SessionStats] = await asyncio.gather(*coros)
    elapsed = time.time() - t0

    # Aggregate
    all_lats = [l for s in all_stats for l in s.asr_latencies_ms]
    n_locked = sum(1 for s in all_stats if s.locked_at > 0)
    n_errors = sum(1 for s in all_stats if s.errors)
    line_updates = sum(s.line_updates for s in all_stats)

    proc = sampler.summary() if sampler else {}

    result = {
        "n_sessions": n_sessions,
        "elapsed_s": round(elapsed, 1),
        "locked_pct": round(100 * n_locked / max(1, n_sessions), 0),
        "errors": n_errors,
        "line_updates_total": line_updates,
        "asr_lag_p50_ms": round(statistics.median(all_lats), 0) if all_lats else 0,
        "asr_lag_p95_ms": round(sorted(all_lats)[int(0.95*len(all_lats))], 0) if len(all_lats) > 5 else 0,
        "cpu_avg_pct": round(proc.get("cpu_avg", 0.0), 1),
        "cpu_max_pct": round(proc.get("cpu_max", 0.0), 1),
        "rss_max_mb": round(proc.get("rss_max_mb", 0.0), 0),
    }

    print(
        f"  locked: {n_locked}/{n_sessions} "
        f"({result['locked_pct']:.0f}%); "
        f"line_updates total: {line_updates}; "
        f"errors: {n_errors}"
    )
    print(
        f"  ASR lag: p50={result['asr_lag_p50_ms']}ms "
        f"p95={result['asr_lag_p95_ms']}ms"
    )
    if proc:
        print(
            f"  server: CPU avg={result['cpu_avg_pct']}% max={result['cpu_max_pct']}% "
            f"RSS max={result['rss_max_mb']}MB"
        )

    for s in all_stats:
        if s.errors:
            print(f"  [s{s.session_idx} errors] {s.errors[:2]}")

    return result


async def main_async(args):
    ws_url = f"ws://{args.host}:{args.port}/ws"

    pid = _find_server_pid(args.host, args.port)
    if pid is None and psutil:
        print(f"[warn] could not find server PID on {args.host}:{args.port}; skipping CPU sampling")
    sampler = ProcessSampler(pid) if pid else None
    if sampler:
        print(f"[load] sampling server pid={pid}")
        sampler.start()

    sessions = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.sessions]
    results = []
    try:
        for n in sessions:
            r = await run_one(
                n_sessions=n,
                ws_url=ws_url,
                audio_path=Path(args.audio),
                shabad_id=args.shabad,
                realtime=not args.no_realtime,
                sampler=sampler,
                max_seconds=args.max_seconds,
            )
            results.append(r)
            # Give server a beat to release sockets between sweeps
            if n != sessions[-1]:
                await asyncio.sleep(3.0)
    finally:
        if sampler:
            await sampler.stop()

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'sessions':>9} {'locked%':>8} {'errors':>7} {'p50 ms':>7} {'p95 ms':>7} {'cpu%':>6} {'rss MB':>7}")
    for r in results:
        print(
            f"{r['n_sessions']:>9} "
            f"{r['locked_pct']:>7.0f}% "
            f"{r['errors']:>7} "
            f"{r['asr_lag_p50_ms']:>7.0f} "
            f"{r['asr_lag_p95_ms']:>7.0f} "
            f"{r['cpu_avg_pct']:>5.0f}% "
            f"{r['rss_max_mb']:>7.0f}"
        )
    print("=" * 78)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"[load] wrote {args.csv}")


def main():
    p = argparse.ArgumentParser(description="Engine WS load tester")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--audio", required=True, help="WAV file to stream from each fake client")
    p.add_argument("--shabad", type=int, default=4377, help="shabad_id for oracle lock")
    p.add_argument("--sessions", type=int, default=1, help="Concurrent sessions (single point)")
    p.add_argument("--sweep", default="", help="Comma-separated session counts, e.g. 1,2,5,10,20")
    p.add_argument("--no-realtime", action="store_true", help="Stream as fast as possible (also requires server ALLOW_AUDIO_BURST=1)")
    p.add_argument("--max-seconds", type=float, default=0.0, help="Clip each session's audio to first N seconds (0 = full)")
    p.add_argument("--csv", default="", help="Write results CSV to path")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
