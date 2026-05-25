# Benchmark Results — Gurbani Captioning v0.2.0

Reproducible results from this repo against the [live-gurbani-captioning-benchmark-v1](https://karanbirsingh.github.io/live-gurbani-captioning-benchmark-v1/) suite.

## What's in this directory

- `README.md` (this file) — reproduction recipe + scores
- `results/*.json` — the actual prediction outputs from the run reported below.
  These are the **exact files** that produced the numbers in the table; you can
  re-score them yourself with `eval.py` (instructions below) or open them in the
  [annotation viewer](https://karanbirsingh.github.io/live-gurbani-captioning-benchmark-v1/annotate.html).

## Headline numbers

- **Overall frame accuracy: 57.9%** (1982/3425 frames, 1s collar)
- **Shabad identification: 10/12 = 83%** correct locks
- Commit: `3b42245` (initial 0.2.0 release squash)
- Model: `v4.int8.onnx` (auto-downloaded from HuggingFace by `scripts/download_models.py`)
- Run date: 2026-05-24, macOS arm64 (CPUExecutionProvider only — CoreMLExecutionProvider
  silently produces `<unk>` on the int8 model, see [engine/_internal/onnx_inference.py](../engine/_internal/onnx_inference.py))

| Case | Accuracy | Frames | Locked? |
|------|----------|--------|---------|
| `kchMJPK9Axs` | 82.0% | 532/649 | ✓ S1341 @ 23s |
| `kchMJPK9Axs_cold33` | 80.1% | 351/438 | ✓ S1341 @ 259s |
| `kchMJPK9Axs_cold66` | 73.0% | 162/222 | ✓ S1341 @ 475s |
| `IZOsmkdmmcg` | 74.1% | 337/455 | ✓ S4377 @ 18s |
| `IZOsmkdmmcg_cold33` | 10.7% | 33/308 | ✗ locked S3643 (wrong) |
| `IZOsmkdmmcg_cold66` | 68.0% | 106/156 | ✓ S4377 @ 322s |
| `kZhIA8P6xWI` | 58.1% | 176/303 | ✓ S1821 @ 33s |
| `kZhIA8P6xWI_cold33` | 3.9% | 8/207 | ✗ locked S24 (wrong) |
| `kZhIA8P6xWI_cold66` | 44.8% | 47/105 | ✓ S1821 @ 242s |
| `zOtIpxMT9hU` | 30.3% | 87/287 | ✓ S3712 @ 188s |
| `zOtIpxMT9hU_cold33` | 49.0% | 96/196 | ✓ S3712 @ 174s |
| `zOtIpxMT9hU_cold66` | 47.5% | 47/99 | ✓ S3712 @ 216s |
| **Overall** | **57.9%** | **1982/3425** | **10/12** |

The two failures are both `_cold33` cases — dropped into the middle of a track
33% of the way in, with no audio history. This remains the hardest variant.

For comparison, the benchmark's [reference baselines](https://karanbirsingh.github.io/live-gurbani-captioning-benchmark-v1/):

| Baseline | Overall |
|---|---|
| `empty` (silent predictions) | 26.0% |
| `shifted_5s` (GT delayed 5s) | 85.5% |
| `perfect` (exact GT copy) | 100.0% |

## How to reproduce

You'll need:
- This repo at commit `3b42245` or later (`git rev-parse HEAD` should match if running on `main`)
- Python 3.11+ with `requirements.txt` installed
- Benchmark `.wav` files in some directory (the upstream benchmark page hosts them)
- ~30 seconds of CPU time on Apple Silicon (longer on Intel / WASM)

```bash
# 1. Install deps + auto-download the ONNX model (one-time, ~180 MB)
pip install -r requirements.txt
pip install soundfile websockets   # benchmark-client only
python scripts/download_models.py

# 2. Start the server in benchmark mode.
#    ALLOW_AUDIO_BURST=true does two things:
#      (a) disables wall-clock audio pacing so the client can stream as
#          fast as the server can ASR
#      (b) implies auto_lock_enabled=true (benchmark has no human to click
#          "Confirm" on candidates)
ALLOW_AUDIO_BURST=true python server.py
# → server prints `ws://0.0.0.0:8765/ws`

# 3. In another terminal, run the benchmark client.
python benchmark/benchmark_client.py \
  --api ws://127.0.0.1:8765/ws \
  --gt benchmark/test/ \
  --audio-dir /path/to/benchmark/audio/ \
  --output benchmark/results/ \
  --no-realtime

# 4. Score against ground truth
python benchmark/eval.py \
  --gt benchmark/test \
  --pred benchmark/results
```

## How to score just the JSONs in this directory

You don't need to re-run the server. The `results/*.json` files in this directory
ARE the predictions from step 3 above. To score them directly:

```bash
python benchmark/eval.py --gt benchmark/test --pred benchmark/results
```

You should get the same Overall = 57.9% number as the table above. If you see
something different, either the ground-truth `test/*.json` has been updated
(matcher-impacting) or the scorer's collar / framing logic changed.

## Per-case visualization

To inspect a single case (where exactly it locked, which lines it predicted
vs ground truth, second-by-second):

```bash
# Upload your prediction JSON to the public annotation viewer:
open "https://karanbirsingh.github.io/live-gurbani-captioning-benchmark-v1/annotate.html?case=kchMJPK9Axs"
```

Then drag-drop the corresponding `results/<case>.json` into the viewer.

## Tuning knobs that affect these numbers

Anything in `engine/config.py` will change the result. The most impactful:

| Env var | Default | What it does |
|---|---|---|
| `BANI_AUTO_LOCK` | `false` (desktop), `true` (benchmark) | Auto-confirm top candidate |
| `IDENTIFICATION_INTERVAL` | `5.0` | Seconds between candidate refreshes |
| `MAX_ID_SECONDS` | `30` | Tail of audio fed to ID-phase ASR |
| `TRACKING_WINDOW` | `15.0` | Sliding window for line tracking |
| `LOCK_CONFIDENCE_THRESHOLD` | `0.8` | Score gap required to auto-lock |
| `ONNX_FORCE_CPU` | `true` (macOS), `false` (Windows) | Skip CoreML / DirectML providers |

To re-score with a different setting, just set the env var when starting the
server in step 2 above.

## Known issues / caveats

- **CoreMLExecutionProvider produces `<unk>` on this int8 model on macOS.**
  Disabled by default in `engine/_internal/onnx_inference.py`. CPU on
  Apple Silicon is fast enough (~30× realtime) that this isn't user-visible.
- **The two `_cold33` failures** are a real weakness — when dropped into the
  middle of a track, the matcher's first 30s of audio doesn't have enough
  unique tokens to distinguish similar shabads. Filed as a known limitation.
- **No noise / instrumental robustness testing yet.** All 12 cases are
  clean Shabad recordings with minimal background.
