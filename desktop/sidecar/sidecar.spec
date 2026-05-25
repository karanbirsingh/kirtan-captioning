# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the desktop sidecar.

What this builds:
    dist/gurbani-captioning              (single onefile executable, ~250 MB)
        ↳ on launch, extracts to a temp dir (sys._MEIPASS) and serves
          the same /mic/ UX from there. ~3-5s cold start (acceptable for
          a daily-launch desktop app); subsequent ASR is real-time.

The Tauri shell (added in step 6) spawns `gurbani-captioning`, parses
`BANI_READY port=<n>` from stdout, and points its WebView at
http://127.0.0.1:<port>/mic/. Same UX as the hosted version but native
ONNX Runtime (5-30× faster than browser WASM on Windows).

Why onefile (vs --onedir): one executable is simpler to ship via Tauri's
externalBin, simpler to attach to a GH Releases artifact, simpler for
the user (no `_internal/` folder they might accidentally delete or
quarantine separately from the binary). Cold start is the only cost,
and it amortizes to zero after first launch since macOS keeps the temp
extraction warm in the page cache.

Build:
    cd desktop/sidecar
    pyinstaller --clean --noconfirm sidecar.spec

Smoke the bundle:
    ./dist/gurbani-captioning > /tmp/sidecar.log 2>&1 &
    python ../../desktop/smoke_ws.py

Size budget (Windows, May 2026): ~220 MB onefile (compressed). ~180 MB is the
ONNX model, ~23 MB is sggs_corpus.json, the rest is onnxruntime CPU+DirectML +
numpy + Python runtime. CUDA/TensorRT provider DLLs (~400 MB) are stripped;
DirectML (~10 MB) is kept for GPU acceleration on any Windows GPU.
"""
import subprocess
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# --- Paths -----------------------------------------------------------------
# This file lives at desktop/sidecar/sidecar.spec, so two parents up = repo
# root. PyInstaller's spec runner sets __file__ to the spec path on 6.x.
SPEC_DIR = Path(SPECPATH).resolve()      # noqa: F821 (PyInstaller injects)
REPO_ROOT = SPEC_DIR.parent.parent

ENTRY = SPEC_DIR / "server.py"
ENGINE_DIR = REPO_ROOT / "engine"
FRONTEND_DIR = REPO_ROOT / "desktop" / "frontend-stub"
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models" / "shabad-id-models"

# --- Bootstrap: pull model files from HuggingFace if missing ---------------
# The ~180 MB ONNX checkpoint and tokenizer aren't tracked in git (too big,
# and they belong on a model registry anyway). On a fresh clone they get
# materialized into models/shabad-id-models/ by scripts/download_models.py
# from karansea/indicconformer-stt-pa-ctc-shabad-preview on HF. We invoke
# that script automatically here so `pyinstaller sidecar.spec` Just Works
# instead of failing 30 seconds in with a confusing "FileNotFoundError"
# from inside Analysis().
_REQUIRED_MODEL_FILES = ["v4.int8.onnx", "v4_tokenizer.model"]
if not all((MODELS_DIR / f).exists() for f in _REQUIRED_MODEL_FILES):
    print(
        f"[sidecar.spec] model files missing under {MODELS_DIR}, "
        f"running scripts/download_models.py ...",
        flush=True,
    )
    subprocess.check_call(
        [sys.executable, str(REPO_ROOT / "scripts" / "download_models.py")]
    )
    missing = [f for f in _REQUIRED_MODEL_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        raise SystemExit(
            f"[sidecar.spec] download_models.py finished but these are still "
            f"missing: {missing}. Run `python scripts/download_models.py` "
            f"manually to debug."
        )

# --- Frontend HTML + model files -------------------------------------------
# Desktop mode only needs mic.html (served at /mic/) and the ONNX model.
# The edge-inference JS assets (edge-worker.js, etc.) are NOT needed in
# desktop mode — the page detects bani-mode=desktop and uses WebSocket
# instead of in-browser WASM inference.

datas = []

# mic.html
mic_html = FRONTEND_DIR / "mic.html"
if mic_html.exists():
    datas.append((str(mic_html), "desktop/frontend-stub"))

# Model files: bundle from MODELS_DIR (where download_models.py puts them)
# into "models/shabad-id-models/" inside the bundle so server.py's default
# model path resolves cleanly at runtime.
_MIN_MODEL_BYTES = 1024  # tokenizer is ~240 kB, ONNX is ~180 MB; 1 kB rules out stubs
for local_name in ("v4.int8.onnx", "v4_tokenizer.model"):
    src = MODELS_DIR / local_name
    if not src.exists():
        raise SystemExit(
            f"[sidecar.spec] missing {local_name} under {MODELS_DIR}. "
            f"Run `python scripts/download_models.py` first."
        )
    real = src.resolve()
    if not real.exists() or real.stat().st_size < _MIN_MODEL_BYTES:
        raise SystemExit(
            f"[sidecar.spec] {local_name} resolves to {real} which is "
            f"missing or only {real.stat().st_size if real.exists() else 0} "
            f"bytes — looks like a broken symlink stub. Rerun "
            f"`python scripts/download_models.py` to materialize real files."
        )
    datas.append((str(src), "models/shabad-id-models"))

# --- Corpus ----------------------------------------------------------------
# ShabadCorpus loads from data/sggs_corpus.json (single consolidated file).
datas += [
    (str(DATA_DIR / "sggs_corpus.json"), "data"),
]

# Bundle engine/ as a package so all imports resolve correctly.
for fn in ENGINE_DIR.rglob("*.py"):
    rel_dir = str(fn.parent.relative_to(REPO_ROOT)).replace("\\", "/")
    datas.append((str(fn), rel_dir))

# --- Hidden imports --------------------------------------------------------
# PyInstaller's static analysis usually finds onnxruntime + sentencepiece +
# rapidfuzz on its own, but onnxruntime ships some submodules dynamically
# (capi/ in particular) and aiohttp is a worth-being-explicit case. We
# also pin the engine package modules so they're discoverable.
hiddenimports = [
    "engine",
    "engine.asr",
    "engine.config",
    "engine.corpus",
    "engine.event_types",
    "engine.google_auth",
    "engine.matcher",
    "engine.matcher_state",
    "engine.protocols",
    "engine.routes",
    "engine.session",
    "engine._internal",
    "engine._internal.hard_ctc",
    "engine._internal.onnx_inference",
    "aiohttp",
    "sentencepiece",
    "rapidfuzz",
    "rapidfuzz.fuzz",
    "rapidfuzz.distance",
    "numpy",
    # engine/asr.py started importing `wave` (stdlib) in v0.2.0 to load
    # bundled benchmark clips. PyInstaller 6.20 sometimes misses stdlib
    # modules pulled in via deferred imports on Python 3.14; pinning here
    # avoids "ModuleNotFoundError: No module named 'wave'" at boot.
    "wave",
]
# Google Cloud auth — used by engine.google_auth + asr.GoogleCloudASR for
# the production customer flow (uploaded service account JSON) and the
# dev flow (gcloud ADC fallback). PyInstaller's static analysis finds
# the obvious `google.auth` / `google.oauth2.service_account` imports
# but a couple of submodules (transport.requests, _default discovery
# helpers) are reached via late import inside google-auth itself.
# Cryptography is the RS256 signer for service account JWT exchange —
# pulls in native OpenSSL bindings that PyInstaller sometimes misses
# when only the high-level facade is imported.
hiddenimports += [
    "google.auth",
    "google.auth.exceptions",
    "google.auth.transport.requests",
    "google.oauth2",
    "google.oauth2.service_account",
    "google.oauth2._client",
    "cachetools",
    "rsa",
    "pyasn1",
    "pyasn1_modules",
    "cryptography",
    "cryptography.hazmat.backends.openssl",
    "cryptography.hazmat.primitives.serialization",
    "cryptography.hazmat.primitives.asymmetric.rsa",
    "cryptography.hazmat.primitives.asymmetric.padding",
    "cryptography.hazmat.primitives.hashes",
]
# Only pull the onnxruntime submodules we actually use at runtime.
# collect_submodules("onnxruntime") drags in ~170 modules (tools, flatbuffers,
# quantization, optimizers) that add bulk without benefit.
hiddenimports += [
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "onnxruntime.capi.onnxruntime_inference_collection",
    "onnxruntime.capi.onnxruntime_validation",
]

# Pull in onnxruntime's runtime data files (DLLs/dylibs, providers).
# `collect_data_files` returns (src, dest) tuples already.
# Filter out tools/transformers/optimizer Python files we don't need.
_ort_datas = collect_data_files("onnxruntime")
_ort_datas = [(s, d) for s, d in _ort_datas
              if not any(skip in d for skip in ("tools", "transformers", "quantization", "optimizer"))]
datas += _ort_datas
# sentencepiece's compiled .so is a binary, not a data file — PyInstaller
# handles it via the Analysis() binaries automatic discovery.

# --- Modules to *exclude* to keep the bundle small -------------------------
# These come along for the ride if we don't ban them. Together they're
# roughly 500 MB of wheels we never call from the desktop ONNX path. Easy
# wins; if a runtime ImportError surfaces in `desktop/smoke_ws.py`, drop
# the offender from this list.
excludes = [
    # Big ML libs we replaced with onnxruntime
    "torch", "torchaudio", "torchvision",
    "transformers", "huggingface_hub", "hf_xet", "tokenizers",
    "tiktoken",
    "tensorflow", "tensorboard",
    "onnx",                    # the protobuf-IR pkg, distinct from onnxruntime
    # Heavy data-science extras pulled in transitively
    "matplotlib", "PIL", "pandas", "pyarrow",
    "scipy",
    "numba", "llvmlite",       # ~220 MB combined; we don't JIT anything
    "sklearn",
    # Stdlib + dev tooling we don't ship
    "tkinter", "test", "unittest",
    "IPython", "jupyter_client", "ipykernel", "notebook",
    "pytest", "_pytest",
    "soundfile",               # only used by training/eval scripts
]


# --- Build pipeline --------------------------------------------------------
a = Analysis(
    [str(ENTRY)],
    pathex=[str(ENGINE_DIR)],   # so `import shabad_engine` resolves
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# --- Strip CUDA/TensorRT provider DLLs we never use ----------------------
# Keep DirectML (~10 MB) for GPU acceleration on any Windows GPU.
# Strip CUDA (~400 MB) and TensorRT — DirectML covers NVIDIA too via DX12.
_STRIP_PROVIDER_PATTERNS = (
    "onnxruntime_providers_cuda",
    "onnxruntime_providers_tensorrt",
    "onnxruntime_providers_shared",   # only needed by CUDA/TRT providers
    "libcudart",
    "libcublas",
    "libcufft",
    "libcurand",
    "libcudnn",
    "cudnn",
    "cublas",
    "cufft",
    "curand",
    "cudart",
    "nvinfer",
)

def _is_stripped_binary(name: str) -> bool:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return any(pat.lower() in base.lower() for pat in _STRIP_PROVIDER_PATTERNS)

_before = len(a.binaries)
a.binaries = [(n, p, t) for n, p, t in a.binaries if not _is_stripped_binary(n)]
_stripped = _before - len(a.binaries)
if _stripped:
    print(f"[sidecar.spec] stripped {_stripped} CUDA/TRT provider binaries (kept DirectML)", flush=True)

_before_d = len(a.datas)
a.datas = [(n, p, t) for n, p, t in a.datas if not _is_stripped_binary(n)]
_stripped_d = _before_d - len(a.datas)
if _stripped_d:
    print(f"[sidecar.spec] stripped {_stripped_d} CUDA/TRT provider data files", flush=True)

pyz = PYZ(a.pure)

# Onefile EXE: bundles a.binaries + a.datas directly into the executable
# instead of leaving them in a sibling _internal/ folder. PyInstaller's
# bootloader extracts them to sys._MEIPASS at launch time. The 3-5s cold
# start is the cost; the win is one file to ship.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="gurbani-captioning",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                     # UPX corrupts onnxruntime dylibs on macOS
    runtime_tmpdir=None,           # use system default (/var/folders on Mac)
    console=False,                  # Tauri captures stdout/stderr via pipes;
                                    # console=True on Windows opens a visible
                                    # cmd window and creates a separate process
                                    # group that prevents clean kill on exit.
    disable_windowed_traceback=False,
    target_arch=None,              # native (arm64 on Apple Silicon)
    codesign_identity=None,
    entitlements_file=None,
)
# No COLLECT() block — onefile mode skips it. The single `dist/gurbani-captioning`
# binary is everything Tauri needs to externalBin.
