#!/usr/bin/env python3
"""
ONNX inference wrapper for v4 IndicConformer CTC.

    from engine._internal.onnx_inference import OnnxIndicConformer
    model = OnnxIndicConformer("models/shabad-id-models/v4.int8.onnx")
    text  = model.transcribe(audio_waveform)          # str
    logp  = model.extract_logprobs(audio_waveform)    # np.ndarray [T, 257]

Public API: `transcribe`, PA logprob extraction, vocab list.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime not installed. Install with `pip install onnxruntime`."
    ) from e

try:
    import sentencepiece as spm
except ImportError as e:
    raise ImportError(
        "sentencepiece not installed. Install with `pip install sentencepiece`."
    ) from e


SAMPLE_RATE = 16000
PA_BLANK_LOCAL = 256  # last index in PA-masked logprobs (256 PA tokens + blank)


class OnnxIndicConformer:
    """ONNX-based inference wrapper for v4 IndicConformer CTC.

    Only fp32 float32 waveforms at 16 kHz are supported (single channel).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_path: Optional[str | Path] = None,
        num_threads: Optional[int] = None,
        force_cpu: bool = False,
    ):
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Default tokenizer sidecar next to the .onnx file.
        if tokenizer_path is None:
            tokenizer_path = onnx_path.with_name(
                onnx_path.stem.split(".")[0] + "_tokenizer.model"
            )
        tokenizer_path = Path(tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}. "
                "Export produces this alongside the .onnx file."
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads is not None:
            sess_opts.intra_op_num_threads = num_threads
            sess_opts.inter_op_num_threads = 1

        # ── Provider selection + diagnostics ──────────────────────────
        available_providers = ort.get_available_providers()
        # Prefer GPU-accelerated providers if available, fall back to CPU.
        # DirectML = AMD/Intel/NVIDIA on Windows via DirectX 12.
        # CUDA = NVIDIA only, needs cuDNN.
        # force_cpu = True overrides everything (used by hosted servers
        # where GPU providers may not be thread-safe under concurrent
        # session calls).
        if force_cpu:
            provider_priority = ["CPUExecutionProvider"]
        else:
            provider_priority = [
                # CoreML disabled (May 2026): silently produces garbage logits
                # on the int8-quantized IndicConformer model — every output collapses
                # to <unk>. Suspect unsupported op falling back without proper
                # weight dequantization. CPU on Apple Silicon is plenty fast
                # (~0.2-0.6s for 20s window) so the speedup wasn't worth the
                # silent corruption. Verified with /tmp/test3.py: CPU returns
                # real Gurmukhi, CoreML returns "⁇" on identical input.
                # "CoreMLExecutionProvider",
                "CUDAExecutionProvider",     # NVIDIA GPU
                "DmlExecutionProvider",      # Windows: DirectX 12 GPU
                "CPUExecutionProvider",
            ]
        selected = [p for p in provider_priority if p in available_providers]
        if not selected:
            selected = ["CPUExecutionProvider"]

        t0 = time.time()
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=selected,
        )
        self.load_time = time.time() - t0
        self._active_providers = selected

        # Thread-safety strategy depends on the active provider:
        #   - CPUExecutionProvider: thread-safe; ORT's thread pool handles
        #     parallelism within each call AND concurrent calls fan out
        #     across the asyncio thread pool. NO LOCK needed.
        #   - DmlExecutionProvider / CUDAExecutionProvider: historically
        #     not safe under concurrent session.run() (we saw native
        #     segfaults on DML at N=2+). Add a defensive serialization
        #     lock so the desktop GPU path stays safe.
        # The lock is acquired in run(); on CPU it's a no-op contextlib.
        import threading
        import contextlib
        if selected and selected[0] == "CPUExecutionProvider":
            self._run_lock = contextlib.nullcontext()
        else:
            self._run_lock = threading.Lock()

        # Log what we actually got (provider negotiation can silently fall back)
        active_providers = self.session.get_providers()
        import platform
        import multiprocessing
        print(f"[onnx] available providers: {available_providers}", flush=True)
        print(f"[onnx] requested: {selected}", flush=True)
        print(f"[onnx] active:    {active_providers}", flush=True)
        print(f"[onnx] model loaded in {self.load_time:.2f}s", flush=True)
        print(f"[onnx] onnxruntime {ort.__version__}, "
              f"Python {platform.python_version()}, "
              f"{platform.system()} {platform.release()} {platform.machine()}, "
              f"CPUs={multiprocessing.cpu_count()}, "
              f"threads={sess_opts.intra_op_num_threads or 'auto'}",
              flush=True)

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tokenizer_path))

        self.onnx_path = onnx_path
        self.tokenizer_path = tokenizer_path

    # ─────────────────────────────────────────────────────────────────────
    # Low-level
    # ─────────────────────────────────────────────────────────────────────
    def _prep_audio(self, audio) -> tuple[np.ndarray, np.ndarray]:
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]  # [1, T]
        if audio.ndim != 2:
            raise ValueError(f"Expected [T] or [B, T], got shape {audio.shape}")
        audio_len = np.array(
            [audio.shape[1]] * audio.shape[0], dtype=np.int64
        )
        return audio, audio_len

    def run(self, audio) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass → (log_probs [B, T', 257], out_len [B])."""
        a, al = self._prep_audio(audio)
        with self._run_lock:
            lp, ol = self.session.run(
                ["log_probs", "out_len"],
                {"audio": a, "audio_len": al},
            )
        return lp, ol

    # ─────────────────────────────────────────────────────────────────────
    # High-level API (matches IndicConformerCTC)
    # ─────────────────────────────────────────────────────────────────────
    def transcribe(self, audio) -> list[str]:
        """Transcribe audio → list of Gurmukhi strings (one per batch item)."""
        lp, ol = self.run(audio)
        return self._greedy_decode(lp, ol)

    def extract_logprobs(self, audio) -> Optional[np.ndarray]:
        """Return PA log-probs [T, 257] for a single waveform, or None if too short."""
        a, _ = self._prep_audio(audio)
        if a.shape[1] < SAMPLE_RATE * 0.5:
            return None
        from ..corpus import normalize_quiet_audio
        a = normalize_quiet_audio(a)
        lp, ol = self.run(a)
        T = int(ol[0])
        return lp[0, :T]

    def get_pa_vocab(self) -> list[str]:
        """Return Punjabi vocab list (256 tokens) for CTC / constrained decoding."""
        return [self.tokenizer.IdToPiece(i) for i in range(self.tokenizer.GetPieceSize())]

    # ─────────────────────────────────────────────────────────────────────
    # Decoding
    # ─────────────────────────────────────────────────────────────────────
    def _greedy_decode(
        self, log_probs: np.ndarray, lengths: np.ndarray
    ) -> list[str]:
        """CTC greedy decode with blank removal and consecutive-dup removal."""
        B = log_probs.shape[0]
        texts = []
        for b in range(B):
            T = int(lengths[b])
            seq = log_probs[b, :T].argmax(-1).tolist()
            decoded = []
            prev = -1
            for tok in seq:
                if tok != PA_BLANK_LOCAL and tok != prev:
                    decoded.append(int(tok))
                prev = tok
            texts.append(self.tokenizer.decode(decoded))
        return texts


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument(
        "--model", default="models/shabad-id-models/v4.int8.onnx",
        help="Path to .onnx model",
    )
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from nemo_standalone_inference import load_audio

    print(f"Loading {args.model} ...")
    model = OnnxIndicConformer(args.model, num_threads=args.threads)
    print(f"  loaded in {model.load_time:.2f}s")

    audio, _ = load_audio(args.audio)
    max_samp = int(args.max_seconds * SAMPLE_RATE)
    if audio.shape[1] > max_samp:
        audio = audio[:, :max_samp]
    dur = audio.shape[1] / SAMPLE_RATE
    print(f"\nAudio: {args.audio}  ({dur:.1f}s)")

    t0 = time.time()
    texts = model.transcribe(audio.numpy())
    dt = time.time() - t0
    print(f"\nTranscription ({dt:.2f}s, {dur/dt:.1f}× realtime):")
    print("─" * 60)
    print(texts[0])
    print("─" * 60)


if __name__ == "__main__":
    main()
