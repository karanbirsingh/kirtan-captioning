"""ASR backend protocol.

Any model that turns audio into Gurmukhi text can be used as a backend.
Implement `transcribe()` and you're done. CTC models that can also expose
logprobs for constrained decoding optionally implement `extract_logprobs()`
and `get_vocab()` — the engine uses these for trie-constrained decoding
when available, and falls back to plain greedy transcription otherwise.

The session calls `transcribe_async()` from its event loop and never blocks
the loop. The default impl offloads sync `transcribe()` to a thread pool;
inherently async backends (streaming network APIs, etc.) can override
`transcribe_async()` directly.

Examples (not shipped, illustrative):
    class ChirpBackend(ASRBackend):
        def transcribe(self, audio): return google_chirp_api(audio, lang="pa")

    class WhisperBackend(ASRBackend):
        def transcribe(self, audio): return faster_whisper(audio, lang="pa")
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np

from .corpus import SAMPLE_RATE


logger = logging.getLogger("live_detection")


@dataclass(frozen=True)
class ASRCadence:
    """Per-backend tick/window overrides for the session ASR loop.

    Free backends (ONNX) can tick fast with large windows. Paid backends
    (Google batch) trade latency for cost — slower ticks, smaller windows.
    `None` means "use the engine.config default".
    """
    id_tick: Optional[float] = None       # seconds between identification ticks
    id_window: Optional[float] = None     # seconds of audio sent per ID call
    track_tick: Optional[float] = None    # seconds between tracking ticks
    track_window: Optional[float] = None  # seconds of audio sent per tracking call


class ASRBackend:
    """Base class for ASR backends.

    Optional CTC support: also implement extract_logprobs() and get_vocab()
    to enable constrained CTC decoding (better accuracy for known shabads).
    """
    cadence: ASRCadence = ASRCadence()  # default: use engine.config values

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio (16kHz float32 mono) to Gurmukhi text.

        Sync method; called from worker threads. Async callers should
        use `transcribe_async()` instead.
        """
        raise NotImplementedError

    async def transcribe_async(
        self,
        audio: np.ndarray,
        bias_phrases: Optional[list[dict]] = None,
    ) -> str:
        """Async wrapper. Default offloads `transcribe()` to a thread so
        the asyncio event loop stays responsive. Override for backends
        that are inherently async (batched, network APIs, etc.).

        `bias_phrases` is an optional list of {"value": str, "boost": float}
        for cloud backends that support speech adaptation (e.g. Chirp 2).
        Backends that don't support biasing should ignore it.
        """
        return await asyncio.to_thread(self.transcribe, audio)

    def extract_logprobs(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Return CTC log-probabilities [T, V] or None if not supported."""
        return None  # default: CTC not supported

    def get_vocab(self) -> Optional[list[str]]:
        """Return token vocabulary for CTC decoding, or None if not supported."""
        return None  # default: CTC not supported

    @property
    def supports_ctc(self) -> bool:
        """True if this backend supports constrained CTC decoding."""
        return (
            type(self).extract_logprobs is not ASRBackend.extract_logprobs
            and type(self).get_vocab is not ASRBackend.get_vocab
        )


class OnnxBackend(ASRBackend):
    """ONNX Runtime backend using IndicConformer v4 int8.

    This is the default backend shipped with the desktop app. It runs
    on CPU (or DirectML/CUDA if available) without torch.
    """

    def __init__(self, onnx_path: str | Path, force_cpu: bool = False, num_threads: int = 0):
        from ._internal.onnx_inference import OnnxIndicConformer
        onnx_path = str(onnx_path)
        logger.info("Loading ONNX model from %s...", onnx_path)
        start = time.time()
        self.model = OnnxIndicConformer(
            onnx_path,
            force_cpu=force_cpu,
            num_threads=num_threads or None,
        )
        logger.info(
            "ONNX model loaded in %.1fs (file: %s)",
            time.time() - start,
            Path(onnx_path).name,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        from .corpus import normalize_quiet_audio
        audio = normalize_quiet_audio(audio, log=True)
        arr = np.asarray(audio, dtype=np.float32)
        try:
            texts = self.model.transcribe(arr)
            return texts[0] if texts else ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""

    def extract_logprobs(self, audio: np.ndarray) -> Optional[np.ndarray]:
        from .corpus import normalize_quiet_audio
        if len(audio) < SAMPLE_RATE * 0.5:
            return None
        audio = normalize_quiet_audio(audio)
        return self.model.extract_logprobs(audio)

    def get_vocab(self) -> list[str]:
        return self.model.get_pa_vocab()


# ─── Devanagari → Gurmukhi transliteration ───────────────────────────
# Devanagari (U+0900-U+097F) and Gurmukhi (U+0A00-U+0A7F) share parallel
# structure: most codepoints map +0x100. Used to normalize cloud ASR output
# (Chirp sometimes returns Punjabi text in Devanagari script, especially on
# short utterances). The matcher always sees Gurmukhi.
#
# Specific exceptions to the +0x100 offset:
#  - Danda ।/॥ (U+0964/0965) are shared Indic punctuation — keep as-is
#    (Gurmukhi uses the same code points; offset would give Gurmukhi digits)
#  - Anusvara ं (U+0902) → tippi ੰ (U+0A70), the standard Gurmukhi nasal mark
#    used over consonants in Punjabi orthography (offset would give bindi ਂ)
#  - Devanagari nukta combinations (e.g. ड़) need post-fix to Gurmukhi singletons
_CHAR_OVERRIDES = {
    "\u0964": "\u0964",       # । danda — keep as-is
    "\u0965": "\u0965",       # ॥ double danda — keep as-is
    "\u0902": "\u0a70",       # ं anusvara → ੰ tippi (not ਂ bindi)
    "\u0901": "\u0a70",       # ँ candrabindu → ੰ tippi
    "\u0943": "\u0a4d\u0a30\u0a3f",  # ृ vocalic R sign → ੍ਰਿ (halant + ra + short i)
}
_POST_FIXES = {
    "\u0a21\u0a3c": "\u0a5c",  # ਡ਼ → ੜ (DA+nukta → RRA singleton)
}

# Vowel signs after which the nasal mark should be bindi ਂ (not tippi ੰ).
# Punjabi orthography: bindi after long vowels / vowel signs that sit to the
# right or above the consonant; tippi after short vowels (ੁ, ਿ) or no vowel
# (plain consonant).
_LONG_VOWELS_FOR_BINDI = set("\u0a3e\u0a40\u0a47\u0a48\u0a4b\u0a4c")  # ਾ ੀ ੇ ੈ ੋ ੌ


def _transliterate_to_gurmukhi(text: str) -> str:
    """Convert any Devanagari characters in text to their Gurmukhi equivalents."""
    if not text:
        return text
    out = []
    for ch in text:
        if ch in _CHAR_OVERRIDES:
            out.append(_CHAR_OVERRIDES[ch])
            continue
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            mapped = cp + 0x100
            # Only map if the target is a valid Gurmukhi codepoint
            if 0x0A00 <= mapped <= 0x0A7F:
                out.append(chr(mapped))
            else:
                out.append(ch)  # rare Devanagari char with no Gurmukhi twin
        else:
            out.append(ch)
    result = "".join(out)
    for old, new in _POST_FIXES.items():
        if old in result:
            result = result.replace(old, new)
    # Tippi → bindi after long vowel signs (Punjabi orthography rule)
    if "\u0a70" in result:
        chars = list(result)
        for i in range(1, len(chars)):
            if chars[i] == "\u0a70" and chars[i - 1] in _LONG_VOWELS_FOR_BINDI:
                chars[i] = "\u0a02"  # ਂ bindi
        result = "".join(chars)
    return result


class GoogleCloudASR(ASRBackend):
    """Google Cloud Speech-to-Text V2 (Chirp 2) backend.

    Uses the REST API — no Google SDK needed for transcription itself;
    we POST signed audio frames directly. Auth tokens come from
    ``engine.google_auth`` which resolves credentials in this order:

      1. Service account JSON uploaded via Settings → Google Chirp 2
         (the customer-facing flow; see GOOGLE_CHIRP_SETUP.md).
      2. Application Default Credentials (gcloud
         ``auth application-default login``) — kept as a dev fallback.

    project_id is auto-derived from the uploaded key file's
    ``project_id`` field if not passed explicitly to the constructor.
    """
    # Paid backend: trade latency for cost.
    # ID phase: every 6s, send 18s window. Tracking: every 4s, send 12s window.
    # Worst case (never locks, continuous ID): 18s × (1/6s) = 3 audio-min/min
    # × $0.016/min ≈ ~$3/hr. Tracking after lock: 12s × (1/4s) = 3 audio-min/min
    # ≈ ~$3/hr. Combined ceiling: ~$3/hr in either phase.
    cadence = ASRCadence(id_tick=6.0, id_window=18.0, track_tick=4.0, track_window=12.0)

    def __init__(self, project_id: str = "", api_key: str = "", model: str = "chirp_2",
                 region: str = "us-central1"):
        # Resolve project_id at construction time. Customer flow: uploaded
        # service-account JSON embeds the project; we read it from there.
        # Dev flow: gcloud ADC has an associated project (set via
        # `gcloud config set project`). Either way, the caller can also
        # override with an explicit project_id arg — useful for tests.
        if not project_id:
            try:
                from .google_auth import load_credentials
                stored = load_credentials()
                if stored:
                    project_id = stored.get("project_id", "")
            except Exception:
                pass
            if not project_id:
                # Last resort: ADC project (resolved lazily on first use
                # to avoid a slow import during construction). If neither
                # path yields a project, _get_access_token will raise a
                # human-readable error on first call.
                project_id = "<unresolved>"
        self.project_id = project_id
        self.api_key = api_key
        self.model = model
        self.region = region
        self._session = None
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0
        logger.info("Google Cloud ASR initialized (project=%s, model=%s, region=%s)",
                     project_id, model, region)

    def _endpoint(self) -> str:
        return (
            f"https://{self.region}-speech.googleapis.com/v2/projects/{self.project_id}"
            f"/locations/{self.region}/recognizers/_:recognize"
        )

    async def _get_access_token(self) -> str:
        """Return a fresh Google Cloud OAuth access token.

        Resolves credentials from the central ``engine.google_auth`` module
        so the prod path (uploaded service account JSON) and the dev path
        (gcloud ADC) go through one validated code path. Token refresh
        and caching are delegated to ``google-auth`` itself.

        Also lazily fills in self.project_id if it was unresolved at
        construction time (e.g. customer uploaded creds *after* selecting
        Google Chirp once).
        """
        if self._access_token and time.time() < self._token_expiry - 60:
            return self._access_token
        from .google_auth import get_access_token as _resolve
        # get_access_token() does refresh-if-stale internally.
        token, resolved_project = await asyncio.get_event_loop().run_in_executor(
            None, _resolve
        )
        if self.project_id in ("", "<unresolved>") and resolved_project:
            self.project_id = resolved_project
        self._access_token = token
        # google-auth refreshes ~5 minutes before expiry; we re-check at
        # the 60s mark below. Re-fetching on every call would be wasteful,
        # but credentials.token may stay valid for the full hour after
        # last refresh, so cache aggressively.
        self._token_expiry = time.time() + 3000  # ~50 min, safely under 1hr
        return token

    def transcribe(self, audio: np.ndarray) -> str:
        # Sync fallback — not used; we override transcribe_async
        raise NotImplementedError("Use transcribe_async for GoogleCloudASR")

    # Cap audio sent per API call. The session already windows audio
    # (60s for identification, 15s for tracking). This is a safety cap.
    MAX_SEND_SECONDS: float = 15.0

    async def transcribe_async(
        self,
        audio: np.ndarray,
        bias_phrases: Optional[list[dict]] = None,
    ) -> str:
        if len(audio) < SAMPLE_RATE * 0.3:
            return ""

        # Cap at MAX_SEND_SECONDS (safety — session already windows)
        max_samples = int(self.MAX_SEND_SECONDS * SAMPLE_RATE)
        if len(audio) > max_samples:
            audio = audio[-max_samples:]

        # Convert float32 [-1,1] to WAV (required by autoDecodingConfig)
        pcm = np.clip(audio, -1.0, 1.0)
        pcm_int16 = (pcm * 32767).astype(np.int16)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_int16.tobytes())
        audio_b64 = base64.b64encode(wav_buf.getvalue()).decode("ascii")

        config: dict = {
            "autoDecodingConfig": {},
            "languageCodes": ["pa-Guru-IN"],
            "model": self.model,
        }
        # Speech adaptation (phrase biasing). Chirp 2 supports inline
        # PhraseSet but NOT custom classes or class tokens, so we keep
        # the payload to plain {"value", "boost"} entries.
        if bias_phrases:
            config["adaptation"] = {
                "phraseSets": [
                    {
                        "inlinePhraseSet": {
                            "phrases": bias_phrases,
                        },
                    }
                ],
            }

        body = {
            "config": config,
            "content": audio_b64,
        }

        try:
            token = await self._get_access_token()
        except Exception as e:
            logger.error("Google ASR auth failed: %s", e)
            return ""

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            start = time.time()
            async with self._session.post(
                self._endpoint(), json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                elapsed = time.time() - start
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error("Google STT error %d: %s", resp.status, error_text[:200])
                    return ""
                data = await resp.json()

            # Extract transcript from response
            results = data.get("results", [])
            chunk_text = ""
            for r in results:
                alts = r.get("alternatives", [])
                if alts:
                    chunk_text += alts[0].get("transcript", "")

            chunk_text = chunk_text.strip()
            # Chirp sometimes returns Punjabi text in Devanagari (especially
            # short utterances). Normalize to Gurmukhi so matcher sees one script.
            chunk_text = _transliterate_to_gurmukhi(chunk_text)
            logger.info("Google STT took %.2fs, text='%s' (%.1fs audio)",
                       elapsed, chunk_text[:100], len(audio) / SAMPLE_RATE)
            return chunk_text

        except Exception as e:
            logger.error("Google STT request failed: %s", e)
            return ""

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


__all__ = ["ASRBackend", "OnnxBackend"]
