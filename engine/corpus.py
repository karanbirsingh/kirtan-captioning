"""Shabad corpus loader.

Loads SGGS shabads from a single consolidated JSON file (the canonical
artifact). The JSON is an array of shabad objects, each with `shabad_id`,
`verses` (each verse has `gurmukhi`, `unicode`, `transliteration_english`,
`translation_english`, `line_no`, `page_no`), plus metadata (`raag`, `writer`).

The per-shabad directory layout (one JSON file per shabad) lives in the
private `bani/` repo as data-prep source. The public repo only carries
the consolidated file — a build script regenerates it from upstream
(banidb API or the per-shabad source) when you intentionally want to
refresh.

After `load()`, the matcher reads:
    shabads[sid]                  -> full shabad dict
    shabad_texts[sid]             -> concatenated unicode text
    shabad_texts_normalized[sid]  -> punct/numeral-stripped
    shabad_names[sid]             -> first-line snippet
    lines_dict[idx]               -> normalized text per line (for rapidfuzz)
    lines_meta[idx]               -> (shabad_id, line_idx, display_text)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path


logger = logging.getLogger("live_detection")


# Audio constants (used by every audio-consuming module).
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 4  # float32


def normalize_quiet_audio(audio, target_peak: float = 0.8, log: bool = False):
    """Boost quiet recordings to a consistent peak before ASR.

    IndicConformer was trained on near-clipping studio audio; mic captures
    from across a room peak at 0.05-0.3 and confuse the model. We rescale
    anything in the (0.001, 0.5) range so peak hits `target_peak` (0.8).
    Very quiet (likely silence) or already-loud audio is left untouched.

    Returns the (possibly scaled) array. Importing modules can call this
    instead of duplicating the threshold logic.
    """
    import numpy as np
    max_val = float(np.abs(audio).max())
    if 0.001 < max_val < 0.5:
        scale = target_peak / max_val
        scaled = audio * scale
        if log:
            logger.info(
                "Normalized audio by %.1fx (was max %.4f, now %.2f)",
                scale, max_val, float(np.abs(scaled).max()),
            )
        return scaled
    return audio

# Synthetic shabads (Simran class) live at IDs >= this value. They are not
# in SGGS; they represent generic chant patterns (Waheguru, Satnam Waheguru,
# Ik Onkar) so the matcher has a sane label when no real shabad is being
# sung. The matcher re-ranks with bidir F1 when these win, so real shabads
# that share the chant core still win when their distinctive tokens are
# present in the transcription window.
SYNTHETIC_SHABAD_ID_MIN = 9_000_000


class ShabadCorpus:
    """Loads SGGS shabads from a single consolidated JSON file.

    The JSON is an array of shabad objects. Each has `shabad_id` (int)
    and `verses` (list of verse objects). The matcher reads:
        shabads[sid]                  -> full shabad dict
        shabad_texts[sid]             -> concatenated unicode text
        shabad_texts_normalized[sid]  -> punct/numeral-stripped
        shabad_names[sid]             -> first-line snippet
        lines_dict[idx]               -> normalized text per line (for rapidfuzz)
        lines_meta[idx]               -> (shabad_id, line_idx, display_text)
    """

    def __init__(self, corpus_path: str | Path):
        self.corpus_path = Path(corpus_path)
        self.shabads: dict[int, dict] = {}
        self.shabad_texts: dict[int, str] = {}
        self.shabad_texts_normalized: dict[int, str] = {}
        self.shabad_names: dict[int, str] = {}

        # Line-level index for the line tracker (exp 013).
        # lines_dict: idx -> normalized text (for rapidfuzz lookup)
        # lines_meta: idx -> (shabad_id, line_idx, display_text)
        self.lines_dict: dict[int, str] = {}
        self.lines_meta: dict[int, tuple[int, int, str]] = {}

    def _index_shabad(self, data: dict) -> None:
        """Build all in-memory indices for one shabad dict."""
        shabad_id = data["shabad_id"]
        if shabad_id in self.shabads:
            return  # idempotent / dedupe

        self.shabads[shabad_id] = data

        verses = data.get("verses", [])
        full_text = " ".join(v.get("unicode", "") for v in verses)
        self.shabad_texts[shabad_id] = full_text

        normalized = re.sub(r"[॥੦੧੨੩੪੫੬੭੮੯।]", "", full_text)
        normalized = " ".join(normalized.split())
        self.shabad_texts_normalized[shabad_id] = normalized

        if verses:
            gurmukhi = verses[0].get("gurmukhi", {})
            if isinstance(gurmukhi, dict) and gurmukhi.get("unicode"):
                first_line = gurmukhi["unicode"][:60]
            else:
                first_line = verses[0].get("unicode", "")[:60]
            self.shabad_names[shabad_id] = first_line
        else:
            self.shabad_names[shabad_id] = f"Shabad {shabad_id}"

        # Line-level index (exp 013).
        for line_idx, verse in enumerate(verses):
            gurmukhi = verse.get("gurmukhi", {})
            if isinstance(gurmukhi, dict):
                display_text = gurmukhi.get("unicode", verse.get("unicode", ""))
            else:
                display_text = verse.get("unicode", "")

            line_normalized = re.sub(r"[॥੦੧੨੩੪੫੬੭੮੯।]", "", display_text)
            line_normalized = " ".join(line_normalized.split())

            if line_normalized:  # skip empty lines
                idx = len(self.lines_dict)
                self.lines_dict[idx] = line_normalized
                self.lines_meta[idx] = (shabad_id, line_idx, display_text)

    def load(self) -> None:
        """Load all shabads from the consolidated JSON file."""
        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus JSON not found: {self.corpus_path}. "
                "Set Config.corpus_path (env: CORPUS_PATH) or pass a Path "
                "to ShabadCorpus."
            )

        # System diagnostics — helps debug slow-CPU lag reports.
        import platform
        import multiprocessing
        logger.info(
            "[system] Python %s, OS %s %s, CPU cores=%d, arch=%s",
            platform.python_version(),
            platform.system(), platform.release(),
            multiprocessing.cpu_count(),
            platform.machine(),
        )

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(
                f"Corpus JSON {self.corpus_path} must be a JSON array of "
                f"shabad objects; got {type(data).__name__}."
            )

        for shabad in data:
            self._index_shabad(shabad)

        logger.info(
            "Loaded corpus from %s: %d shabads, %d lines indexed",
            self.corpus_path.name,
            len(self.shabads),
            len(self.lines_dict),
        )

    def get_lines(self, shabad_id: int) -> list[dict]:
        """Get all lines/verses for a shabad."""
        shabad = self.shabads.get(shabad_id, {})
        return shabad.get("verses", [])


__all__ = [
    "ShabadCorpus",
    "SAMPLE_RATE",
    "BYTES_PER_SAMPLE",
    "SYNTHETIC_SHABAD_ID_MIN",
]
