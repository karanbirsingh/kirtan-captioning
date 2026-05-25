"""Fetch shabad phrase sets from BaniDB for Google ASR adaptation biasing.

Called once on shabad lock (per session). The fetched phrase list is then
passed to GoogleCloudASR.transcribe_async() on every tracking ASR call so
Chirp biases its decoder toward the locked shabad's actual lyrics.

Phrase tiers (per Google adaptation best practices):
  - Full verse lines               boost 15
  - Vishraam-segmented sub-phrases boost 10  (semantic breath units;
                                              match how kirtaniyas phrase
                                              lines audibly)

Caching: module-level dict keyed by shabad_id. ~3 KB per shabad, unbounded
but bounded in practice by number of distinct shabads sung. Background
fetch is best-effort — if BaniDB is unreachable, returns [] and tracking
proceeds without biasing.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("live_detection")

_CACHE: dict[int, list[dict[str, Any]]] = {}
_BANIDB_URL = "https://api.banidb.com/v2/shabads/{shabad_id}"

# Floor for sub-segment usefulness. 1-word segments add noise without
# meaningfully helping Chirp pick the right token in context.
_MIN_SEGMENT_WORDS = 2

# Google Speech-to-Text PhraseSet limits (V1/V2 shared, Chirp 2 follows).
# We stay comfortably below these — the SGGS corpus's largest shabad
# generates ~1600 phrases / ~22 KB chars — but enforce hard caps so an
# unexpectedly long shabad (Dasam Granth, custom upload) can't silently
# break tracking.
_MAX_PHRASES = 4500          # 5000 hard limit; keep 10% safety margin
_MAX_TOTAL_CHARS = 90_000    # 100,000 hard limit; same margin
_MAX_PHRASE_CHARS = 100      # hard limit per phrase; any longer is dropped


def _clean(text: str) -> str:
    """Strip danda/digit decorations Chirp never emits anyway."""
    if not text:
        return ""
    out = text.replace("॥", "").replace("।", "")
    # Strip Gurmukhi digit runs (verse numbers like ੧, ੬੦) — these appear
    # at line ends and aren't sung as words.
    out = "".join(ch for ch in out if not ("\u0a66" <= ch <= "\u0a6f"))
    return " ".join(out.split())  # collapse whitespace


def _segment_by_vishraam(words: list[str], visraam: list[dict]) -> list[str]:
    """Split a verse into sub-phrases at every pause point.

    `visraam` entries have shape {"p": word_index, "t": "v"|"y"} where the
    pause falls AFTER the word at index p. We use both heavy (v) and light
    (y) pauses since both correspond to audible breaths.
    """
    if not visraam:
        return []
    cuts = sorted({entry["p"] + 1 for entry in visraam if "p" in entry})
    segments: list[str] = []
    prev = 0
    for c in list(cuts) + [len(words)]:
        if c > prev and c <= len(words):
            seg = " ".join(words[prev:c]).strip()
            if seg and len(seg.split()) >= _MIN_SEGMENT_WORDS:
                segments.append(seg)
            prev = c
    return segments


async def fetch_phrases(shabad_id: int) -> list[dict[str, Any]]:
    """Return list of {"value": str, "boost": float} for a shabad.

    Cached: subsequent calls for the same shabad_id are O(1). On any
    network/parsing failure returns [] and stores [] so we don't retry in
    a hot loop.
    """
    if shabad_id in _CACHE:
        return _CACHE[shabad_id]
    _CACHE[shabad_id] = []  # negative cache while in-flight / on failure

    try:
        import aiohttp  # type: ignore
    except ImportError:
        logger.warning("aiohttp not available; skipping BaniDB phrase fetch")
        return []

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                _BANIDB_URL.format(shabad_id=shabad_id),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "BaniDB shabad %d: HTTP %d", shabad_id, resp.status,
                    )
                    return []
                data = await resp.json()
    except Exception as e:
        logger.warning("BaniDB shabad %d fetch failed: %s", shabad_id, e)
        return []

    phrases: list[dict[str, Any]] = []
    seen: set[str] = set()
    total_chars = 0
    dropped_long = 0
    lines_dropped_for_size = 0

    def _try_add(value: str, boost: float) -> bool:
        """Add a phrase if it fits within all limits. Returns False on cap hit."""
        nonlocal total_chars, dropped_long
        if not value or value in seen:
            return True
        if len(value) > _MAX_PHRASE_CHARS:
            dropped_long += 1
            return True  # not a hard stop; just skip this entry
        if len(phrases) >= _MAX_PHRASES:
            return False
        if total_chars + len(value) > _MAX_TOTAL_CHARS:
            return False
        phrases.append({"value": value, "boost": boost})
        seen.add(value)
        total_chars += len(value)
        return True

    verses = data.get("verses", [])

    # Pass 1: full lines only. Lines are the most important biasing tier;
    # we always want them in if possible. Worst-case SGGS shabad has ~400
    # verses → ~10 KB chars → well under both caps.
    for v in verses:
        verse_obj = v.get("verse") or {}
        line = _clean(verse_obj.get("unicode", ""))
        if not line:
            continue
        if not _try_add(line, 15.0):
            lines_dropped_for_size += 1

    # Pass 2: vishraam-segmented sub-phrases. These are a bonus tier —
    # if adding them would exceed any cap, we stop and the request goes
    # out with full lines only. This degrades gracefully without ever
    # silently dropping line coverage.
    segments_added = 0
    segments_skipped_full = False
    for v in verses:
        if segments_skipped_full:
            break
        verse_obj = v.get("verse") or {}
        line = _clean(verse_obj.get("unicode", ""))
        if not line:
            continue
        words = line.split()
        visraam_obj = v.get("visraam") or v.get("vishraam") or {}
        # BaniDB exposes three vendor splits (sttm, igurbani, sttm2);
        # sttm is the most widely curated.
        marks = visraam_obj.get("sttm") or visraam_obj.get("igurbani") or []
        for seg in _segment_by_vishraam(words, marks):
            if not _try_add(seg, 10.0):
                segments_skipped_full = True
                break
            segments_added += 1

    _CACHE[shabad_id] = phrases
    notes: list[str] = []
    if lines_dropped_for_size:
        notes.append(f"{lines_dropped_for_size} lines too long")
    if segments_skipped_full:
        notes.append("vishraam tier truncated (cap)")
    if dropped_long:
        notes.append(f"{dropped_long} segments >{_MAX_PHRASE_CHARS}ch")
    note_str = (" [" + ", ".join(notes) + "]") if notes else ""
    logger.info(
        "BaniDB shabad %d: %d phrases (%d lines + %d vishraam, %d chars, %d verses)%s",
        shabad_id,
        len(phrases),
        len(phrases) - segments_added,
        segments_added,
        total_chars,
        len(verses),
        note_str,
    )
    return phrases
