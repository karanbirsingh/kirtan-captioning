"""Shabad and line matching.

ShabadMatcher does two jobs:

  1. **Shabad identification** (Phase 1) — given an ASR transcript, return
     the top-K most likely shabads from the corpus, ranked by multi-line
     agreement (exp 020 §6-7).

  2. **Line tracking** (Phase 2) — once a shabad is locked, find the best
     matching line within it for the current 15s window (exp 024 fuzzy
     keyword bidir).

The matcher is consumed by the state machine (matcher_state.py) and the
HTTP session (session.py). It does not hold any session state; you can
share one instance across all sessions.

Inputs are Gurmukhi text. ASR and audio handling live elsewhere.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .corpus import ShabadCorpus, SYNTHETIC_SHABAD_ID_MIN
from .config import Config
from .event_types import ShabadCandidate, LineMatch


logger = logging.getLogger("live_detection")


class ShabadMatcher:
    """Fuzzy matching between transcription and shabads.

    Phase 1 (Shabad ID): multi-line agreement scoring (exp 020 §6-7)
      - Score = best_line * 0.7 + min(good_lines, 5) * 6.0
      - Tiny shabads (<3 lines) get no agreement bonus
      - 98.9% top-5, 76.2% auto-lock, 3.1% false lock at 120 tracks

    Phase 2 (Line Tracking): fuzzy_kw_bidir (exp 024)
      - Bidirectional fuzzy keyword overlap (F1 of recall + precision)
      - 87.2% GT accuracy, FAQS 89.6 at 112 tracks
    """

    # Rubric/header line detection (exp 067). A line is a "header" if short
    # and/or composed only of rubric tokens (raag / mahalla / ghar / padee /
    # salok etc.) that appear as preludes and would otherwise match any
    # Gurmukhi query via partial_ratio on tiny substrings.
    _HEADER_TOKENS = {
        "ਮਹਲਾ", "ਮਃ", "ਗਉੜੀ", "ਆਸਾ", "ਆਸਾਵਰੀ", "ਸੋਰਠਿ", "ਸੂਹੀ", "ਤਿਲੰਗ",
        "ਭੈਰਉ", "ਬਸੰਤ", "ਬਸੰਤੁ", "ਮਾਝ", "ਮਾਰੂ", "ਸਾਰਗ", "ਬਿਲਾਵਲੁ", "ਬਿਲਾਵਲ",
        "ਧਨਾਸਰੀ", "ਕਾਨੜਾ", "ਮਲਾਰ", "ਮਾਲੀ", "ਮਾਲੀਗਉੜਾ", "ਨਟ", "ਟੋਡੀ", "ਜੈਤਸਰੀ",
        "ਲਲਤ", "ਨਾਮਦੇਵ", "ਕਬੀਰ", "ਰਾਗੁ", "ਮਝ", "ਘਰੁ", "ਸਲੋਕੁ", "ਸਲੋਕ", "ਚਉਪਦੇ",
        "ਦੁਪਦੇ", "ਤਿਪਦੇ", "ਪਉੜੀ", "ਸਲੋਕਮਃ", "ਸਲੋਕਮਹਲਾ", "ਜ਼ੋਲਨਾ", "ਛੰਤ",
        "ਛੰਤੁ", "ਰਹਾਉ", "ਅਸਟਪਦੀ", "ਅਸਟਪਦੀਆ", "ਵਾਰ", "ਦੋਹਰਾ",
    }

    def __init__(self, corpus: ShabadCorpus, config: Optional[Config] = None):
        self.corpus = corpus
        self.config: Config = config or Config()
        # Build line-level index for multi-line agreement scoring.
        # shabad_line_texts: sid -> normalized line texts (used by line
        # tracking / display). The flat index is used for the global
        # partial_ratio scan in find_candidates(); content lines only
        # (headers filtered).
        self.shabad_line_texts: dict[int, list[str]] = {}
        self._flat_lines: dict[int, tuple[int, str]] = {}
        self._flat_texts: dict[int, str] = {}
        self._content_line_count: dict[int, int] = {}
        for sid, data in corpus.shabads.items():
            lines = []
            content_count = 0
            for verse in data.get("verses", []):
                text = self.get_verse_text(verse)
                normalized = self.normalize_text(text)
                if len(normalized) >= 5:
                    lines.append(normalized)
                    if not self._is_header_line(normalized):
                        content_count += 1
                        idx = len(self._flat_lines)
                        self._flat_lines[idx] = (sid, normalized)
                        self._flat_texts[idx] = normalized
            self.shabad_line_texts[sid] = lines
            self._content_line_count[sid] = content_count
        logger.info(
            f"Built line index: {len(self.shabad_line_texts)} shabads, "
            f"{len(self._flat_texts)} content lines (filtered from total verses)"
        )

    @classmethod
    def _is_header_line(cls, normalized: str) -> bool:
        """Heuristic: line is a rubric/header (raag-mahalla-ghar tag) with no
        sung content. Very short lines or lines whose every word is a rubric
        token are flagged."""
        if len(normalized) < 15:
            return True
        words = [w for w in normalized.split() if w]
        if not words:
            return True
        # Every word must be either a known rubric token, digit-only, or short (<=2)
        for w in words:
            if w in cls._HEADER_TOKENS:
                continue
            if w.isdigit():
                continue
            # Punjabi digit forms (੦-੯) appear as pure digit markers too
            if all(ch in "੦੧੨੩੪੫੬੭੮੯" for ch in w):
                continue
            if len(w) <= 2:  # ਘਰੁ, ਮਃ etc.
                continue
            return False  # found a real content word
        return True

    @staticmethod
    def get_verse_text(verse: dict) -> str:
        """Extract unicode text from a verse dict."""
        gurmukhi = verse.get("gurmukhi", {})
        if isinstance(gurmukhi, dict):
            text = gurmukhi.get("unicode", verse.get("unicode", ""))
            if text and " " in text:
                return text
        return verse.get("unicode", "")

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize Gurmukhi text for matching (same as eval scripts)."""
        text = re.sub(r"[॥੦੧੨੩੪੫੬੭੮੯।]", "", text)
        text = re.sub(r"([\u0a3e-\u0a4d\u0a70\u0a71])\1+", r"\1", text)
        return " ".join(text.split()).strip()

    @staticmethod
    def fuzzy_keyword_bidir(window_norm: str, line_norm: str) -> float:
        """Bidirectional fuzzy keyword overlap — best line-tracking matcher (exp 024).

        F1 of:
          - line recall:  fraction of line words found (fuzzy ≥60) in window
          - window precision:  fraction of window words found (fuzzy ≥60) in line

        Returns 0-100 score.
        """
        from rapidfuzz import fuzz as rfuzz
        w_words = [w for w in window_norm.split() if len(w) >= 2]
        l_words = [w for w in line_norm.split() if len(w) >= 2]
        if not l_words or not w_words:
            return 0.0
        l_recall = sum(
            1 for lw in l_words
            if max((rfuzz.ratio(lw, ww) for ww in w_words), default=0) >= 60
        ) / len(l_words)
        w_prec = sum(
            1 for ww in w_words
            if max((rfuzz.ratio(lw, ww) for lw in l_words), default=0) >= 60
        ) / len(w_words)
        if l_recall + w_prec == 0:
            return 0.0
        return 2 * l_recall * w_prec / (l_recall + w_prec) * 100

    def find_candidates(self, transcription: str, top_k: int = 10) -> list[ShabadCandidate]:
        """Find top-K shabad candidates using multi-line agreement scoring.

        From experiment 020 §6-7:
          Score = best_line * 0.7 + min(good_lines, 5) * 6
          Tiny shabads (<3 lines) get no agreement bonus

        98.9% top-5 at 120 tracks. This is the production matcher for
        Phase 1 (shabad identification).
        """
        from rapidfuzz import fuzz, process

        if not transcription.strip():
            return []

        query = self.normalize_text(transcription)
        min_content_lines = self.config.min_content_lines
        good_threshold = self.config.good_line_threshold

        # Use pre-built flat index (built once at init, not per-call)
        matches = process.extract(
            query, self._flat_texts, scorer=fuzz.partial_ratio, limit=top_k * 20
        )

        # Collect ALL line scores per shabad
        shabad_line_scores: dict[int, list[float]] = {}
        for _, score, idx in matches:
            sid, _ = self._flat_lines[idx]
            shabad_line_scores.setdefault(sid, []).append(score)

        # Score each shabad using multi-line agreement (exp 067: absolute
        # bonus instead of ratio-based — long salok collections were
        # underweighted on ratio).
        shabad_scores: dict[int, float] = {}
        for sid, scores in shabad_line_scores.items():
            best = max(scores)
            good_count = sum(1 for s in scores if s >= good_threshold)
            content_total = self._content_line_count.get(sid, 0)
            # Tiny shabads can't prove multi-line agreement (exp 020 §7)
            if content_total < min_content_lines:
                agreement_bonus = 0.0
            else:
                agreement_bonus = min(good_count, 5) * 6.0  # max +30
            final_score = best * 0.7 + agreement_bonus
            shabad_scores[sid] = final_score

        # Synthetic-class re-rank: synthetic shabads (id >= 9_000_000) like
        # the Simran entries match via partial_ratio regardless of extra
        # tokens in the query. If a synthetic is top, re-score top-K using
        # bidir F1, which penalizes unmatched query tokens, so real shabads
        # with distinctive tokens correctly overtake.
        sorted_results = sorted(shabad_scores.items(), key=lambda x: -x[1])
        if sorted_results and sorted_results[0][0] >= SYNTHETIC_SHABAD_ID_MIN:
            rerank_k = min(5, len(sorted_results))
            rescored: list[tuple[int, float]] = []
            for sid, _orig in sorted_results[:rerank_k]:
                line_scores = [
                    self.fuzzy_keyword_bidir(query, ln)
                    for ln in self.shabad_line_texts.get(sid, [])
                ]
                rescored.append((sid, max(line_scores) if line_scores else 0.0))
            rescored.sort(key=lambda x: -x[1])
            # Only swap if re-rank changes the winner AND the new leader is a
            # real shabad — avoids reshuffling on noise.
            if (
                rescored[0][0] != sorted_results[0][0]
                and rescored[0][0] < SYNTHETIC_SHABAD_ID_MIN
            ):
                new_top_sid, new_top_score = rescored[0]
                remaining = [(s, sc) for (s, sc) in sorted_results if s != new_top_sid]
                sorted_results = [(new_top_sid, new_top_score)] + remaining

        candidates = []
        for rank, (shabad_id, score) in enumerate(sorted_results[:top_k], 1):
            candidates.append(ShabadCandidate(
                shabad_id=shabad_id,
                name=self.corpus.shabad_names.get(shabad_id, f"Shabad {shabad_id}"),
                score=round(score, 1),
                rank=rank,
            ))
        return candidates

    def match_line(
        self,
        transcription: str,
        shabad_id: int,
        current_line: int = 0,
    ) -> Optional[LineMatch]:
        """Find the best matching line within a shabad using fuzzy_kw_bidir.

        From experiment 024: bidirectional fuzzy keyword overlap is the
        best line-tracking matcher at 87.2% GT accuracy (FAQS 89.6).

        Searches ALL lines — experiments showed that radius-based search
        and temporal smoothing both hurt accuracy. The 15s ASR window
        already provides natural smoothing.

        Args:
            transcription: The transcribed text to match
            shabad_id: The shabad to search within
            current_line: Current line index (unused — full search is best)

        Returns:
            LineMatch with the best matching line, or None
        """
        lines = self.corpus.get_lines(shabad_id)
        if not lines:
            return None

        query = self.normalize_text(transcription)
        if not query:
            return None

        best_idx = -1
        best_score = 0.0
        second_best_score = 0.0

        # Score ALL lines with fuzzy_kw_bidir (no radius, no smoothing — exp 024)
        for idx, line in enumerate(lines):
            line_text = self.get_verse_text(line)
            line_norm = self.normalize_text(line_text)
            if len(line_norm) < 5:
                continue
            score = self.fuzzy_keyword_bidir(query, line_norm)
            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_idx = idx
            elif score > second_best_score:
                second_best_score = score

        if best_idx >= 0:
            line = lines[best_idx]
            display_text = self.get_verse_text(line)
            margin = best_score - second_best_score
            return LineMatch(
                line_index=best_idx,
                text=display_text,
                translation=line.get("translation_english", ""),
                transliteration=line.get("transliteration_english", ""),
                score=best_score,
                margin=margin,
            )

        return None


__all__ = ["ShabadMatcher"]
