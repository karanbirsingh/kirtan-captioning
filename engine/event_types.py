"""Public data types for the engine.

Pure dataclasses; no I/O, no globals, no engine dependencies. Safe to
import from anywhere (matcher, session, tests, fixtures).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ShabadCandidate:
    """A candidate shabad with score and matched line.

    Returned from `ShabadMatcher.find_candidates()`. The first five fields
    are required; the matched-line fields are populated during
    identification (for UI display) but omitted from `to_dict()` if absent.
    """
    shabad_id: int
    name: str
    score: float
    rank: int
    # Matched line info (populated during identification)
    matched_line_index: Optional[int] = None
    matched_line_text: Optional[str] = None
    matched_line_translation: Optional[str] = None
    matched_line_transliteration: Optional[str] = None
    matched_line_score: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Only include line info if present
        if self.matched_line_index is None:
            del d["matched_line_index"]
            del d["matched_line_text"]
            del d["matched_line_translation"]
            del d["matched_line_transliteration"]
            del d["matched_line_score"]
        return d


@dataclass
class LineMatch:
    """A matched line within the locked shabad.

    Returned from `ShabadMatcher.match_line()`. `margin` is the score gap
    between the best and second-best line; the state machine uses it for
    hysteresis (exp 029) to avoid line-flip ping-pong.
    """
    line_index: int
    text: str
    translation: str
    transliteration: str
    score: float
    margin: float = 0.0


__all__ = ["ShabadCandidate", "LineMatch"]
