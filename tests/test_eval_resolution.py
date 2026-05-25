#!/usr/bin/env python3
"""Fast regression checks for benchmark label resolution."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

from eval import NO_MATCH, pred_segments_to_frames  # noqa: E402


def test_top_level_gt_shabad_id_resolves_line_idx() -> None:
    gt = {
        "video_id": "unit",
        "shabad_id": 3712,
        "total_duration": 4,
        "lines": [{"line_idx": 1, "text": "x"}],
        "segments": [{"start": 0, "end": 4, "line_idx": 1}],
    }
    pred = {
        "video_id": "unit",
        "shabad_id": 3712,
        "segments": [{"start": 0, "end": 4, "line_idx": 1}],
    }

    assert pred_segments_to_frames(pred["segments"], gt, 4, pred_top_level=pred) == [1, 1, 1, 1]


def test_wrong_top_level_shabad_id_is_penalized() -> None:
    gt = {
        "video_id": "unit",
        "shabad_id": 3712,
        "total_duration": 4,
        "lines": [{"line_idx": 1, "text": "x"}],
        "segments": [{"start": 0, "end": 4, "line_idx": 1}],
    }
    pred = {
        "video_id": "unit",
        "shabad_id": 4075,
        "segments": [{"start": 0, "end": 4, "line_idx": 1}],
    }

    assert pred_segments_to_frames(pred["segments"], gt, 4, pred_top_level=pred) == [NO_MATCH] * 4


if __name__ == "__main__":
    test_top_level_gt_shabad_id_resolves_line_idx()
    test_wrong_top_level_shabad_id_is_penalized()
    print("PASS: eval resolution tests")