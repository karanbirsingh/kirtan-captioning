#!/usr/bin/env python3
"""Regression test: feed known Gurmukhi text to the matcher/SM (no audio,
no ONNX, no server) and record every event. After refactoring, re-run and
diff to verify zero behavioral change.

Acts as a fake STTM target: we know exactly what lines are being "sung"
and feed them at realistic intervals. The output is a deterministic event
log that can be compared byte-for-byte.

Usage:
    python tests/regression_sttm.py                    # run + save baseline
    python tests/regression_sttm.py --check baseline   # run + diff against baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Setup path to engine package ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def build_engine():
    """Construct corpus + matcher + SM without any server/ASR."""
    from engine import Config, MatcherStateMachine, ShabadCorpus, ShabadMatcher

    corpus = ShabadCorpus(REPO_ROOT / "data" / "sggs_corpus.json")
    corpus.load()
    matcher = ShabadMatcher(corpus)
    # Pin candidates_to_show=10 for the regression test: the baseline JSON
    # was captured with top-10 emission, and the test asserts that exact
    # event stream. Production default dropped to 5 (matches web/edge.js)
    # for UX reasons, but the regression suite tests the matcher's RANKING
    # not its truncation — so we keep 10 here to exercise more of the list.
    sm = MatcherStateMachine(matcher=matcher, config=Config.from_env(auto_lock_enabled=True, candidates_to_show=10))
    return sm, corpus


def strip_volatile(events: list[dict]) -> list[dict]:
    """Remove fields that vary across runs / corpus versions.

    Strips:
      - duration_seconds / session_id / _ts: per-run noise
      - translation_english / line_translation / matched_line_translation:
        depend on whether the loaded corpus is the slim (matcher-only) or
        full schema. The MATCHING decisions (which shabad, which line)
        are corpus-version-independent and that's what we care about.
    """
    _volatile_top = {"duration_seconds", "session_id", "_ts"}
    _volatile_display = {"line_translation", "matched_line_translation", "translation"}
    out = []
    for ev in events:
        clean: dict = {}
        for k, v in ev.items():
            if k in _volatile_top:
                continue
            if k in _volatile_display:
                continue
            # Also strip display fields from nested candidates / lines
            if isinstance(v, list):
                v = [
                    {kk: vv for kk, vv in item.items() if kk not in _volatile_display}
                    if isinstance(item, dict) else item
                    for item in v
                ]
            clean[k] = v
        out.append(clean)
    return out


def run_scenario(sm, corpus) -> list[dict]:
    """Simulate a kirtan session for shabad 1512 (Asa Mehla 5).

    Feed lines progressively as if a singer is going through the shabad.
    Uses identification first (simulating ASR producing partial matches),
    then manual lock, then tracking with exact line text.

    Returns the full event log.
    """
    all_events = []
    t = 0.0  # simulated duration

    # Get the actual lines of shabad 1512
    lines = corpus.get_lines(1512)
    line_texts = []
    for v in lines:
        text = v.get("unicode", "")
        if text and len(text.strip()) > 5:
            line_texts.append(text)

    # --- Phase 1: Identification ---
    # Feed progressively more text, simulating ASR output improving

    # t=5s: first ASR attempt, partial first line
    t = 5.0
    sm.mark_identification_tick(t)
    events = sm.tick_identification(line_texts[0][:20], t)
    all_events.extend(events)

    # t=10s: more of the shabad
    t = 10.0
    sm.mark_identification_tick(t)
    combined = " ".join(line_texts[0:2])
    events = sm.tick_identification(combined, t)
    all_events.extend(events)

    # t=15s: enough text for identification
    t = 15.0
    sm.mark_identification_tick(t)
    combined = " ".join(line_texts[0:3])
    events = sm.tick_identification(combined, t)
    all_events.extend(events)

    # t=20s: another tick
    t = 20.0
    sm.mark_identification_tick(t)
    combined = " ".join(line_texts[0:4])
    events = sm.tick_identification(combined, t)
    all_events.extend(events)

    # t=25s: yet another (should have candidates by now)
    t = 25.0
    sm.mark_identification_tick(t)
    combined = " ".join(line_texts[0:5])
    events = sm.tick_identification(combined, t)
    all_events.extend(events)

    # --- Phase 2: Manual lock (user confirms) ---
    t = 26.0
    events = sm.manual_lock(shabad_id=1512, start_line=0, duration=t)
    all_events.extend(events)

    assert sm.phase == "tracking", f"Expected tracking after lock, got {sm.phase}"
    assert sm.locked_shabad_id == 1512

    # --- Phase 3: Line tracking ---
    # Feed each line's text as if the singer is progressing through the shabad.
    # Each line takes ~5s to sing, tracking ticks every 2s.

    for line_idx, line_text in enumerate(line_texts[:10]):  # first 10 lines
        # Simulate 2 tracking ticks per line (overlapping 15s window would
        # contain this line + surrounding context)
        for tick in range(2):
            t += 2.0
            sm.mark_tracking_tick(t)
            # Window would contain current line + some neighboring text
            window_start = max(0, line_idx - 1)
            window_end = min(len(line_texts), line_idx + 2)
            window_text = " ".join(line_texts[window_start:window_end])
            events = sm.tick_tracking(window_text, t)
            all_events.extend(events)

    return all_events


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", metavar="BASELINE_FILE",
                        help="Compare against a saved baseline file")
    parser.add_argument("--out", default=str(REPO_ROOT / "tests" / "regression_baseline.json"),
                        help="Output file for the event log")
    args = parser.parse_args()

    print("Building engine (corpus + matcher + SM)...", flush=True)
    sm, corpus = build_engine()
    print(f"  {len(corpus.shabads)} shabads loaded", flush=True)

    print("Running scenario (shabad 1512)...", flush=True)
    events = run_scenario(sm, corpus)
    clean = strip_volatile(events)

    # Summarize
    types = {}
    for ev in clean:
        t = ev.get("type", "?")
        types[t] = types.get(t, 0) + 1
    print(f"  {len(clean)} events: {types}", flush=True)

    # Check for expected events
    has_candidates = any(e.get("type") == "candidates" for e in clean)
    has_locked = any(e.get("type") == "locked" for e in clean)
    has_line_update = any(e.get("type") == "line_update" for e in clean)
    print(f"  candidates={has_candidates}, locked={has_locked}, line_update={has_line_update}")

    if args.check:
        # Diff against baseline
        baseline_path = Path(args.check)
        if not baseline_path.exists():
            print(f"ERROR: baseline file not found: {baseline_path}")
            return 1
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if clean == baseline:
            print(f"\nPASS: {len(clean)} events match baseline exactly.")
            return 0
        else:
            # Find first difference
            for i, (a, b) in enumerate(zip(clean, baseline)):
                if a != b:
                    print(f"\nFAIL at event {i}:")
                    print(f"  expected: {json.dumps(b, ensure_ascii=False)[:200]}")
                    print(f"  got:      {json.dumps(a, ensure_ascii=False)[:200]}")
                    break
            if len(clean) != len(baseline):
                print(f"  Event count differs: {len(clean)} vs {len(baseline)}")
            return 1
    else:
        # Save as new baseline
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(clean, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        print(f"\nBaseline saved to {out_path}")
        print(f"After refactoring, run: python {Path(__file__).name} --check {out_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
