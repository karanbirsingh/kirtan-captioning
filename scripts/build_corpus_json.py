#!/usr/bin/env python3
"""Consolidate per-shabad JSON files into a single sggs_corpus.json.

This is the producer of the canonical runtime artifact:
    data/sggs_corpus.json    (~30-70 MB depending on source schema)

The consumer is `engine.corpus.ShabadCorpus(path)`. The output schema is a
JSON array of shabad objects, each with the FULL field set from the
upstream banidb-style per-shabad data \u2014 the matcher needs `gurmukhi.unicode`
(spaced display form) for proper tokenization, and clients (browser line
displays, etc.) need translations + transliterations passed through.

Usage:
    # rebuild from banidb-style per-shabad directories
    python scripts/build_corpus_json.py \\
        --src /path/to/sggs_full/shabads \\
        --src /path/to/corpus/shabads \\
        --out data/sggs_corpus.json

    # rebuild from any banidb-style per-shabad export
    python scripts/build_corpus_json.py --src ./shabads --out corpus.json

If a shabad_id appears in multiple --src dirs, the FIRST occurrence wins
(same precedence as the old directory loader). This lets a small "curated"
overlay dir take precedence over the full SGGS bulk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_dir(d: Path, seen: set[int]) -> tuple[list[dict], int]:
    """Read every *.json in `d` (sorted). Returns (new_shabads, dup_count)."""
    new: list[dict] = []
    dups = 0
    if not d.exists():
        print(f"[warn] source not found: {d}", file=sys.stderr)
        return new, 0
    for f in sorted(d.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        sid = data.get("shabad_id")
        if sid is None:
            print(f"[warn] skipping {f.name}: no shabad_id", file=sys.stderr)
            continue
        if sid in seen:
            dups += 1
            continue
        seen.add(sid)
        new.append(data)
    return new, dups


def slim_shabad(shabad: dict) -> dict:
    """Strip a shabad to matcher-essential fields only.

    Drops the display-only data (translations, transliterations, ASCII
    gurmukhi, line/page numbers). Keeps:
      - shabad_id
      - verses[*].verse_id  (stable upstream ID; clients use it to fetch
        translations from banidb)
      - verses[*].unicode   (no-space gurmukhi, used for shabad-level
        normalization)
      - verses[*].gurmukhi.unicode (spaced gurmukhi, used for line
        tokenization and matcher rapidfuzz scoring)

    Result is roughly 1/5 the size of the full schema. Clients who need
    translations / transliterations can fetch per-shabad data from
    api.banidb.com/v2/shabad/{shabad_id} on demand.
    """
    slim_verses = []
    for v in shabad.get("verses", []):
        out = {"verse_id": v.get("verse_id"), "unicode": v.get("unicode", "")}
        gur = v.get("gurmukhi")
        if isinstance(gur, dict) and gur.get("unicode"):
            out["gurmukhi"] = {"unicode": gur["unicode"]}
        # Keep translation + transliteration (used by desktop UI)
        tr = v.get("translation_english")
        if isinstance(tr, dict):
            out["translation_english"] = tr.get("en") or tr.get("english") or tr.get("bdb") or ""
        elif isinstance(tr, str):
            out["translation_english"] = tr
        tl = v.get("transliteration_english")
        if isinstance(tl, dict):
            out["transliteration_english"] = tl.get("en") or tl.get("english") or ""
        elif isinstance(tl, str):
            out["transliteration_english"] = tl
        slim_verses.append(out)
    return {"shabad_id": shabad["shabad_id"], "verses": slim_verses}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", action="append", required=True,
                   help="Source directory containing per-shabad *.json (repeatable; precedence = order given)")
    p.add_argument("--out", required=True, help="Output path for consolidated JSON")
    p.add_argument("--slim", action="store_true",
                   help="Keep only matcher-essential fields (drops translations + transliterations; ~5x smaller)")
    p.add_argument("--pretty", action="store_true", help="Indent the output (larger file, easier to diff)")
    args = p.parse_args()

    all_shabads: list[dict] = []
    seen: set[int] = set()
    for src in args.src:
        d = Path(src)
        new, dups = load_dir(d, seen)
        print(f"  {d}: +{len(new)} new, {dups} dupes (already seen)")
        all_shabads.extend(new)

    if args.slim:
        all_shabads = [slim_shabad(s) for s in all_shabads]

    # Normalize dict-typed fields to plain strings (source data has nested dicts
    # for transliteration_english and sometimes translation_english)
    for shabad in all_shabads:
        for v in shabad.get("verses", []):
            tl = v.get("transliteration_english")
            if isinstance(tl, dict):
                v["transliteration_english"] = tl.get("en") or tl.get("english") or ""
            tr = v.get("translation_english")
            if isinstance(tr, dict):
                v["translation_english"] = tr.get("en") or tr.get("bdb") or tr.get("english") or ""

    # NOTE: We do NOT sort the output. Order is preserved as filename
    # alphabetical (per dir.glob("*.json") sorted), then concatenated in
    # --src order. This matches the legacy directory loader's iteration so
    # that tie-broken candidate orderings (matcher rapidfuzz when scores
    # match exactly) produce identical results between the two paths.

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(all_shabads, f, ensure_ascii=False, indent=2)
        else:
            json.dump(all_shabads, f, ensure_ascii=False, separators=(",", ":"))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    total_verses = sum(len(s.get("verses", [])) for s in all_shabads)
    print(
        f"\nWrote {out_path}: {len(all_shabads)} shabads, "
        f"{total_verses} verses, {size_mb:.1f} MB"
    )


if __name__ == "__main__":
    main()
