#!/usr/bin/env python3
"""
Transcribe audio via Google Chirp 3 (Speech-to-Text V2 BatchRecognize) with
word-level timestamps. Results are cached to a local JSON so you only pay once.

This is the cheap-data-collection path we used to bulk-transcribe Kirtan audio
for training: upload audio to GCS once, batch-recognise with chirp_3, cache the
result JSON locally, and never hit the API again for the same file.

Setup:
    1. `pip install google-cloud-speech`
    2. `gcloud auth application-default login`
    3. Set your GCP project id:
         export GCP_PROJECT_ID=your-project-id
    4. Upload audio to a bucket you own:
         gsutil cp my_audio.mp3 gs://my-bucket/my_audio.mp3

Usage:
    # Transcribe (hits API, caches result):
    python chirp_transcribe.py \
        --gcs-uri "gs://my-bucket/my_audio.mp3" \
        --cache cache/my_audio.json

    # Just print cached results (no API call):
    python chirp_transcribe.py \
        --cache cache/my_audio.json \
        --print-only

Notes:
    - We use `model="chirp_3"` and `language_codes=["pa-Guru-IN"]` for Punjabi
      in the Gurmukhi script. Swap the language code for other Indic targets.
    - Batch pricing is cheaper than streaming. Chunking long files before
      upload is a cost/reliability choice (smaller retries on quota errors),
      not a quality requirement we measured.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_REGION = "us"  # multi-region; supports chirp_3 + pa-Guru-IN


def transcribe_chirp3(
    gcs_uri: str,
    project_id: str,
    region: str = DEFAULT_REGION,
    language: str = "pa-Guru-IN",
) -> dict:
    """Send audio to Chirp 3 BatchRecognize with word-level timestamps.

    Returns a dict with all results serialised for caching.
    """
    from google.api_core.client_options import ClientOptions
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech

    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{region}-speech.googleapis.com",
        )
    )

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language],
        model="chirp_3",
        features=cloud_speech.RecognitionFeatures(
            enable_word_time_offsets=True,
        ),
    )

    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)

    request = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{project_id}/locations/{region}/recognizers/_",
        config=config,
        files=[file_metadata],
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            inline_response_config=cloud_speech.InlineOutputConfig(),
        ),
    )

    print(f"Sending BatchRecognize request for {gcs_uri} ...")
    print(f"  model=chirp_3  lang={language}  word_timestamps=True")
    t0 = time.time()
    operation = client.batch_recognize(request=request)

    print("Waiting for operation to complete ...")
    response = operation.result(timeout=600)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    transcript_obj = response.results[gcs_uri].transcript
    results = []
    for r in transcript_obj.results:
        alt = r.alternatives[0] if r.alternatives else None
        if alt is None:
            continue
        words = []
        for w in alt.words:
            words.append({
                "word": w.word,
                "start": w.start_offset.total_seconds(),
                "end": w.end_offset.total_seconds(),
                "confidence": round(w.confidence, 4) if w.confidence else None,
            })
        results.append({
            "transcript": alt.transcript,
            "confidence": round(alt.confidence, 4) if alt.confidence else None,
            "language": r.language_code,
            "words": words,
        })

    cache = {
        "gcs_uri": gcs_uri,
        "model": "chirp_3",
        "language_requested": language,
        "elapsed_seconds": round(elapsed, 1),
        "num_results": len(results),
        "results": results,
    }
    return cache


def print_results(cache: dict):
    """Pretty-print word-level results from a cached transcription."""
    results = cache["results"]
    print(f"\n{'=' * 100}")
    print(f"CHIRP 3 TRANSCRIPTION — {cache.get('language_requested', '?')}")
    print(f"  source: {cache.get('gcs_uri', '?')}")
    print(f"  results: {cache.get('num_results', '?')} utterances")
    print(f"  api time: {cache.get('elapsed_seconds', '?')}s")
    print(f"{'=' * 100}")

    total_words = 0
    for i, r in enumerate(results):
        words = r.get("words", [])
        total_words += len(words)
        lang = r.get("language", "?")

        print(f"\n{'─' * 100}")
        print(f"[Utterance {i + 1}]  lang={lang}  words={len(words)}")
        print(f"  {r['transcript']}")

        if words:
            print(f"  {'Word':<30s}  {'Start':>7s}  {'End':>7s}  {'Conf':>6s}")
            print(f"  {'─' * 55}")
            for w in words:
                start = f"{w['start']:.2f}s"
                end = f"{w['end']:.2f}s"
                conf = f"{w['confidence']:.2f}" if w.get("confidence") is not None else "  -"
                print(f"  {w['word']:<30s}  {start:>7s}  {end:>7s}  {conf:>6s}")

    print(f"\n{'=' * 100}")
    print(f"TOTAL: {total_words} words across {len(results)} utterances")
    print(f"{'=' * 100}")


def main():
    parser = argparse.ArgumentParser(
        description="Chirp 3 transcription with caching",
    )
    parser.add_argument("--gcs-uri", help="GCS URI of audio file (gs://...)")
    parser.add_argument("--cache", required=True, help="Path to cache JSON file")
    parser.add_argument("--language", default="pa-Guru-IN",
                        help="BCP-47 language code (default: pa-Guru-IN)")
    parser.add_argument("--project-id", default=os.environ.get("GCP_PROJECT_ID"),
                        help="GCP project id (or set GCP_PROJECT_ID env var)")
    parser.add_argument("--region", default=DEFAULT_REGION,
                        help=f"GCP region (default: {DEFAULT_REGION})")
    parser.add_argument("--print-only", action="store_true",
                        help="Only print cached results, don't call API")
    args = parser.parse_args()

    cache_path = Path(args.cache)

    if args.print_only:
        if not cache_path.exists():
            print(f"ERROR: Cache file not found: {cache_path}", file=sys.stderr)
            sys.exit(1)
        cache = json.load(open(cache_path))
        print_results(cache)
        return

    if cache_path.exists():
        print(f"Cache already exists: {cache_path}")
        resp = input("Re-run API call? (y/N): ").strip().lower()
        if resp != "y":
            cache = json.load(open(cache_path))
            print_results(cache)
            return

    if not args.gcs_uri:
        print("ERROR: --gcs-uri required for transcription", file=sys.stderr)
        sys.exit(1)

    if not args.project_id:
        print("ERROR: --project-id or GCP_PROJECT_ID env var required",
              file=sys.stderr)
        sys.exit(1)

    cache = transcribe_chirp3(
        args.gcs_uri,
        project_id=args.project_id,
        region=args.region,
        language=args.language,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"\nCached to {cache_path} ({cache_path.stat().st_size / 1024:.1f} KB)")

    print_results(cache)


if __name__ == "__main__":
    main()
