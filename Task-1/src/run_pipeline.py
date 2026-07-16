"""Orchestrates the full pipeline: Step A (parse Item Master) -> Steps B1/B2
(clean pharmacy sheets) -> Step C (retrieval + tiering) -> Step D (LLM
adjudication via gemma4:31b-cloud) -> Step E (build output workbook).

Safe to stop and resume: every stage caches its output to cache/*.pkl, and
Step D additionally checkpoints every individual LLM decision to
cache/*.jsonl, so an interrupted run resumes without redoing completed work.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import parse_item_master
import clean_pharmacy1
import clean_pharmacy2
import match_items
import llm_adjudicate
import build_output


def main():
    t0 = time.time()
    config.CACHE_DIR.mkdir(exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)
    config.OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print("=" * 70)
    print("STEP A: parse Item Master")
    print("=" * 70)
    parse_item_master.main()

    print("=" * 70)
    print("STEP B1: clean Pharmacy 1")
    print("=" * 70)
    clean_pharmacy1.main()

    print("=" * 70)
    print("STEP B2: clean Pharmacy 2")
    print("=" * 70)
    clean_pharmacy2.main()

    print("=" * 70)
    print("STEP C: retrieval + fuzzy rerank + tiering")
    print("=" * 70)
    match_items.main()

    print("=" * 70)
    print(f"STEP D: LLM adjudication via {config.CLOUD_MODEL}")
    print("=" * 70)
    llm_adjudicate.main()

    print("=" * 70)
    print("STEP E: build output workbook")
    print("=" * 70)
    build_output.main()

    elapsed = time.time() - t0
    print(f"Pipeline complete in {elapsed/60:.1f} minutes. "
          f"Output: {config.OUTPUT_PATH}")


if __name__ == "__main__":
    main()
