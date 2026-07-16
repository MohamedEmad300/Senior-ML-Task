"""Purge LLM_FAILED entries from the adjudication caches so the next
run_pipeline.py reprocesses exactly those rows.

Usage:
    python src/retry_failed.py             # purge both caches
    python src/retry_failed.py --match      # scope to the match cache
    python src/retry_failed.py --parse      # scope to the parse cache
    python src/retry_failed.py --dry-run    # report counts only, change nothing
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def purge_failed(cache_path: Path, dry_run: bool):
    if not cache_path.exists():
        return 0, 0
    kept, failed = [], 0
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("decision") == "LLM_FAILED":
                failed += 1
            else:
                kept.append(line)
    total = failed + len(kept)
    if not dry_run and failed:
        backup = cache_path.with_suffix(cache_path.suffix + ".bak")
        shutil.copy2(cache_path, backup)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
        print(f"  backed up pre-purge cache to {backup.name}")
    return failed, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", action="store_true", help="scope to only the match cache")
    parser.add_argument("--parse", action="store_true", help="scope to only the parse cache")
    parser.add_argument("--dry-run", action="store_true", help="report counts only, change nothing")
    args = parser.parse_args()

    only_match = args.match and not args.parse
    only_parse = args.parse and not args.match
    targets = []
    if not only_parse:
        targets.append(("match", config.LLM_MATCH_CACHE_PATH))
    if not only_match:
        targets.append(("parse", config.LLM_PARSE_CACHE_PATH))

    any_failed = False
    for label, path in targets:
        failed, total = purge_failed(path, args.dry_run)
        any_failed = any_failed or failed > 0
        verb = "would purge" if args.dry_run else "purged"
        print(f"{label} cache: {verb} {failed}/{total} LLM_FAILED entries")

    if args.dry_run:
        print("Dry run -- nothing was changed. Drop --dry-run to actually purge.")
    elif any_failed:
        print("Done. Run `python src/run_pipeline.py` again to reprocess "
              "the purged rows -- every other cached decision is left untouched.")
    else:
        print("No LLM_FAILED entries found -- nothing to do.")


if __name__ == "__main__":
    main()
