"""Step D: batched LLM adjudication using gemma4:31b-cloud, covering two
pools --

1. MEDIUM_LLM_REVIEW match candidates from Step C (name + top-3 candidates
   -> CONFIRM / CORRECTED_MATCH / NO_MATCH)
2. Needs_LLM_Review parse rows from Step A (item name -> structured field
   extraction: Trade Name / Dosage Form / Pack Size / Unit / Flavour)

Single model throughout, no local/cloud tiering. An earlier design tried a
fast local model (gemma4:e2b) first, escalating to this cloud model only on
low self-reported confidence. Manual error analysis found the local model
was responsible for most matching mistakes -- and specifically that it was
often confidently wrong in ways the confidence-based escalation trigger
never caught. Switching to the cloud model alone, with a prompt refined to
directly address the observed error patterns (see the rule list in
MATCH_PROMPT_TEMPLATE below), measurably improved accuracy. Full comparison
in ../Approach_Comparison.md.

Network guardrails: gemma4:31b-cloud is routed through ollama.com, so every
call is exposed to connection instability (TLS handshake failures,
connection resets, transient 5xx). Every call is wrapped with retry +
backoff, every failure is logged with the full exception text and batch
offset (not swallowed), and every decision is cache-checkpointed
immediately so an interrupted run resumes without re-paying for completed
work.
"""
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import ollama_utils

MATCH_DECISIONS = {"CONFIRM", "CORRECTED_MATCH", "NO_MATCH"}

MATCH_RESULT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "decision": {"type": "string", "enum": ["CONFIRM", "CORRECTED_MATCH", "NO_MATCH"]},
            "chosen_match": {"type": ["string", "null"]},
            "reasoning": {"type": "string"},
            "self_reported_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["id", "decision", "self_reported_confidence"],
    },
}

PARSE_RESULT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "trade_name": {"type": ["string", "null"]},
            "dosage_form": {"type": ["string", "null"]},
            "pack_size": {"type": ["string", "null"]},
            "unit_of_measure": {"type": ["string", "null"]},
            "flavour": {"type": ["string", "null"]},
            "self_reported_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        },
        "required": ["id", "self_reported_confidence"],
    },
}

MATCH_PROMPT_TEMPLATE = """You are adjudicating pharmacy inventory item names against a master drug/product catalog, to confirm or correct fuzzy-matched candidates.

There are exactly {n_items} input items below. You MUST return exactly {n_items} results, one per input item, matched by "id". Do not stop after the first item. Do not return a single object -- return a JSON array.

For each item, decide whether one of its candidates is the SAME product entry, using these rules in order:

1. BRAND NAME is the primary signal. The chosen candidate's brand/trade name must match the input's brand name closely -- typos, transliteration variants, reordering, and abbreviation differences are fine (e.g. "IVY PAXAL" ~ "IVY PAXAL SYP"). A candidate with a DIFFERENT brand name is NOT a match, even if it treats the same condition or shares an active ingredient (e.g. input "RHEUXICAM" is NOT the same product as candidate "LORNOXICAM" -- those are different brand vs. generic names, do not substitute one for the other). If a candidate is an exact or near-exact string match to the input's brand name, prefer it strongly over a generic-sounding or differently-branded candidate, even if the generic candidate's dosage matches better.
2. STRENGTH/DOSAGE must match if both the input and candidate state one (e.g. "40 MG" vs "20MG" is NOT a match -- different strengths are different products, not formatting variance). If strength is missing from one side, that's not disqualifying by itself.
3. DOSAGE FORM should be compatible (e.g. "TAB" ~ "TABLET" is fine; "CREAM" vs "TABLET" is not).
4. PACK SIZE / QUANTITY differences alone are NOT grounds for rejection. Inventory commonly carries a product in a different pack count/volume than the master catalog entry -- if brand, strength, and form all line up, a pack-size mismatch (e.g. "10 SACHETS" vs "20 SACHETS", "50 GM" vs "100 GM") should still be CONFIRM or CORRECTED_MATCH, not NO_MATCH.
5. Your "reasoning" must name the SPECIFIC evidence (which tokens matched: brand/strength/form), not a vague impression like "known product name" or "plausible match" -- if you can't point to concrete matching tokens beyond a generic ingredient or dosage, choose NO_MATCH instead.
6. If your own reasoning notes a real discrepancy (different brand spelling beyond a minor typo, different strength, no clear evidence), your decision must reflect that -- do not CONFIRM or CORRECTED_MATCH while your reasoning describes why the candidate might not actually match.
7. self_reported_confidence: use "high" only when brand AND strength both clearly match (pack size may still differ). Use "medium" when only brand matches and strength is absent/unclear on one side. Use "low" when you are relying on weak evidence (e.g. a generic-name candidate, a distant spelling match).

If no candidate satisfies these, say NO_MATCH.

Each result must have exactly these fields:
- "id": the input id (integer, copy from input)
- "decision": one of "CONFIRM" (first candidate is correct), "CORRECTED_MATCH" (a different candidate is correct), "NO_MATCH" (none are correct)
- "chosen_match": the exact candidate string you selected (copied character-for-character from that item's candidates list), or null if NO_MATCH
- "reasoning": a short phrase, 10 words or fewer
- "self_reported_confidence": one of "low", "medium", "high"

Input items ({n_items} total):
{items_json}

Respond with ONLY a JSON array of {n_items} results, no other text, no markdown fences."""

PARSE_PROMPT_TEMPLATE = """You are extracting structured fields from pharmacy/drug item names for a master catalog.

There are exactly {n_items} input items below. You MUST return exactly {n_items} results, one per input item, matched by "id". Do not stop after the first item. Do not return a single object -- return a JSON array.

For each item name, extract:
- "trade_name": the brand/product name, with dosage form, pack size, strength, and flavour words removed
- "dosage_form": the pharmaceutical form (e.g. Tablet, Capsule, Syrup, Cream, Gel, Ointment, Drops, Suppository, Ampoule, Vial, Injection, Solution, Lotion, Shampoo, Powder, Spray, Sachet, Patch, Lozenge, Inhaler, Mouthwash), or null if you cannot tell
- "pack_size": the numeric pack size/count/volume if present, or null
- "unit_of_measure": the singular unit matching the dosage form (e.g. "Tablet", "Capsule", "Bottle", "Tube", "Ampoule", "ML", "GM"), or null
- "flavour": the flavour if the name mentions one (e.g. Orange, Mint, Strawberry), or null
- "self_reported_confidence": one of "low", "medium", "high"

Each result must include "id" (copy from input) plus the fields above.

Input items ({n_items} total):
{items_json}

Respond with ONLY a JSON array of {n_items} results, no other text, no markdown fences."""


class RunLogger:
    def __init__(self, name: str):
        config.LOG_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = config.LOG_DIR / f"run_{name}_{ts}.log"
        self.fh = open(self.path, "a", encoding="utf-8")
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.retries = 0
        self.network_errors = 0
        self.processed = 0
        self.total = 0
        self.t_start = time.time()

    def log(self, msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def progress(self):
        elapsed = time.time() - self.t_start
        rate = self.processed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.processed) / rate if rate > 0 else float("nan")
        cache_rate = self.cache_hits / max(1, self.processed + self.cache_hits)
        self.log(
            f"progress {self.processed}/{self.total} "
            f"({self.processed/max(1,self.total):.1%}) | "
            f"calls={self.calls} cache_hit_rate={cache_rate:.1%} "
            f"retries={self.retries} network_errors={self.network_errors} "
            f"failures={self.failures} | "
            f"rate={rate:.2f}/s elapsed={elapsed:.0f}s eta={eta:.0f}s"
        )

    def close(self):
        self.progress()
        self.fh.close()


def normalize_key(text: str) -> str:
    if not isinstance(text, str):
        return ""
    n = text.upper()
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_cache(cache_path: Path) -> dict:
    cache = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def append_cache(cache_path: Path, record: dict):
    config.CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


NETWORK_ERROR_MARKERS = (
    "tls:", "read tcp", "dial tcp", "connection reset", "EOF",
    "context deadline exceeded", "no such host", "wsarecv",
    "connection refused", "timeout", "unauthorized",
)


def _is_network_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker.lower() in msg for marker in NETWORK_ERROR_MARKERS)


def extract_json_array(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
        return None
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def call_cloud_model(client, prompt: str, response_schema, logger: RunLogger, max_attempts=3):
    """Call gemma4:31b-cloud with retry/backoff on network errors specifically
    (distinct from malformed-output retries, which are handled by the caller).
    "Unauthorized" is treated as a network-shaped transient error too -- in
    practice it showed up as a passing hiccup under load, not a real auth
    failure (the same unmodified credentials worked before and after)."""
    backoffs = [3, 8, 20]
    last_exc = None
    for attempt in range(max_attempts):
        try:
            resp = client.generate(model=config.CLOUD_MODEL, prompt=prompt,
                                    format=response_schema, stream=False,
                                    options={"temperature": 0, "num_predict": 4096})
            logger.calls += 1
            return resp["response"]
        except Exception as e:
            last_exc = e
            if _is_network_error(e):
                logger.network_errors += 1
                delay = backoffs[min(attempt, len(backoffs) - 1)]
                logger.log(f"NETWORK ERROR calling {config.CLOUD_MODEL} (attempt {attempt+1}/"
                           f"{max_attempts}): {e}. Backing off {delay}s...")
                time.sleep(delay)
                continue
            raise
    raise last_exc


def adjudicate_pool(pool_items, prompt_template, response_schema, valid_result_fn,
                     client, logger, cache_path, batch_size=None):
    batch_size = batch_size or config.LLM_BATCH_SIZE
    cache = load_cache(cache_path)
    results = {}

    to_process = []
    for item in pool_items:
        if item["key"] in cache:
            results[item["key"]] = cache[item["key"]]
            logger.cache_hits += 1
        else:
            to_process.append(item)

    logger.total = len(pool_items)
    logger.processed = logger.cache_hits
    logger.log(f"{len(pool_items)} items in pool, {logger.cache_hits} already cached, "
               f"{len(to_process)} to process.")

    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start:batch_start + batch_size]
        for i, item in enumerate(batch):
            item["id"] = i
        by_id = {item["id"]: item for item in batch}

        items_json = json.dumps(
            [{k: v for k, v in item.items() if k != "key"} for item in batch],
            ensure_ascii=False
        )
        prompt = prompt_template.format(items_json=items_json, n_items=len(batch))

        parsed = None
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                raw = call_cloud_model(client, prompt, response_schema, logger)
                candidate = extract_json_array(raw)
                if (candidate is not None and len(candidate) == len(batch)
                        and all(isinstance(e, dict) and e.get("id") in by_id
                                and valid_result_fn(e, by_id[e["id"]]) for e in candidate)):
                    parsed = candidate
                    break
                logger.retries += 1
                logger.log(f"WARNING: malformed/incomplete response at offset {batch_start} "
                           f"(attempt {attempt + 1}); raw: {str(raw)[:300]!r}")
            except Exception as e:
                logger.retries += 1
                logger.log(f"WARNING: {config.CLOUD_MODEL} call failed at offset {batch_start} "
                           f"(attempt {attempt + 1}) after retries/backoff exhausted: {e}")

        if parsed is None:
            logger.failures += len(batch)
            logger.log(f"FAILURE: batch at offset {batch_start} could not be adjudicated "
                       f"after {config.LLM_MAX_RETRIES + 1} attempts; marking "
                       f"{len(batch)} item(s) LLM_FAILED.")
            for item in batch:
                rec = {"key": item["key"], "id": item["id"], "decision": "LLM_FAILED",
                       "model_used": config.CLOUD_MODEL, "timestamp": datetime.now().isoformat()}
                results[item["key"]] = rec
                append_cache(cache_path, rec)
                logger.processed += 1
            continue

        for entry in parsed:
            item = by_id[entry["id"]]
            rec = dict(entry)
            if rec.get("decision") == "NO_MATCH":
                rec["chosen_match"] = None
            rec["key"] = item["key"]
            rec["model_used"] = config.CLOUD_MODEL
            rec["timestamp"] = datetime.now().isoformat()
            results[item["key"]] = rec
            append_cache(cache_path, rec)
            logger.processed += 1

        logger.progress()

    return results


def _valid_match_entry(e: dict, item: dict) -> bool:
    if not (
        isinstance(e, dict) and "id" in e
        and e.get("decision") in MATCH_DECISIONS
        and "self_reported_confidence" in e
    ):
        return False
    if e["decision"] in ("CONFIRM", "CORRECTED_MATCH"):
        if e.get("chosen_match") not in item.get("candidates", []):
            return False
    return True


def _valid_parse_entry(e: dict, item: dict) -> bool:
    return isinstance(e, dict) and "id" in e and "self_reported_confidence" in e


def adjudicate_match_pool(medium_df, client, logger):
    pool_items = []
    for idx, row in medium_df.iterrows():
        key = normalize_key(row["Name_Normalized"])
        candidates = [c["name"] for c in row["Top_Candidates"]]
        raw_name = row.get("Item_Name")
        if pd.isna(raw_name):
            raw_name = row.get("Product English Name")
        pool_items.append({"key": key, "raw_name": raw_name, "candidates": candidates})

    return adjudicate_pool(pool_items, MATCH_PROMPT_TEMPLATE, MATCH_RESULT_SCHEMA,
                            _valid_match_entry, client, logger, config.LLM_MATCH_CACHE_PATH)


def adjudicate_parse_pool(review_df, client, logger):
    pool_items = [
        {"key": normalize_key(row["ITEM_LOOKUP_NAME"]), "raw_name": row["ITEM_LOOKUP_NAME"]}
        for _, row in review_df.iterrows()
    ]
    return adjudicate_pool(pool_items, PARSE_PROMPT_TEMPLATE, PARSE_RESULT_SCHEMA,
                            _valid_parse_entry, client, logger, config.LLM_PARSE_CACHE_PATH)


def load_artifacts():
    """Load Steps A/C's outputs (produced by parse_item_master.py,
    clean_pharmacy1.py, clean_pharmacy2.py, match_items.py -- run those
    first) with explicit integrity checks: confirms each file loads, row
    counts match the known-correct values, and expected columns are
    present. Any mismatch raises loudly rather than silently propagating
    bad/partial data into Step D."""
    problems = []

    for path, label in [
        (config.ITEM_MASTER_PARSED_PATH, "item_master_parsed.pkl"),
        (config.PHARMACY1_MATCHED_PATH, "pharmacy1_matched.pkl"),
        (config.PHARMACY2_MATCHED_PATH, "pharmacy2_matched.pkl"),
    ]:
        if not path.exists():
            raise SystemExit(
                f"Missing artifact {label} at {path}.\n"
                f"Run Steps A-C first: parse_item_master.py, clean_pharmacy1.py, "
                f"clean_pharmacy2.py, match_items.py (or just run_pipeline.py, "
                f"which runs everything in order)."
            )

    try:
        item_master = pd.read_pickle(config.ITEM_MASTER_PARSED_PATH)
    except Exception as e:
        raise SystemExit(f"FAILED to load {config.ITEM_MASTER_PARSED_PATH}: {e}")
    try:
        p1 = pd.read_pickle(config.PHARMACY1_MATCHED_PATH)
    except Exception as e:
        raise SystemExit(f"FAILED to load {config.PHARMACY1_MATCHED_PATH}: {e}")
    try:
        p2 = pd.read_pickle(config.PHARMACY2_MATCHED_PATH)
    except Exception as e:
        raise SystemExit(f"FAILED to load {config.PHARMACY2_MATCHED_PATH}: {e}")

    if len(item_master) != 60936:
        problems.append(f"item_master_parsed.pkl has {len(item_master)} rows, expected 60936")
    if len(p1) != 4709:
        problems.append(f"pharmacy1_matched.pkl has {len(p1)} rows, expected 4709")
    if len(p2) != 1000:
        problems.append(f"pharmacy2_matched.pkl has {len(p2)} rows, expected 1000")
    for name, df, col in [("item_master_parsed", item_master, "Needs_LLM_Review"),
                           ("pharmacy1_matched", p1, "Tier"),
                           ("pharmacy2_matched", p2, "Tier")]:
        if col not in df.columns:
            problems.append(f"{name}.pkl is missing expected column {col!r}")

    if problems:
        msg = "Artifact integrity check FAILED:\n  " + "\n  ".join(problems)
        raise SystemExit(msg)

    print("Artifact integrity check passed "
          f"(item_master={len(item_master)}, pharmacy1={len(p1)}, pharmacy2={len(p2)}).")
    return item_master, p1, p2


def main():
    import ollama

    item_master, p1, p2 = load_artifacts()

    client = ollama.Client(host=config.OLLAMA_HOST)
    ollama_utils.unload_other_models(config.CLOUD_MODEL)

    medium = pd.concat([
        p1[p1["Tier"] == "MEDIUM_LLM_REVIEW"],
        p2[p2["Tier"] == "MEDIUM_LLM_REVIEW"],
    ], ignore_index=False)
    print(f"MEDIUM_LLM_REVIEW pool: {len(medium)} rows")

    match_logger = RunLogger("match_adjudication")
    adjudicate_match_pool(medium, client, match_logger)
    match_logger.close()

    review_rows = item_master[item_master["Needs_LLM_Review"]]
    print(f"Needs_LLM_Review (parse) pool: {len(review_rows)} rows")

    parse_logger = RunLogger("parse_adjudication")
    adjudicate_parse_pool(review_rows, client, parse_logger)
    parse_logger.close()

    print("Done.")


if __name__ == "__main__":
    main()
