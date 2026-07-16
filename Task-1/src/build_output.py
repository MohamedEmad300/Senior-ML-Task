"""Step E: assemble output/Item_Mapping_Result.xlsx from Steps A-C's outputs
plus gemma4:31b-cloud's adjudication decisions.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from llm_adjudicate import load_cache, normalize_key

APPROACH_NOTES = f"""\
TASK 1 - DATA CLEANING & ITEM MAPPING: APPROACH NOTES

APPROACHES CONSIDERED

1. Exact string match only
   Match pharmacy item names to Item Master by exact (normalized) string equality.
   Trivial to implement, zero LLM/compute cost. Rejected as the sole approach:
   measured recall is only ~2.5% (Pharmacy 2) and ~9.6% (Pharmacy 1) against the
   real data -- pharmacy exports use different naming/abbreviation/ordering
   conventions than the master catalog, so exact match alone misses the vast
   majority of true matches.

2. TF-IDF character n-gram + fuzzy match only
   Retrieve nearest neighbors via char n-gram TF-IDF cosine similarity, rerank
   with rapidfuzz. Cheap, catches typos/reordering/abbreviation well. Rejected as
   the sole retrieval signal: it is a purely lexical/surface signal and misses
   brand<->generic or synonym relationships (e.g. brand name vs. active ingredient,
   English vs. transliterated Arabic) that share no character substrings.

3. Embedding (semantic) retrieval only
   Retrieve nearest neighbors via embeddinggemma cosine similarity only. Catches
   semantic/brand-generic relationships TF-IDF misses, but on this data set typo-
   heavy and reordering-heavy names (very common in the raw exports) can drift the
   embedding away from the correct neighbor in ways char n-grams don't. Rejected
   as the sole retrieval signal for the same reason TF-IDF-only was rejected --
   each signal has a distinct, non-overlapping failure mode.

4. Pure LLM pairwise matching (LLM decides every pharmacy-item x candidate pair,
   or worse, every pharmacy-item x full master-catalog pair)
   Most accurate in principle, but computationally infeasible at this scale
   (60,936 x 5,709 pairs) and unnecessary for the ~9-in-10 items retrieval can
   already resolve confidently and cheaply. Rejected as the sole approach on cost
   grounds -- would turn a few-hundred-batch LLM job into a multi-hundred-
   thousand-call job for no measurable accuracy gain on the easy majority of rows.

5. CHOSEN: Hybrid retrieval (TF-IDF union embeddings) + fuzzy rerank + tiered
   LLM adjudication only on the ambiguous middle tier
   Fit TF-IDF and embeddinggemma retrieval once over the full deduplicated Item
   Master (60,085 unique names), take the union of each signal's top-k candidates
   per pharmacy item (so either signal's strengths cover the other's blind spots),
   rerank the union with rapidfuzz token_sort_ratio, and tier the best score:
   >=90 auto-accepted, 70-89 sent to an LLM for adjudication, <70 marked no-match.
   This balances accuracy and performance as the task explicitly asks: retrieval
   does the cheap bulk work over all 60,936 rows, and LLM calls -- the expensive,
   slow resource -- are reserved for the bounded subset of genuinely ambiguous
   cases, and are cached so repeat runs cost nothing.

MODEL CHOICE WITHIN THE CHOSEN APPROACH

Two Step D configurations were built and compared on the identical ambiguous
pool (see ../Approach_Comparison.md for the full write-up):

- A hybrid design: a fast local model (gemma4:e2b) handled the bulk of
  adjudication calls, escalating to a larger cloud model (gemma4:31b-cloud)
  only when the local model reported low confidence.
- The cloud-only design actually implemented here: every adjudication call
  goes to {config.CLOUD_MODEL}, with no local model and no confidence-based
  escalation tier.

Note: gemma4:31b is an open-weights model, not a proprietary closed one.
It was accessed via Ollama's cloud inference API only because the
available hardware (a laptop GPU with 6GB VRAM) can't run a 31B-parameter
model locally -- running gemma4:e2b (5.1B) locally was already at the edge
of what that hardware could hold. Given adequate VRAM, gemma4:31b could run
entirely on-device with the same prompt and accuracy and no dependency on
network connectivity or cloud usage caps. This was a hardware-availability
workaround, not a preference for cloud/closed inference.

Manually auditing the hybrid run's errors found the local model was
responsible for the large majority of matching mistakes, and specifically
that it was often confidently (not just uncertainly) wrong -- exactly the
failure mode the confidence-based escalation trigger was never designed to
catch, since it only escalates on self-reported LOW confidence. Switching
to the cloud model alone, combined with a prompt rewritten to state explicit
matching rules (brand name is the primary signal, strength must match,
pack-size differences alone don't disqualify a match, reasoning must cite
concrete evidence) rather than a vague "same product" instruction, produced
a measurably better result on a full-scale, side-by-side comparison against
identical inputs -- see ../Approach_Comparison.md for the agreement/
disagreement audit and specific before/after examples.

A third option -- Google Gemini as the Step D provider -- was also
implemented and tested (src_gemini in the full project, not included in
this submission) but could not be run to completion: Gemini's free tier
caps gemma-class models at 20 requests/day per API key, and this workload
needs roughly 1,750 requests, which would require on the order of 90
different free-tier keys (or a paid Gemini account) to complete in a
reasonable timeframe. Ollama's gemma4:31b-cloud, used in the submitted
result, does not have this constraint at the volumes this task required.

KNOWN LIMITATIONS

- Tier cutoffs (90/70) and parse confidence threshold (0.7) are heuristic
  defaults sanity-checked against a small hand-labeled sample, not formally
  cross-validated.
- No escalation tier in the final design means a genuinely hard case has no
  second opinion within a single run -- if {config.CLOUD_MODEL} is wrong,
  it's wrong. The side-by-side comparison against the hybrid run's output
  (Approach_Comparison.md) is the closest substitute for a second opinion
  currently in place.
- Item Master names are extremely noisy free text (brand-only names with no
  parseable dosage form, promotional noise like "///OFFER" and "20%OFF",
  non-pharma merchandise mixed with drug items). Rows the regex parser
  can't confidently field-extract are routed to LLM review rather than
  guessed at.
- Item Master has 851 exact-duplicate ITEM_LOOKUP_NAME rows (1,133 after
  whitespace normalization). The retrieval index is built over the
  deduplicated unique names; the cleaned output sheet keeps all 60,936
  original rows so no source data is dropped.
- Manual audit of a sample of disagreements against the earlier hybrid run
  found the pack-size-tolerance rule may be slightly too lenient in a
  minority of cases (e.g. confirming against a candidate with a different
  size *qualifier*, like "NEWBORN" vs a numbered size, rather than just a
  different pack count) -- see Approach_Comparison.md's Future Work section
  for specific proposed refinements.
"""


def merge_parse_results(item_master_df: pd.DataFrame) -> pd.DataFrame:
    df = item_master_df.copy()
    df["Parse_Source"] = df["Needs_LLM_Review"].map(lambda x: "LLM" if x else "Regex")

    if config.LLM_PARSE_CACHE_PATH.exists():
        cache = load_cache(config.LLM_PARSE_CACHE_PATH)
        field_map = {
            "trade_name": "Trade Name", "dosage_form": "Dosage Form",
            "pack_size": "Pack Size", "unit_of_measure": "Unit of Measure",
            "flavour": "Flavour",
        }
        for idx, row in df[df["Needs_LLM_Review"]].iterrows():
            key = normalize_key(row["ITEM_LOOKUP_NAME"])
            rec = cache.get(key)
            if not rec or rec.get("decision") == "LLM_FAILED":
                df.at[idx, "Parse_Source"] = "LLM_FAILED" if rec else "Unresolved"
                continue
            for src, dst in field_map.items():
                if rec.get(src) is not None:
                    df.at[idx, dst] = rec[src]
    return df


def merge_match_results(matched_df: pd.DataFrame) -> pd.DataFrame:
    df = matched_df.copy()
    df["Final_Status"] = None
    df.loc[df["Tier"] == "HIGH_AUTO_ACCEPT", "Final_Status"] = "AUTO_ACCEPTED"
    df.loc[df["Tier"] == "LOW_NO_MATCH", "Final_Status"] = "NO_MATCH"

    if config.LLM_MATCH_CACHE_PATH.exists():
        cache = load_cache(config.LLM_MATCH_CACHE_PATH)
        for idx, row in df[df["Tier"] == "MEDIUM_LLM_REVIEW"].iterrows():
            key = normalize_key(row["Name_Normalized"])
            rec = cache.get(key)
            if not rec:
                df.at[idx, "Final_Status"] = "Unresolved"
                continue
            decision = rec.get("decision")
            if decision == "LLM_FAILED":
                df.at[idx, "Final_Status"] = "LLM_FAILED"
            elif decision == "NO_MATCH":
                df.at[idx, "Final_Status"] = "LLM_REJECTED"
                df.at[idx, "Best_Match_Name"] = None
            elif decision in ("CONFIRM", "CORRECTED_MATCH"):
                df.at[idx, "Final_Status"] = "LLM_CONFIRMED"
                if rec.get("chosen_match"):
                    df.at[idx, "Best_Match_Name"] = rec["chosen_match"]
    df = df.drop(columns=["Top_Candidates"], errors="ignore")
    return df


def build_summary(item_master_df, p1_df, p2_df) -> pd.DataFrame:
    rows = [
        ("Model", config.CLOUD_MODEL),
        ("", ""),
        ("Item Master total rows", len(item_master_df)),
        ("Item Master unique ITEM_LOOKUP_NAME", item_master_df["ITEM_LOOKUP_NAME"].nunique()),
    ]
    for conf_bucket, label in [(True, "Needs_LLM_Review rows"), (False, "Regex-parsed rows (no LLM)")]:
        n = (item_master_df["Needs_LLM_Review"] == conf_bucket).sum()
        rows.append((label, n))
    for src in item_master_df["Parse_Source"].unique():
        rows.append((f"Parse_Source = {src}", (item_master_df["Parse_Source"] == src).sum()))

    rows.append(("", ""))
    for label, df in [("Pharmacy 1", p1_df), ("Pharmacy 2", p2_df)]:
        rows.append((f"{label} total rows", len(df)))
        for tier in ["HIGH_AUTO_ACCEPT", "MEDIUM_LLM_REVIEW", "LOW_NO_MATCH"]:
            n = (df["Tier"] == tier).sum()
            rows.append((f"{label} tier: {tier}", n))
        for status in df["Final_Status"].dropna().unique():
            n = (df["Final_Status"] == status).sum()
            rows.append((f"{label} final status: {status}", n))

    rows.append(("", ""))
    match_cache = load_cache(config.LLM_MATCH_CACHE_PATH) if config.LLM_MATCH_CACHE_PATH.exists() else {}
    parse_cache = load_cache(config.LLM_PARSE_CACHE_PATH) if config.LLM_PARSE_CACHE_PATH.exists() else {}
    for cache_name, cache in [("Match adjudication", match_cache), ("Parse adjudication", parse_cache)]:
        rows.append((f"{cache_name}: total cached decisions", len(cache)))
        n_failed = sum(1 for r in cache.values() if r.get("decision") == "LLM_FAILED")
        rows.append((f"{cache_name}: LLM_FAILED", n_failed))

    return pd.DataFrame(rows, columns=["Metric", "Value"])


def main():
    print("Loading Step A/C outputs + LLM adjudication decisions ...")
    item_master = pd.read_pickle(config.ITEM_MASTER_PARSED_PATH)
    p1 = pd.read_pickle(config.PHARMACY1_MATCHED_PATH)
    p2 = pd.read_pickle(config.PHARMACY2_MATCHED_PATH)

    print("Merging LLM parse decisions into Item Master ...")
    item_master_final = merge_parse_results(item_master)

    print("Merging LLM match decisions into pharmacy sheets ...")
    p1_final = merge_match_results(p1)
    p2_final = merge_match_results(p2)

    print("Building summary ...")
    summary = build_summary(item_master_final, p1_final, p2_final)

    config.OUTPUT_PATH.parent.mkdir(exist_ok=True)
    print(f"Writing {config.OUTPUT_PATH} ...")
    with pd.ExcelWriter(config.OUTPUT_PATH, engine="openpyxl") as writer:
        item_master_final.to_excel(writer, sheet_name="Item Master - Cleaned", index=False)
        p1_final.to_excel(writer, sheet_name="Pharmacy 1 - Matched", index=False)
        p2_final.to_excel(writer, sheet_name="Pharmacy 2 - Matched", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame({"Approach Notes": APPROACH_NOTES.split("\n")}).to_excel(
            writer, sheet_name="Approach Notes", index=False, header=False
        )

    print("Done.")
    return item_master_final, p1_final, p2_final, summary


if __name__ == "__main__":
    main()
