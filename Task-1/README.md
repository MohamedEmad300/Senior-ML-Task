# Task 1 — Data Cleaning & Item Mapping

## What's in this folder

| File / folder | What it is |
|---|---|
| `Item_Mapping_Result_CloudOnly_RECOMMENDED.xlsx` | **The recommended result.** Full pipeline output using the final, refined approach. |
| `Item_Mapping_Result_Hybrid.xlsx` | An earlier iteration's output, kept for comparison — see `Approach_Comparison.md`. |
| `Approach_Comparison.md` | Full write-up comparing the two result files above: what changed, why, and evidence for which one performs better. |
| `Column_Reference.md` | Explains every column in every sheet of the result workbooks. |
| `src/` | The implementation — Python source for the recommended pipeline, runnable end-to-end. |
| `Item Mapping.xlsx` | The original input file, included so `src/` is self-contained and runnable as-is. |

## The task

Given `Item Mapping.xlsx` (Item Master File + two raw pharmacy exports):
1. Extract structured fields (Trade Name, Dosage Form, Pack Size, Unit of
   Measure, Flavour) from the Item Master's free-text item names.
2. Clean both pharmacy sheets.
3. Match every pharmacy item to its Item Master row.

## Approach considered, then implemented

Five approaches were evaluated before choosing one (full reasoning in each
result workbook's "Approach Notes" sheet, or `src/build_output.py`):

1. **Exact string match only** — rejected, ~2.5-9.6% recall on this data.
2. **TF-IDF character n-gram + fuzzy match only** — rejected as sole signal, misses brand↔generic relationships.
3. **Embedding (semantic) retrieval only** — rejected as sole signal, misses typo/reordering cases TF-IDF catches.
4. **Pure LLM pairwise matching** (every item × every candidate) — rejected on cost grounds, infeasible at 60,936 × 5,709 scale.
5. **Chosen: hybrid retrieval (TF-IDF ∪ embeddings) → fuzzy rerank → tiered LLM adjudication** on only the ambiguous middle tier. Retrieval does the cheap bulk work over all 60,936 rows; an LLM is reserved for the bounded subset of genuinely ambiguous cases.

Within that chosen approach, **two LLM configurations for the adjudication
step were built and compared**:

- **Hybrid**: a fast local model (`gemma4:e2b`) handled most calls, escalating
  to a larger cloud model (`gemma4:31b-cloud`) only on low self-reported
  confidence.
- **Cloud-only (recommended)**: every adjudication call goes to
  `gemma4:31b-cloud`, with a prompt rewritten around explicit matching rules
  after auditing the hybrid run's errors.

Note: `gemma4:31b` is an open-weights model, not a proprietary closed one —
cloud inference was used only because the available hardware (a laptop GPU
with 6GB VRAM) cannot run a 31B-parameter model locally. Given suitable
hardware, this exact model could run fully on-device with no cloud
dependency; the cloud API was a hardware-availability workaround, not an
architectural preference for cloud/closed inference.

Manually auditing the hybrid run's mistakes found the local model was
responsible for the large majority of them, and specifically that it was
often *confidently* wrong — a failure mode the original escalation trigger
(which only fires on low confidence) never caught. The cloud-only
configuration, tested side-by-side on the identical ambiguous pool, resolved
an estimated 90%+ of the resulting disagreements correctly. Full evidence
and methodology in `Approach_Comparison.md`.

## A third option was tried and is not part of this submission: Google Gemini

A Gemini-based version of Step D was also implemented and tested. It could
not be completed: **Gemini's free tier caps `gemini-2.5-flash` at 20
requests/day per API key**, and this workload needs roughly 1,750 requests
total — that would require on the order of **90 different free-tier API
keys** (or a paid Gemini account with materially higher quotas) to finish in
a reasonable timeframe. Since neither was available, this path was set
aside in favor of the Ollama-based cloud model, which had no such
constraint at the volumes this task required. The Gemini implementation
exists in the full project (not included here) if a paid-tier key becomes
available later.

## Running the code

```powershell
cd Deliverable
pip install -r requirements.txt
ollama list        # confirm embeddinggemma:latest and gemma4:31b-cloud are present
python src\run_pipeline.py
```

This reproduces `Item_Mapping_Result_CloudOnly_RECOMMENDED.xlsx` from
scratch. It's a long run — the embedding pass over ~60,000 unique names
takes ~20-30 minutes on first run, and LLM adjudication over the ambiguous
pool (~26,000 items) is the long pole, on the order of a few hours depending
on network/model latency. Every stage is cache-checkpointed, so an
interrupted run resumes without redoing completed work
(`python src\retry_failed.py` clears any failed rows for reprocessing).

## Results summary

| | Hybrid | Cloud-only (recommended) |
|---|---|---|
| Pharmacy 1 — AUTO_ACCEPTED | 1,549 | 1,549 |
| Pharmacy 1 — LLM_CONFIRMED | 2,085 | 2,016 |
| Pharmacy 1 — LLM_REJECTED | 480 | 549 |
| Pharmacy 1 — NO_MATCH | 595 | 595 |
| Pharmacy 2 — AUTO_ACCEPTED | 118 | 118 |
| Pharmacy 2 — LLM_CONFIRMED | 362 | 286 |
| Pharmacy 2 — LLM_REJECTED | 169 | 245 |
| Pharmacy 2 — NO_MATCH | 351 | 351 |
| Item Master — parsed by regex | 37,804 (62.0%) | 37,804 (62.0%) |
| Item Master — parsed by LLM | 23,132 (38.0%) | 23,132 (38.0%) |
| `LLM_FAILED` rows (either pool) | 0 | 0 |

Both runs are fully resolved (no unprocessed/failed rows). See
`Approach_Comparison.md` for why the numbers differ and which is more
accurate.
