# Senior-ML Tasks Repo

Two independent data tasks: a data-cleaning/entity-matching pipeline (Task 1)
and a monthly sales-forecasting pipeline (Task 2). Each lives in its own
folder with its own README and full documentation; this file is the map and
the short version of "how it was solved."

```
repo/
├── Task-1/     <- Data Cleaning & Item Mapping
└── Task2/      <- Next-Month Sales Forecasting
```

---

## Task 1 — Data Cleaning & Item Mapping

### The brief

> Evaluate the data-cleaning workflow, including the use of an LLM where
> appropriate, and select the most suitable approach while balancing
> performance and accuracy. Outline the possible approaches considered, then
> implement only one.
>
> The provided file has three sheets: an **Item Master File** (needs
> structured fields — Trade Name, Dosage Form, Unit of Measure, Pack Size —
> extracted from its free-text `Item Name` column, e.g. `Panadol Extra 24
> Tablets` → Trade Name `Panadol Extra`, Dosage Form `Tablets`, Pack Size
> `24`, Unit of Measure `Tablet`), and two **raw pharmacy sheets** that need
> cleaning, then matching against the Item Master.

### Approaches considered

Five were evaluated before choosing one (full reasoning in
[`Task-1/README.md`](./Task-1/README.md) and each result workbook's
"Approach Notes" sheet):

| # | Approach | Verdict |
|---|---|---|
| 1 | Exact string match | Rejected — 2.5-9.6% recall |
| 2 | TF-IDF character n-gram + fuzzy match, alone | Rejected as sole signal — misses brand↔generic relationships |
| 3 | Embedding (semantic) retrieval, alone | Rejected as sole signal — misses typo/reordering cases TF-IDF catches |
| 4 | Pure LLM pairwise matching (every item × every candidate) | Rejected — infeasible at 60,936 × 5,709 scale |
| 5 | **Hybrid retrieval (TF-IDF ∪ embeddings) → fuzzy rerank → tiered LLM adjudication** | **Chosen** |

### What was implemented

Retrieval does the cheap bulk work over all 60,936 Item Master rows; an LLM
is reserved only for the genuinely ambiguous middle tier (3,096 items) —
high-confidence fuzzy matches (≥90) auto-accept, low-confidence ones (<70)
auto-reject, and only the middle band goes to adjudication.

1. **Parse** `Item Name` into Trade Name / Dosage Form / Pack Size / Unit of
   Measure / Flavour — regex first (62.0% of rows), LLM fallback for the
   irregular remainder (38.0%).
2. **Clean** both pharmacy sheets.
3. **Match** each pharmacy item to its Item Master row via hybrid retrieval
   → fuzzy rerank → tiered LLM adjudication.

Two LLM configurations were built and compared for the adjudication step:

- **Hybrid**: local `gemma4:e2b` handles most calls, escalating to cloud
  `gemma4:31b-cloud` on low self-reported confidence.
- **Cloud-only (recommended)**: every adjudication call goes to
  `gemma4:31b-cloud`, with a prompt rewritten around explicit matching rules
  after auditing the hybrid run's errors.

`gemma4:31b` is an open-weights model — cloud inference was used only
because the available hardware (6GB VRAM laptop GPU) can't run a
31B-parameter model locally, not an architectural preference for closed/cloud
models. A third option, Google Gemini, was implemented and tested but
dropped: its free tier caps `gemini-2.5-flash` at 20 requests/day, and this
workload needs ~1,750 requests (would require ~90 free-tier keys).

### Which one won, and why

A manual audit of all 517 items where the two configurations disagreed
(full breakdown in
[`Task-1/Approach_Comparison.md`](./Task-1/Approach_Comparison.md)) found
**cloud-only correct in an estimated 90%+ of disagreements**:

- Hybrid's main failure mode: confidently wrong brand-substitution errors
  (e.g. confirming `DESA 5 MG 20 TAB` as a match for `ALLEVO 5 MG 20 TAB`) —
  a confidently-wrong pattern the original low-confidence-only escalation
  trigger never caught.
- Cloud-only's small new risk: a slightly-too-relaxed pack-size tolerance on
  ~12-20 of 186 items.

**Result: `Item_Mapping_Result_CloudOnly_BestResult.xlsx`** is the
recommended deliverable. Both runs are fully resolved — 0 unprocessed/failed
rows in either pool.

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
| Item Master parsed — regex / LLM | 37,804 (62.0%) / 23,132 (38.0%) | same |

### Folder contents

| File / folder | What it is |
|---|---|
| `Item Mapping.xlsx` | Original input (Item Master + 2 raw pharmacy sheets) |
| `Item_Mapping_Result_CloudOnly_BestResult.xlsx` | **Recommended result** |
| `Item_Mapping_Result_Hybrid.xlsx` | Earlier iteration, kept for comparison |
| `Approach_Comparison.md` | Full hybrid-vs-cloud-only audit and evidence |
| `Column_Reference.md` | Meaning of every output column |
| `src/` | Runnable pipeline (entry point: `run_pipeline.py`) |

### Running it

```powershell
cd Task-1
pip install -r requirements.txt
ollama list        # confirm embeddinggemma:latest and gemma4:31b-cloud are present
python src\run_pipeline.py
```

Long run: embedding pass ~20-30 min, LLM adjudication over ~26,000 items
takes a few hours depending on network/model latency. Every stage is
cache-checkpointed, so an interrupted run resumes without redoing completed
work (`python src\retry_failed.py` clears failed rows for reprocessing).

---

## Task 2 — Next-Month Sales Forecasting

### The brief

> Given sales data for 2025/2026, outage data for 2025/2026, and an
> availability history file, build a prediction model to forecast next
> month's sales, accounting for item seasonality, availability, and other
> relevant factors. **Accuracy** = forecast within ±20% absolute error of
> actual (e.g. forecast 10 → actuals 8-12 are accurate, anything outside is
> inaccurate).

### Data understanding & cleaning

Five raw files (`Sales2025/2026`, `Outage2025/2026`, `AvailabilityHistory`)
were inspected, then cleaned:

- **AvailabilityHistory**: dropped 29,289 exact duplicates, fixed
  inconsistent `Segment` whitespace (10 raw categories → correct 5).
- **Outage**: `EXP######` codes tested against the Sales item universe —
  404 rows re-keyed with a new `IsExpFlagged` column; 286 non-matching rows
  excluded to an audit-trail CSV.
- **Sales**: added `NetSales`, `Returns`, `PositiveSales` without discarding
  return-driven negative values.

A monthly panel was built at **(Date, Item, Warehouse)** grain — populated
from the union of Sales/Outage entities (not Sales alone), so fully-outaged
zero-sales periods remain real, trackable rows rather than silently missing
ones.

### Critical discovery: a truncated month

The 9th month (2026-05) turned out to be a truncated data extract — 1,788
raw rows vs. ~140-148k for every other month, causing CatBoost's holdout
evaluation on it to be meaningless. **Fix**: excluded 202605 from the
modeling window entirely, at the source. A later ad-hoc check confirmed 114
of that month's 115 items also appear in prior months (skewed toward top
sellers), so a separate Fold 4 was reconstructed later to evaluate on just
those trustworthy items.

### The business accuracy metric

From this point, model selection used **Business Accuracy** (share of
forecasts within ±20% of actual) instead of MAE/RMSE — chosen because a
model can win on RMSE while losing on the business's actual definition of
"correct" (documented in
[`Task2/DOCUMENTATION.md`](./Task2/DOCUMENTATION.md), along with the
zero-handling and rounding assumptions made explicit for stakeholder
sign-off).

### The pivot that mattered: row-level → item-month grain

Row-level (Date × Item × Warehouse) models never beat the naive "repeat last
month" (Lag1) baseline. A hierarchical experiment isolated why: item-level
demand alone was highly forecastable (65.79% BA), but the *warehouse
allocation* stage ate all the gain (59.17% BA, ties Lag1). A follow-up hurdle
model (existence × magnitude) confirmed the allocation step, not the demand
signal, was the bottleneck — and underperformed further.

**Decision**: pivot to forecasting directly at **(Item, Month)** grain,
folding warehouse information into item-level supply features
(`TotalOutageDays`, `PctWarehousesAffected`, etc.) instead of predicting
per-warehouse shares.

### Approaches tried and abandoned (reported honestly, not hidden)

- Hierarchical national-demand × warehouse-share split — proved the
  diagnosis, then superseded by the item-month pivot.
- Hurdle (existence classifier × conditional magnitude) for warehouse share —
  underperformed the plain share regression in every variant tried.
- CatBoost alone at item-month grain — never cleared the Lag1 baseline
  without a rule-based "stability blend" fallback (route high-volume/stable
  items to Lag1, everything else to the model).

### Final model selection

At item-month grain, log1p target, MAE-family loss, walk-forward validated
across 3 expanding-window folds:

| Rank | Model | Mean Business Accuracy |
|---|---|---|
| 1 | XGBoost + stability blend | 65.93% |
| 2 | XGBoost alone | 65.89% |
| 3 | LightGBM + stability blend | 65.86% |
| 4 | **LightGBM alone (recommended)** | **65.83%** |
| 5 | CatBoost + stability blend | 65.28% |
| 6 | Lag1 baseline | 64.60% |
| 7 | CatBoost alone | 61.74% |
| 8 | RollingMean2 baseline | 59.22% |
| 9 | RollingMean3 baseline | 52.04% |

**Recommendation: LightGBM alone** — it beats the naive baseline in every
single fold with no fallback logic needed, and is statistically tied with
XGBoost (65.83% vs 65.89%, within noise), confirming the gain came from the
modeling decisions (grain, target transform, loss function) rather than the
specific GBM library.

A separate Fold 4 (reconstructed May-2026, scored only on the 114
confirmed-real items) showed every raw ML model losing to Lag1 on that
high-volume-skewed slice — consistent with, not contradicting, the main
result: high-volume/established items favor persistence, which is exactly
what the stability blend already routes to Lag1.

### Folder contents

| File / folder | What it is |
|---|---|
| `data/` | Full lineage: raw → cleaned → row-level panel → featured → final item-month table |
| `scripts/` | 21 scripts, flat, in numbered execution order (see `Task2/README.md`) |
| `outputs/` | `Model_Predictions_AllFolds.xlsx`, tuned hyperparameters, feature importances, error analysis |
| `DOCUMENTATION.md` | Full phase-by-phase narrative, every decision and dead end |

### Running it

```powershell
cd Task2
pip install pandas numpy catboost lightgbm xgboost optuna openpyxl
# data/ already contains every intermediate file — jump straight to modeling, e.g.:
python scripts\tune_and_evaluate_lightgbm.py
```

To regenerate from raw CSVs instead, delete the derived files from `data/`
and run the scripts in the order listed in
[`Task2/README.md`](./Task2/README.md). Note each script has `BASE`
hardcoded to a `data/` path — update it if this folder is moved.

---

## Common thread across both tasks

Both follow the same discipline: **enumerate multiple approaches before
picking one, implement the chosen approach fully, and report negative
results (abandoned models, rejected configurations) as explicitly as
positive ones** rather than only presenting the winning path.
