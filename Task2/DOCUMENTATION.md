# Sales Forecast — Project Documentation

## Objective

Build a monthly demand forecasting pipeline from five raw CSV extracts (Sales 2025/2026, Outage 2025/2026, Availability History), covering data understanding, cleaning, feature engineering, and model selection, evaluated against a business-defined accuracy metric rather than pure statistical error.

---

## Phase 1 — Data Understanding

Five raw files inspected: `Sales2025.csv`, `Sales2026.csv`, `Outage2025.csv`, `Outage2026.csv`, `AvailabilityHistory.csv`.

| File | Rows | Unique Items | Unique WH | Months | Data quality issues found |
|---|---|---|---|---|---|
| Sales2025 | 574,508 | 18,242 | 22 | 202501–202504 | None (clean) |
| Sales2026 | 578,088 | 14,306 | 22 | 202601–202605 | None (clean) |
| Outage2025 | 562,618 | 28,362 | 22 | 202501–202504 | `Item` column mixed-type (2 rows: `Y90006`) |
| Outage2026 | 577,832 | 28,249 | 22 | 202601–202604 | `Item` column mixed-type (690 rows: `EXP######` codes) |
| AvailabilityHistory | 1,032,349 | 35,256 | n/a | 202307–202606 (36 months) | 29,289 exact duplicate rows; `Segment` values had inconsistent trailing whitespace (`"RARE"` vs `"RARE "`) |

Other findings: all Sales/Outage files share the exact same 22 warehouse codes; Sales has a large share of negative `NET_QTY` (returns-driven, not a data error).

---

## Phase 2 — Cleaning

- **AvailabilityHistory**: dropped 29,289 exact duplicates; stripped `Segment` whitespace (`RARE `→`RARE`, etc.) — 10 raw categories collapsed to the correct 5 (RARE/AVAL/NEW/SHTG/ROFF). **ROFF was kept** (not treated as noise) — it's informative (Removed/Retired/Off-market state).
- **Outage**: cast `Item` to string. `EXP######` codes: stripped the prefix and tested the numeric remainder against the full Sales item universe — **404 rows** (2026) matched a real sold item and were re-keyed with a new `IsExpFlagged=1` column (kept in modeling set); **286 rows** never matched any Sales item and were excluded (audit trail: `Outage2026_excluded.csv`). `Y90006` (2 rows, 2025) never matched → excluded (`Outage2025_excluded.csv`).
- **Sales**: `Item_Code` cast to string. No rows removed for negative sales. Added `NetSales` (=NET_QTY, sign preserved), `Returns` (=RETURNS_QTY, ≤0), `PositiveSales` (=max(NET_QTY,0)) — demand-only signal without discarding the return-precedes-zero-demand information in `NetSales`.

---

## Phase 3–5 — Master Dataset (row-level panel)

Built a full monthly panel indexed by **(Date, Item, WH)** rather than starting from Sales rows (which would silently drop item/warehouse/month combos with zero sales, e.g. from a 100% outage).

- Entity population = union of (Item, WH) pairs seen in **Sales or Outage** (not Sales alone), so fully-outaged combos with zero sales are still real, trackable entities.
- Merge order: Sales (base demand, missing→0) → Outage (left join, missing→0) → Availability (join on Date+Item only — no WH in that file — broadcast to all warehouses, missing→`"UNKNOWN"`).
- Result: 2,813,922 rows initially (9 months) — later rebuilt to 2,500,824 rows (8 months) after the May-2026 discovery below.
- Time axis: `Year`, `Month`, `Quarter`, `MonthSin`/`MonthCos`, and later `TimeIndex` (1..N sequential period index — necessary because the panel has a calendar gap between Apr-2025 and Jan-2026).
- Explicit zero-fill: ~55-60% of panel rows are genuine zero-sales periods that would otherwise be silently missing rows — these are real signal (a "0,0,250" sales pattern is different from "250,260,250").

**Phase 6 — Availability UNKNOWN investigation**: 27.0% of panel rows had `Segment="UNKNOWN"`. Investigated using the full 36-month Availability history (not just the panel window) via forward/backward `merge_asof` per item:

| Case | Count | Meaning | Action |
|---|---|---|---|
| Not yet launched | 170,248 | No record before this date, one exists later | Correctly UNKNOWN, no fix |
| Disappeared/gap | 285,244→188,142(after month exclusion) | Prior record exists, this month missing | **Fixed**: LOCF (carried forward) |
| Never tracked | 305,559→271,592 | Item has zero Availability records ever | Correctly UNKNOWN, no fix possible |

UNKNOWN dropped from 27.0% → 17.7% of the panel after LOCF. Added `HasAvailabilityRecord` (0/1, computed **before** imputation) so models can use the missingness itself as a signal.

---

## Feature Engineering (Stages A–H, row-level)

Built on the (Date, Item, WH) panel, all lag/rolling/streak features computed on `TimeIndex` (not raw calendar date, due to the Apr-2025→Jan-2026 gap):

- **A. Lags**: Lag1/2/3 (PositiveSales), Lag1_NetSales, Lag1_Returns, Lag1/2/3_Outage
- **B. Rolling stats**: RollingMean2/3, RollingStd2/3, RollingMax3, RollingMin3, RollingMedian3 (built from lags, so no leakage)
- **C. Trend**: Momentum, GrowthRatio, RollingMeanDelta_3_2, ConsecutiveZeroMonths, ConsecutivePositiveMonths, MonthsSinceLastSale
- **D. Availability lifecycle** (uses full 36-month history): ItemAge, CurrentSegmentDuration, SegmentTransitionCount, EverRARE/SHTG/ROFF, MonthsInAVAL/RARE/SHTG/ROFF
- **E. Outage**: RollingOutage2/3, OutageFrequency, ConsecutiveOutageMonths, EverOutaged, IsExpFlagged retained
- **F. Aggregates** (item-level & warehouse-level, all **lagged by 1 period** to avoid a warehouse's own current sales leaking into its own "national total" feature): NationalSales_Lag1, WarehousesSellingCount_Lag1, NationalGrowth, NationalRollingMean3, WarehouseVolume_Lag1, WarehouseGrowth, WarehouseAvgSales_Lag1
- **G. Interactions**: Outage_x_Segment, Sales_x_Outage, ItemAge_x_Segment, National_x_Warehouse, HasAvail_x_Segment (via a `SegmentSeverity` ordinal encoding: AVAL=0→ROFF=4)
- **H. Temporal**: TimeIndex (1..N sequential), Month/Quarter/MonthSin/MonthCos

Result: `Feature_Dataset.csv`, 65 columns, one row per (Date, Item, WH).

---

## Phase 8–10 — Baselines, CatBoost #1, Feature Importance (row-level)

Four baselines (Lag1, RollingMean2, RollingMean3, Simple Exponential Smoothing) vs. default-hyperparameter CatBoost, walk-forward evaluated (train on early TimeIndex periods, test on the next).

### Critical discovery: the 9th month (2026-05) was a truncated data extract

CatBoost lost to **every single baseline** on the initial 9-month panel, and its own overfitting detector picked iteration 0 (i.e. every additional tree made test performance worse). Investigation traced this to the test period itself: `Sales2026.csv` month `202605` had only **1,788 raw rows vs. ~140-148k for every other month** — a ~99% shortfall. The resulting test slice was 99.4% zero-valued (vs. 58.9% in train), making every model's holdout evaluation meaningless. This was not feature leakage — the top features (Lag1, RollingMean2/3, etc.) were exactly the sensible ones expected.

**Fix**: excluded 202605 entirely from the modeling window (both train and test) at the source — `build_row_level_panel.py` filters it out before the panel is even built. Modeling window reduced to 8 complete months (202501–202504, 202601–202604). Master/Feature datasets rebuilt.

### Re-evaluation with walk-forward validation (3 expanding-window folds, test on TimeIndex 6/7/8)

| Model | Mean MAE | Mean RMSE | Mean wMAPE |
|---|---|---|---|
| Lag1 | 49.59 | 664.5 | 65.9% |
| RollingMean2 | 50.22 | 724.6 | 66.8% |
| RollingMean3 | 52.95 | 696.2 | 70.4% |
| SES | 55.14 | 574.1 | 73.3% |
| CatBoost (default, RMSE loss) | 70.52 | **455.8** | 93.7% |

CatBoost lost on MAE (default RMSE loss chases rare huge values, hedging with small non-zero predictions on the ~56% zero rows) but won decisively on RMSE/R² (0.31-0.47 vs. mostly-negative baseline R²). This is a genuine loss-function/metric mismatch, not a bug — it's the reason the project moved to a business-defined accuracy metric instead of MAE/RMSE.

Feature importance (built-in + SHAP) top features: `RollingMean3`, `WarehousesSellingCount_Lag1`, `Lag1_NetSales`, `RollingMean2`, `National_x_Warehouse` — no suspicious/leaky feature at the top (`Item`, `Date`, `TimeIndex` all near zero importance), confirming no leakage.

---

## The Business Accuracy Metric

Introduced per business direction: **share of forecasts within ±20% of actual**, chosen as the model-selection criterion over MAE/RMSE from this point forward.

### Documented assumptions (stated explicitly per business direction, not silently decided)

> **Zero-handling**: the spec doesn't define how actual-zero rows are scored under a ±20% rule (a percentage tolerance around zero is undefined). This work assumes **a forecast is correct for an actual-zero row iff the rounded prediction is exactly 0** — no invented slack (e.g. "actual=0, prediction≤5 counts as correct") was applied, since that would make the reported accuracy incomparable to a plain reading of "±20%". **This should be confirmed with stakeholders before deployment.**
>
> **Rounding**: `PositiveSales`/`SalesTarget` is a whole-unit quantity. Continuous model output is rounded to the nearest integer before both the exact-zero check and the ±20% check — a data-type correction, not a business tolerance.

Implemented in `business_accuracy_metrics.py` (`business_accuracy()`, reused by every phase from here on).

---

## Log Target + Loss Function Comparison (Phase 11)

On a dev fold (train TimeIndex 1-6, test 7), compared CatBoost with `log1p(PositiveSales)` target across three loss functions:

| Loss | MAE | Business Accuracy |
|---|---|---|
| RMSE | 46.14 | 51.68% |
| **MAE** | 46.05 | **57.99%** |
| Huber(δ=1.0) | 45.38 | 55.67% |

MAE loss won on Business Accuracy despite RMSE. Reference: Lag1 baseline on this fold = 59.21% — still ahead at this point.

---

## Hierarchical Model Experiment (Phase 12) — confirmed the warehouse-split diagnosis, then set aside

Hypothesis: the row-level model has to solve "how much demand exists" and "which warehouse gets it" simultaneously, and the second part is much noisier.

- **Model 1** (item-level national demand, log1p target): **65.79% Business Accuracy** — dramatically above any row-level result.
- **Model 2** (warehouse share regression) × Model 1, reconciled back to row-level: **59.17%** — essentially tied with Lag1 (59.21%). The warehouse-allocation stage ate all the gain.

This directly proved the item-level signal is real and strong, and the warehouse split is where the difficulty lives.

## Hurdle Model Experiment (Phase 13) — negative result, reported honestly

Tried splitting warehouse allocation into existence (classifier, P(sale>0)) × conditional magnitude (regression on nonzero rows only), per the "two models: will it sell / how much" idea. **Both combination forms underperformed the plain share regression**:

| Approach | Business Accuracy |
|---|---|
| Plain share regression (Phase 12) | 59.17% |
| Hurdle, threshold=0.5 | 55.43% |
| Hurdle, expectation (P×Share) | 50.44% |
| Hurdle, threshold swept to optimum (0.95) | 59.21% (ties Lag1, doesn't beat) |

Reason: chaining 3 independently-fit models compounds error, and the conditional-share model — trained only on nonzero rows — never learned to shrink toward zero the way the plain regression naturally does; Business Accuracy's strict "must round to exactly 0" rule punishes that. **This produced a stray 6.6GB model artifact (`catboost_model2b_share_conditional.cbm`, trained with no early stopping on high-cardinality categoricals) which was deleted after this experiment was abandoned** — it is not part of the final pipeline.

---

## Major Pivot: Item-Month Grain (Phase 14 onward)

Given the hierarchical experiment proved item-level demand is forecastable (65.79% BA) but warehouse allocation isn't, the project pivoted to **forecasting at (Item, Month) grain directly**, folding warehouse information into item-level *supply* features rather than discarding it:

- `SalesTarget` = sum(PositiveSales) across all WH (also NetSalesTotal, ReturnsTotal aggregated)
- Outage folded into: `TotalOutageDays`, `MeanOutage`, `MaxOutage`, `NumWarehousesAffected`, `PctWarehousesAffected`
- Availability/lifecycle features kept as-is (already item-level)
- All Stage A-C/E/G features recomputed on the item-level series (Lag1/2/3, rolling stats, streaks, outage streaks, interactions) — warehouse-specific features (WarehouseVolume_Lag1, HistoricalShare_Lag1, etc.) dropped as no longer applicable

Result: `Item_Feature_Dataset.csv`, 194,736 item-month rows, 24,342 unique items, 62 columns.

### Phase 15 — Baseline check at new grain

| Model | Mean Business Accuracy |
|---|---|
| **Lag1** | **64.60%** |
| RollingMean2 | 59.22% |
| RollingMean3 | 52.04% |
| CatBoost (default) | 57.38% |

Lag1 still dominant — default CatBoost not yet competitive even at the better grain.

### Phase 16 — Optuna tuning (CatBoost)

40-trial search (depth, learning_rate, iterations, l2_leaf_reg, random_strength, bagging_temperature via `bootstrap_type="Bayesian"`), objective = Business Accuracy directly. Best: depth=10, lr=0.215, l2_leaf_reg=14.6, random_strength=0.58, bagging_temperature=3.81, iterations=1320 → **61.23%** (dev fold) — up from 58.39% default, still below Lag1 (65.10% same fold).

### Phase 17 — log1p vs raw target (tuned hyperparameters)

| Target | Business Accuracy |
|---|---|
| Raw | 61.23% |
| **log1p** | **62.99%** |

log1p won; gap to Lag1 narrowed to ~2.1pp.

### Phase 18 — Error analysis by actual-value bucket

| Bucket | % of rows | Lag1 BA | Model BA | Delta |
|---|---|---|---|---|
| 0 | 56.8% | 92.58% | **93.56%** | **+0.98pp** |
| 1-5 | 7.6% | 10.85% | 10.25% | -0.59pp |
| 6-20 | 4.8% | 15.11% | 9.01% | -6.09pp |
| 21-100 | 7.2% | 20.46% | 20.51% | ~tied |
| **100+** | **23.6%** | **40.31%** | **30.44%** | **-9.87pp** |
| Overall | 100% | 65.10% | 62.99% | -2.11pp |

Nearly the entire deficit traced to the **100+ unit/month bucket** (23.6% of rows) — established, high-volume items where naive persistence is already close to optimal, and the ML model was adding regression-to-mean noise.

### Phase 19–20 — Stability blend

Per business direction: framed as a **stability proxy**, not "special-case high volume" — volume (RollingMean3) is a cheap proxy for demand stability, not the claimed underlying mechanism. Rule: route to **Lag1** if `RollingMean3 > 50` OR (`RollingMean3 > 0` AND coefficient-of-variation `RollingStd3/RollingMean3 < 0.5`); otherwise use the tuned model.

Confirmed across all 3 walk-forward folds:

| | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---|---|---|---|
| Lag1 | 64.33% | 65.10% | 64.38% | 64.60% |
| CatBoost tuned alone | 58.12% | 62.99% | 64.11% | 61.74% |
| **CatBoost stability blend** | 64.34% | 65.72% | 65.77% | **65.28%** |

First approach in the project to consistently clear the naive baseline (+0.68pp mean), not just approach it.

### Phase 21 — LightGBM (final resort, per original plan)

Same dataset/folds/metric, 30-trial Optuna tune (num_leaves=161, lr=0.018, n_estimators=382, min_child_samples=96, reg_lambda=21.0), log1p target:

| | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---|---|---|---|
| Lag1 | 64.33% | 65.10% | 64.38% | 64.60% |
| **LightGBM alone** | **65.07%** | **65.97%** | **66.44%** | **65.83%** |
| LightGBM + stability blend | 65.52% | 66.01% | 66.07% | 65.86% |

LightGBM **beat Lag1 outright in every fold without needing the blend crutch** — unlike CatBoost, which needed the blend to clear the baseline at all. The blend adds only marginal, inconsistent value on top (helps Fold 1, roughly neutral elsewhere).

### Phase 22 — XGBoost (confirmatory third model)

Same scope, 30-trial tune (max_depth=11, lr=0.0105, n_estimators=1382, min_child_weight=86, reg_lambda=0.49):

| | Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---|---|---|---|
| **XGBoost alone** | 65.11% | 65.88% | **66.68%** | **65.89%** |
| XGBoost + stability blend | 65.72% | 65.84% | 66.24% | 65.93% |

XGBoost (65.89%) and LightGBM (65.83%) are within 0.06pp — a statistical tie, confirming the gain came from the modeling decisions (item-month grain, log1p target, MAE-family loss, moderate tuning), not the specific GBM library.

---

## Final Model Comparison (all models, mean Business Accuracy across 3 folds)

| Rank | Model | Mean BA |
|---|---|---|
| 1 | XGBoost + stability blend | 65.93% |
| 2 | XGBoost alone | 65.89% |
| 3 | LightGBM + stability blend | 65.86% |
| 4 | LightGBM alone | 65.83% |
| 5 | CatBoost + stability blend | 65.28% |
| 6 | Lag1 baseline | 64.60% |
| 7 | CatBoost alone (no blend) | 61.74% |
| 8 | RollingMean2 baseline | 59.22% |
| 9 | RollingMean3 baseline | 52.04% |

**Recommendation: LightGBM alone** (65.83% BA) — the pragmatic final choice. It beats Lag1 in every fold without relying on a blend fallback, is simpler to deploy than the blend variants, and is statistically indistinguishable from XGBoost (which was tuned second and offers no material advantage). Either LightGBM or XGBoost is defensible; if there's an infrastructure/deployment preference for one library, that should be the deciding factor now, not accuracy.

---

## Ad Hoc Investigation: May 2026 Item Overlap

Follow-up check on the excluded 202605 month: of its 115 unique items, **114 (99.1%) also appear in prior months** (only 1 item, `518692`, is entirely new). These items are heavily skewed toward top sellers (median prior-month total = 10,276 units vs. 97 for the full catalog; average percentile rank 0.81), and the 1,788 rows spread evenly across all 22 warehouses (66-90 rows each). This is consistent with a data pipeline that had only processed/synced the highest-volume, most-established items by the time the extract was pulled — not a corrupted or unrelated item population. Excluding the month from modeling remains correct, but the data itself isn't garbage — just a small, biased partial load.

---

## Fold 4 — May 2026, Evaluated Only on Items Present in Prior Months

A follow-up test, separate from the 3-fold walk-forward validation used for model selection: reconstruct the excluded May 2026 (202605) month as a 4th fold, but score it **only on the 114 items confirmed to also appear in prior months** (the overlap set identified in the earlier ad hoc investigation), since those are the only May figures trusted not to be an artifact of the truncated extract.

### Reconstruction method (`build_may2026_holdout_fold.py`)

TimeIndex 9 (May 2026) doesn't exist in `Item_Feature_Dataset.csv` (excluded at the source, see Phase 3). Rather than rebuild the full pipeline, its feature row was reconstructed directly from each item's known periods 1-8:

- **Lags/rolling/trend**: Lag1/2/3 = periods 8/7/6's `SalesTarget` (and NetSales/Returns/Outage equivalents); rolling stats and trend features recomputed with the same formulas as Phase 14.
- **Streaks** (ConsecutiveZeroMonths, ConsecutivePositiveMonths, ConsecutiveOutageMonths, MonthsSinceLastSale, OutageFrequency, EverOutaged): computed with a closed-form backward walk over the known 8-period sequence per item (period 8's *stored* streak value describes the streak ending at period 7, not period 8, so it can't just be shifted forward).
- **Availability lifecycle** (Segment, ItemAge, CurrentSegmentDuration, SegmentTransitionCount, Ever*/MonthsIn*): pulled directly from `AvailabilityHistory_clean.csv` for `Date=202605` — this file already covers that month (its range is 202307-202606), no reconstruction needed.
- **Current-period Outage aggregates** (TotalOutageDays, MeanOutage, MaxOutage, NumWarehousesAffected, PctWarehousesAffected): `Outage2026.csv` has **zero records** for 202605 (its range is 202601-202604 only) — set to 0, consistent with the "missing outage → 0" convention used throughout. **Documented assumption**: this treats May 2026 as outage-free, which is unverifiable and should be flagged if this fold is ever used for a real decision.
- **Actual target**: `sum(PositiveSales)` from `Sales2026_clean.csv` for `DateMonth=202605`, per item — the raw (truncated) figure, not adjusted.
- Models retrained on TimeIndex 1-8 (all 8 known months), no early stopping/eval_set against this fold for any of the three GBMs (kept consistent across CatBoost/LightGBM/XGBoost to avoid one model getting a peek the others didn't).

### Results (114 overlap items only)

| Model | Business Accuracy | MAE |
|---|---|---|
| **Lag1** | **35.09%** | 3034.0 |
| CatBoost_Blend | 35.09% | 3034.0 |
| LightGBM_Blend | 35.09% | 3034.0 |
| XGBoost_Blend | 35.09% | 3034.0 |
| LightGBM | 30.70% | 3168.7 |
| XGBoost | 29.82% | 3147.3 |
| RollingMean2 | 27.19% | 3017.8 |
| CatBoost | 22.81% | 3717.3 |
| RollingMean3 | 22.81% | 2939.4 |

**Interpretation**:
- All three `_Blend` variants are numerically identical to Lag1 — not a bug. The stability-blend rule routes to Lag1 whenever `RollingMean3 > 50`; since this eval set is *by construction* the 114 highest-volume items in the catalog (median prior-month volume ≈10,276 units, established in the ad hoc investigation), every single item cleared that threshold and 100% got routed to Lag1.
- Every raw (non-blended) ML model loses to Lag1 — independently reconfirming the Phase 18 bucketed error analysis (high-volume/established items favor persistence) on a genuinely different slice of data.
- **Caveat**: absolute accuracy here (35% best) is far below the ~65% seen in Folds 1-3, and that's expected, not evidence the model generalizes poorly. Even the 114 "confirmed real" items' May totals come from only 1,788 rows spread across 114 items × up to 22 warehouses each — most items still aren't fully represented across all their warehouses. The ground truth is very likely an undercount of the item's true full-month total, which penalizes every model in the same direction. The *relative ranking* (Lag1 ≥ blend > raw ML models) is the trustworthy signal from this fold, not the absolute accuracy level.

Saved to **`Fold4_May2026_Analysis.xlsx`** (separate from the main `Model_Predictions_AllFolds.xlsx`): `Summary` sheet, one sheet per model with per-item predictions, and a `May2026_AllItems` sheet showing all 115 May items with an `InOverlapSet` flag for transparency.

---

## File Manifest (execution order for full reproduction)

### Raw inputs (not modified)
`Sales2025.csv`, `Sales2026.csv`, `Outage2025.csv`, `Outage2026.csv`, `AvailabilityHistory.csv`

### Scripts, in run order

| # | Script | Purpose | Key outputs |
|---|---|---|---|
| 1 | `clean_raw_data.py` | Clean all 5 raw files | `Sales2025_clean.csv`, `Sales2026_clean.csv`, `Outage2025_clean.csv`, `Outage2026_clean.csv`, `AvailabilityHistory_clean.csv`, `Outage2025_excluded.csv`, `Outage2026_excluded.csv` |
| 2 | `build_row_level_panel.py` | Build row-level (Date,Item,WH) panel, excludes 202605 | `Master_Dataset.csv` |
| 3 | `fix_availability_segment_gaps.py` | LOCF-fix UNKNOWN Segment, patches in place | `Master_Dataset.csv` (patched), adds `HasAvailabilityRecord` |
| 4 | `engineer_row_level_features.py` | Row-level Stages A-H feature engineering | `Feature_Dataset.csv` |
| 5 | `train_row_level_baselines_and_catboost.py` | Row-level baselines + CatBoost + feature importance (walk-forward) | `feature_importance_builtin.csv`, `feature_importance_shap.csv`, `catboost_model1.cbm` |
| 6 | `business_accuracy_metrics.py` | Shared `business_accuracy()`/`eval_metrics()` — imported by all later scripts | (module, no output) |
| 7 | `compare_log_target_and_loss_functions.py` | Log target + loss function comparison | console only |
| 8 | `train_hierarchical_national_and_share_models.py` | Item-level national demand × warehouse share (exploratory) | `hierarchical_predictions_fold2.csv`, `catboost_model1_national.cbm`, `catboost_model2_share.cbm` |
| 9 | `train_hurdle_existence_and_share_models.py` | Existence+magnitude hurdle for share stage (exploratory, **abandoned**) | `hurdle_predictions_fold2.csv`, `catboost_model2a_exists.cbm` (the 6.6GB `catboost_model2b_share_conditional.cbm` this produced was deleted post-hoc) |
| 10 | `build_item_level_dataset.py` | Rebuild dataset at (Item, Month) grain — the pivot | `Item_Feature_Dataset.csv` |
| 11 | `evaluate_item_level_baselines_and_catboost.py` | Item-level baselines + default CatBoost, walk-forward | console only |
| 12 | `tune_catboost_with_optuna.py` | Optuna tune CatBoost on item-level data | `catboost_item_level_tuned.cbm`, `best_hyperparams.json` |
| 13 | `compare_log_vs_raw_target_tuned.py` | log1p vs raw target, tuned hyperparameters | `catboost_item_level_final.cbm` |
| 14 | `analyze_errors_by_volume_bucket.py` | Bucketed error analysis vs Lag1 | `error_analysis_by_bucket.csv` |
| 15 | `tune_stability_blend_threshold.py` | Stability-proxy threshold sweep, dev fold | `stability_blend_predictions_fold2.csv` |
| 16 | `confirm_stability_blend_walkforward.py` | Confirm CatBoost stability blend across all 3 folds | console only |
| 17 | `tune_and_evaluate_lightgbm.py` | Tune + evaluate LightGBM (final resort) | `best_hyperparams_lgbm.json` |
| 18 | `tune_and_evaluate_xgboost.py` | Tune + evaluate XGBoost (confirmatory) | `best_hyperparams_xgb.json` |
| 19 | `export_all_model_predictions_to_excel.py` | Regenerate full per-row predictions, all models, all folds | `Model_Predictions_AllFolds.xlsx` |
| 20 | `build_may2026_holdout_fold.py` | Reconstruct May 2026 as Fold 4, evaluated on overlap items only | `Fold4_May2026_Analysis.xlsx` |
| 21 | `summarize_fold_fit_counts.py` | Tally exact fit/miss counts (not just %) per model per fold | `fold_fit_counts.csv`, `FoldFitCounts` sheet in `Model_Predictions_AllFolds.xlsx` |

### Key data artifacts

- `Master_Dataset.csv` — row-level (Date, Item, WH) panel, 8 clean months, post-LOCF
- `Feature_Dataset.csv` — row-level, 65 columns (superseded as the final modeling table, but retained for the row-level/hierarchical/hurdle experiments' reproducibility)
- `Item_Feature_Dataset.csv` — **the final modeling table**, (Item, Month) grain, 62 columns
- `best_hyperparams.json` / `best_hyperparams_lgbm.json` / `best_hyperparams_xgb.json` — tuned hyperparameters per model
- `Model_Predictions_AllFolds.xlsx` — per-row predictions, all models, all folds (see below)
- `DOCUMENTATION.md` — this file

### To reproduce from scratch

Run scripts 1-19 in the listed order (each reads the previous step's output via absolute paths hardcoded to the project directory — no relative-path dependencies). Total raw data ≈ 80MB; intermediate `Feature_Dataset.csv` is large (~800MB) since it's row-level (Item×WH×Month); `Item_Feature_Dataset.csv` (~59MB) is the one that matters for the final model.

---

## Excel Deliverables

- **`Model_Predictions_AllFolds.xlsx`** — one sheet per model (`Lag1`, `RollingMean2`, `RollingMean3`, `CatBoost`, `CatBoost_Blend`, `LightGBM`, `LightGBM_Blend`, `XGBoost`, `XGBoost_Blend`), each with per-row predictions across the 3 main walk-forward folds (columns: `Item`, `Date`, `TimeIndex`, `Fold`, `Actual`, `Predicted`, `AbsError`, `Correct`); a `Summary` sheet with Business Accuracy per model per fold and overall mean, sorted best-first; and a `FoldFitCounts` sheet (added by `summarize_fold_fit_counts.py`) giving the exact count of items that fit within ±20% vs. don't, per model per fold — not just the percentage. Also exported standalone as `fold_fit_counts.csv`.
- **`Fold4_May2026_Analysis.xlsx`** — the separate May-2026 overlap-only analysis described above: `Summary` sheet, one sheet per model, and a `May2026_AllItems` sheet with the `InOverlapSet` flag for all 115 May items.
