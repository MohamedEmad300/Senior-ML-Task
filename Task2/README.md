# Task 2 — Sales Forecast: Deliverable Package

## What this is

A monthly demand forecasting pipeline built from 5 raw CSV extracts (Sales 2025/2026, Outage 2025/2026, Availability History): data cleaning → panel construction → feature engineering → model selection, evaluated against a business-defined **Business Accuracy** metric (share of forecasts within ±20% of actual).

**Final recommendation: LightGBM**, forecasting at **(Item, Month)** grain (not Item×Warehouse), tuned with Optuna, log1p target, `MAE` loss. Mean Business Accuracy = **65.83%** across 3 walk-forward folds, vs. **64.60%** for a naive "repeat last month" baseline — and it beats that baseline in every single fold without needing any fallback logic. XGBoost is a statistical tie (65.89%) and is documented as an equally valid choice; CatBoost needed a rule-based blend with the naive baseline to compete at all.

**Start here → [`DOCUMENTATION.md`](./DOCUMENTATION.md)** for the full narrative: every phase, every decision, every dead end (including two exploratory approaches that were tried and abandoned), all documented assumptions, and the complete results tables.

---

## Folder structure

```
Task2_Deliverable/
├── README.md                 <- you are here
├── DOCUMENTATION.md           <- full project narrative, decisions, results, assumptions
├── scripts/                   <- all code, in the order described below
├── data/                      <- self-contained copy of every CSV the scripts read/write
└── outputs/                   <- key result artifacts (see below)
```

This package is **self-contained** for the final recommended pipeline: `data/` holds the raw extracts, cleaned files, and the final (Item, Month) modeling table, and every script's `BASE` path points at `Task2_Deliverable/data`, not the original project folder. You can copy this whole folder elsewhere and it will still run (only if the folder is renamed or moved, update `BASE` in each script to match the new path).

## `data/` contents (~207MB total)

| File | Stage |
|---|---|
| `Sales2025.csv`, `Sales2026.csv`, `Outage2025.csv`, `Outage2026.csv`, `AvailabilityHistory.csv` | Raw inputs (untouched) |
| `Sales2025_clean.csv`, `Sales2026_clean.csv`, `Outage2025_clean.csv`, `Outage2026_clean.csv`, `AvailabilityHistory_clean.csv`, `Outage2025_excluded.csv`, `Outage2026_excluded.csv` | Output of `clean_raw_data.py` |
| `Item_Feature_Dataset.csv` | **The final modeling table**, (Item, Month) grain — output of `build_item_level_dataset.py` |

**Not included, to keep this package a reasonable size**: `Master_Dataset.csv` (~212MB, the row-level Date×Item×WH panel) and `Feature_Dataset.csv` (~800MB, its fully-featured version). These are only consumed by the row-level *exploratory* scripts (`train_row_level_baselines_and_catboost.py`, `train_hierarchical_national_and_share_models.py`, `train_hurdle_existence_and_share_models.py`) — **not** by the final recommended model. If you need to run those specific scripts, regenerate the two files first by running `build_row_level_panel.py` → `fix_availability_segment_gaps.py` → `engineer_row_level_features.py` in order (inputs for all three are already present in `data/`).

Also not included (large, superseded diagnostic outputs from the two abandoned exploratory approaches): `hierarchical_predictions_fold2.csv`, `hurdle_predictions_fold2.csv`, `stability_blend_predictions_fold2.csv`. These remain in the original project folder if needed.

## `outputs/` contents

| File | What it is |
|---|---|
| `Model_Predictions_AllFolds.xlsx` | Per-row predictions for all 9 final models × all 3 walk-forward folds, one sheet per model, a `Summary` sheet ranked by Business Accuracy, and a `FoldFitCounts` sheet with exact fit/miss counts (not just percentages) per model per fold |
| `fold_fit_counts.csv` | Same fit/miss counts as the `FoldFitCounts` sheet, as a standalone CSV |
| `Fold4_May2026_Analysis.xlsx` | Separate analysis: the excluded, truncated May-2026 month, evaluated only on the 114 items confirmed to also appear in prior months |
| `best_hyperparams.json` | Optuna-tuned CatBoost hyperparameters |
| `best_hyperparams_lgbm.json` | Optuna-tuned LightGBM hyperparameters (the final recommended model) |
| `best_hyperparams_xgb.json` | Optuna-tuned XGBoost hyperparameters |
| `feature_importance_builtin.csv` / `feature_importance_shap.csv` | Feature importance from the row-level CatBoost baseline (Phase 8-10) |
| `error_analysis_by_bucket.csv` | Per-item, bucketed error analysis (Phase 18) that identified the high-volume weak spot |

## `scripts/` — what's in here and why

All 21 scripts are flat in one folder (not split into subfolders) because several of them import from each other directly by module name (e.g. `build_item_level_dataset.py` imports a helper function from `engineer_row_level_features.py`, and most later scripts import `business_accuracy()` from `business_accuracy_metrics.py`) — Python resolves these by co-location, so keeping them together avoids having to rewrite import paths. Filenames are descriptive of what each script does; the table below gives execution order (no numeric prefixes on the filenames themselves, since Python module names can't start with a digit).

**Important**: every script has `BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"` hardcoded at the top, pointing at the `data/` folder described below. If you move this package, update `BASE` in each script to the new location first.

| Order | Script | Stage |
|---|---|---|
| 1 | `clean_raw_data.py` | Clean the 5 raw files (dedup, whitespace, EXP/Y outage code handling, NetSales/PositiveSales/Returns) |
| 2 | `build_row_level_panel.py` | Build the row-level (Date, Item, WH) panel; excludes the truncated 2026-05 month |
| 3 | `fix_availability_segment_gaps.py` | LOCF-fix `UNKNOWN` Segment values using the full 36-month Availability history |
| 4 | `engineer_row_level_features.py` | Row-level feature engineering (lags, rolling stats, streaks, lifecycle, aggregates, interactions) |
| 5 | `train_row_level_baselines_and_catboost.py` | Row-level baselines + first CatBoost model + feature importance — **this is where the truncated-month bug was found** |
| 6 | `business_accuracy_metrics.py` | Shared `business_accuracy()` metric — imported by every script from step 7 onward |
| 7 | `compare_log_target_and_loss_functions.py` | log1p target + loss function comparison |
| 8 | `train_hierarchical_national_and_share_models.py` | *Exploratory*: item-level demand × warehouse-share hierarchical model |
| 9 | `train_hurdle_existence_and_share_models.py` | *Exploratory, abandoned*: existence-classifier + magnitude-regression hurdle for the warehouse-share stage (underperformed; superseded by the pivot below) |
| 10 | `build_item_level_dataset.py` | **The pivot**: rebuild the dataset at (Item, Month) grain |
| 11 | `evaluate_item_level_baselines_and_catboost.py` | Item-level baselines + default CatBoost |
| 12 | `tune_catboost_with_optuna.py` | Optuna-tune CatBoost |
| 13 | `compare_log_vs_raw_target_tuned.py` | log1p vs. raw target, tuned hyperparameters |
| 14 | `analyze_errors_by_volume_bucket.py` | Bucketed error analysis — found the high-volume weak spot |
| 15 | `tune_stability_blend_threshold.py` | Stability-proxy blend (route stable/high-volume items to the naive baseline) |
| 16 | `confirm_stability_blend_walkforward.py` | Confirm the blend across all 3 folds |
| 17 | `tune_and_evaluate_lightgbm.py` | Tune + evaluate LightGBM (**final recommended model**) |
| 18 | `tune_and_evaluate_xgboost.py` | Tune + evaluate XGBoost (confirmatory third model) |
| 19 | `export_all_model_predictions_to_excel.py` | Regenerate full per-row predictions → `Model_Predictions_AllFolds.xlsx` |
| 20 | `build_may2026_holdout_fold.py` | Reconstruct May 2026 as a 4th fold → `Fold4_May2026_Analysis.xlsx` |
| 21 | `summarize_fold_fit_counts.py` | Tally exact fit/miss counts (not just %) per model per fold → `fold_fit_counts.csv` + `FoldFitCounts` sheet in `Model_Predictions_AllFolds.xlsx` |

## Requirements

Python 3.12, `pandas`, `numpy`, `catboost`, `lightgbm`, `xgboost`, `optuna`, `openpyxl`.

## To reproduce from scratch

`Item_Feature_Dataset.csv` is already in `data/`, so every final-pipeline script (steps 10-21 above) runs directly, no regeneration needed. To rebuild it from raw, or to run the row-level exploratory scripts (steps 5, 8, 9), run the scripts in order starting from step 1 — each reads the previous step's output from `BASE` and writes back into `data/`.

Full rationale for every decision along the way — including two approaches that were tried and explicitly abandoned, the documented business-accuracy assumptions, and the complete fold-by-fold results — is in [`DOCUMENTATION.md`](./DOCUMENTATION.md).
