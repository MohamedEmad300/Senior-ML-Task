"""
"Fold 4" -- May 2026 (202605), evaluated only on items that also appear in
prior months.

202605 was excluded from the modeling window entirely (Phase 3/8-10 finding:
truncated extract, only 1,788 raw sales rows vs ~140-148k for every other
month). A follow-up check found 114 of its 115 items (99.1%) also appear in
prior months -- for those specific items, the raw May total is probably a
real (if early-in-the-month) number, not an artifact of missing rows for
items that were never processed. This fold tests the final models against
that trustworthy subset only.

Construction (train = TimeIndex 1-8, known; test = TimeIndex 9 = 202605,
reconstructed):
  - Actual target: SalesTarget_9 = sum(PositiveSales) from Sales2026_clean.csv
    for DateMonth=202605, per item -- for ALL items appearing that month
    (not just the overlap set; the overlap restriction is applied only when
    scoring, so predictions can be inspected for the 1 new item too).
  - Lag1/2/3, Lag1_NetSales/Returns, Lag1/2/3_Outage: pulled directly from
    each item's known periods 8/7/6 (TimeIndex 8->Lag1, 7->Lag2, 6->Lag3).
    Rolling stats/trend features recomputed with the same formulas as
    Phase 14.
  - Streaks (ConsecutiveZeroMonths, ConsecutivePositiveMonths,
    ConsecutiveOutageMonths, MonthsSinceLastSale, OutageFrequency,
    EverOutaged): computed by walking the known 8-period sequence per item
    backward from period 8 (a closed-form, not an incremental shift, since
    there's no "period 8's own streak" to extend -- period 8's stored
    streak value describes the streak ending at period 7).
  - Availability lifecycle (Segment, ItemAge, CurrentSegmentDuration,
    SegmentTransitionCount, Ever*/MonthsIn*): pulled from
    AvailabilityHistory_clean.csv directly for Date=202605 via the existing
    build_availability_lifecycle() helper -- this file already covers
    202605 (its range is 202307-202606), so no reconstruction needed.
  - Current-period Outage aggregates (TotalOutageDays, MeanOutage, MaxOutage,
    NumWarehousesAffected, PctWarehousesAffected): Outage2026.csv has NO
    records at all for 202605 (its range is 202601-202604) -- these are set
    to 0, consistent with the "missing outage -> 0" convention used
    throughout the pipeline. NumWarehouses carried forward from period 8
    (structural, doesn't change month to month).
"""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb
from engineer_row_level_features import build_availability_lifecycle

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "SalesTarget"
CAT_FEATURES = ["Item", "Segment"]
DROP_COLS = ["Date", "NetSalesTotal", "ReturnsTotal", TARGET]
VOLUME_THRESHOLD = 50
COV_CUTOFF = 0.5
MAY_DATE = 202605
MAY_TIMEINDEX = 9


def business_accuracy_mask(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.round(np.asarray(y_pred, dtype=float))
    zero_mask = y_true == 0
    correct = np.empty(len(y_true), dtype=bool)
    correct[zero_mask] = pred[zero_mask] == 0
    nz = ~zero_mask
    correct[nz] = np.abs(pred[nz] - y_true[nz]) <= 0.2 * y_true[nz]
    return correct


def main():
    df = pd.read_csv(f"{BASE}\\Item_Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    items = sorted(df["Item"].unique())

    print("Reconstructing May 2026 (TimeIndex 9) actuals...")
    s26 = pd.read_csv(f"{BASE}\\Sales2026_clean.csv", dtype={"Item_Code": str})
    may_actual = (
        s26[s26["DateMonth"] == MAY_DATE].groupby("Item_Code")["PositiveSales"].sum()
        .rename("SalesTarget_actual")
    )
    may_items = set(may_actual.index)
    print(f"  {len(may_items)} unique items in May 2026")

    print("Determining overlap with prior months (Sales2025 + Sales2026 excl. May)...")
    s25 = pd.read_csv(f"{BASE}\\Sales2025_clean.csv", dtype={"Item_Code": str})
    prior_items = set(s25["Item_Code"].unique()) | set(
        s26[s26["DateMonth"] != MAY_DATE]["Item_Code"].unique()
    )
    overlap_items = may_items & prior_items
    print(f"  {len(overlap_items)} of {len(may_items)} May items also appear in prior months "
          f"({len(overlap_items)/len(may_items):.1%})")

    # ---- pivot known history (periods 1-8) per item ----
    pivot_sales = df.pivot(index="Item", columns="TimeIndex", values="SalesTarget")
    pivot_net = df.pivot(index="Item", columns="TimeIndex", values="NetSalesTotal")
    pivot_returns = df.pivot(index="Item", columns="TimeIndex", values="ReturnsTotal")
    pivot_outage = df.pivot(index="Item", columns="TimeIndex", values="MeanOutage")

    lag1 = pivot_sales[8]
    lag2 = pivot_sales[7]
    lag3 = pivot_sales[6]
    lag1_net = pivot_net[8]
    lag1_returns = pivot_returns[8]
    lag1_outage = pivot_outage[8]
    lag2_outage = pivot_outage[7]
    lag3_outage = pivot_outage[6]

    rolling_mean2 = pd.concat([lag1, lag2], axis=1).mean(axis=1)
    rolling_mean3 = pd.concat([lag1, lag2, lag3], axis=1).mean(axis=1)
    rolling_std2 = pd.concat([lag1, lag2], axis=1).std(axis=1)
    rolling_std3 = pd.concat([lag1, lag2, lag3], axis=1).std(axis=1)
    rolling_max3 = pd.concat([lag1, lag2, lag3], axis=1).max(axis=1)
    rolling_min3 = pd.concat([lag1, lag2, lag3], axis=1).min(axis=1)
    rolling_median3 = pd.concat([lag1, lag2, lag3], axis=1).median(axis=1)
    momentum = lag1 - lag2
    growth_ratio = lag1 / (lag2 + 1)
    rolling_mean_delta = rolling_mean3 - rolling_mean2
    rolling_outage2 = pd.concat([lag1_outage, lag2_outage], axis=1).mean(axis=1)
    rolling_outage3 = pd.concat([lag1_outage, lag2_outage, lag3_outage], axis=1).mean(axis=1)

    print("Computing streak features ending at period 8 (closed-form backward walk)...")
    zero_flags = pivot_sales == 0
    pos_flags = pivot_sales > 0
    outage_flags = pivot_outage > 0

    def trailing_streak(flags):
        streak = pd.Series(0, index=flags.index)
        still_going = pd.Series(True, index=flags.index)
        for t in range(8, 0, -1):
            col = flags[t].reindex(flags.index).fillna(False) & still_going
            streak = streak + col.astype(int)
            still_going = still_going & flags[t].reindex(flags.index).fillna(False)
        return streak

    consecutive_zero = trailing_streak(zero_flags)
    consecutive_positive = trailing_streak(pos_flags)
    consecutive_outage = trailing_streak(outage_flags)
    outage_frequency = outage_flags.mean(axis=1)
    ever_outaged = outage_flags.any(axis=1).astype(int)

    last_sale_period = pos_flags.apply(lambda row: max([t for t in range(1, 9) if row[t]], default=np.nan), axis=1)
    months_since_last_sale = MAY_TIMEINDEX - last_sale_period

    num_warehouses_period8 = df[df["TimeIndex"] == 8].set_index("Item")["NumWarehouses"]

    print("Pulling Availability lifecycle features for Date=202605...")
    lifecycle = build_availability_lifecycle(set(items), BASE)
    lifecycle_may = lifecycle[lifecycle["Date"] == MAY_DATE].set_index("Item")

    severity_map = {"AVAL": 0, "NEW": 1, "SHTG": 2, "RARE": 3, "ROFF": 4}

    print("Assembling period-9 feature rows...")
    period9 = pd.DataFrame(index=pd.Index(items, name="Item"))
    period9["Date"] = MAY_DATE
    period9["TimeIndex"] = MAY_TIMEINDEX
    period9["Year"] = 2026
    period9["Month"] = 5
    period9["Quarter"] = 2
    period9["MonthSin"] = np.sin(2 * np.pi * 5 / 12)
    period9["MonthCos"] = np.cos(2 * np.pi * 5 / 12)

    period9["Lag1"] = lag1
    period9["Lag2"] = lag2
    period9["Lag3"] = lag3
    period9["Lag1_NetSales"] = lag1_net
    period9["Lag1_Returns"] = lag1_returns
    period9["Lag1_Outage"] = lag1_outage
    period9["Lag2_Outage"] = lag2_outage
    period9["Lag3_Outage"] = lag3_outage
    period9["Lag1_PctWarehousesAffected"] = df[df["TimeIndex"] == 8].set_index("Item")["PctWarehousesAffected"]

    period9["RollingMean2"] = rolling_mean2
    period9["RollingMean3"] = rolling_mean3
    period9["RollingStd2"] = rolling_std2
    period9["RollingStd3"] = rolling_std3
    period9["RollingMax3"] = rolling_max3
    period9["RollingMin3"] = rolling_min3
    period9["RollingMedian3"] = rolling_median3
    period9["Momentum"] = momentum
    period9["GrowthRatio"] = growth_ratio
    period9["RollingMeanDelta_3_2"] = rolling_mean_delta

    period9["ConsecutiveZeroMonths"] = consecutive_zero
    period9["ConsecutivePositiveMonths"] = consecutive_positive
    period9["MonthsSinceLastSale"] = months_since_last_sale
    period9["RollingOutage2"] = rolling_outage2
    period9["RollingOutage3"] = rolling_outage3
    period9["OutageFrequency"] = outage_frequency
    period9["ConsecutiveOutageMonths"] = consecutive_outage
    period9["EverOutaged"] = ever_outaged

    # current-period outage: no raw data exists for 202605 -> 0 (documented assumption)
    period9["TotalOutageDays"] = 0
    period9["MeanOutage"] = 0
    period9["MaxOutage"] = 0
    period9["NumWarehousesAffected"] = 0
    period9["NumWarehouses"] = num_warehouses_period8
    period9["PctWarehousesAffected"] = 0.0
    period9["AnyExpFlagged"] = False

    # lifecycle helper doesn't return Segment directly -- pull from AvailabilityHistory_clean directly
    avail = pd.read_csv(f"{BASE}\\AvailabilityHistory_clean.csv", dtype={"ITEM_CODE": str})
    seg_may = avail[avail["Date"] == MAY_DATE].drop_duplicates("ITEM_CODE").set_index("ITEM_CODE")["Segment"]
    period9["Segment"] = seg_may.reindex(period9.index).fillna("UNKNOWN")
    period9["SegmentSeverity"] = period9["Segment"].map(severity_map)
    period9["HasAvailabilityRecord"] = seg_may.reindex(period9.index).notna().astype(int)

    for col in ["ItemAge", "CurrentSegmentDuration", "SegmentTransitionCount",
                "EverRARE", "EverSHTG", "EverROFF",
                "MonthsInAVAL", "MonthsInRARE", "MonthsInSHTG", "MonthsInROFF"]:
        period9[col] = lifecycle_may[col].reindex(period9.index)

    period9["Outage_x_Segment"] = period9["Lag1_Outage"] * period9["SegmentSeverity"]
    period9["Sales_x_Outage"] = period9["Lag1"] * period9["Lag1_Outage"]
    period9["ItemAge_x_Segment"] = period9["ItemAge"] * period9["SegmentSeverity"]
    period9["HasAvail_x_Segment"] = period9["HasAvailabilityRecord"] * period9["SegmentSeverity"]

    period9 = period9.reset_index()
    period9["SalesTarget_actual"] = period9["Item"].map(may_actual).fillna(0)
    period9["InOverlapSet"] = period9["Item"].isin(overlap_items)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    missing_cols = [c for c in feature_cols if c not in period9.columns]
    assert not missing_cols, f"Missing feature columns: {missing_cols}"

    # ---- restrict to overlap items for the actual fold ----
    eval_set = period9[period9["InOverlapSet"]].copy()
    print(f"\nEvaluation set: {len(eval_set)} items (overlap-only, per business direction)")

    train = df[df["TimeIndex"] <= 8]
    X_train_full = train[feature_cols].copy()
    y_train_raw = train[TARGET].values
    X_eval_full = eval_set[feature_cols].copy()
    y_eval_raw = eval_set["SalesTarget_actual"].values

    with open(f"{BASE}\\best_hyperparams.json") as f:
        cb_params = json.load(f)
    with open(f"{BASE}\\best_hyperparams_lgbm.json") as f:
        lgb_params = json.load(f)
    with open(f"{BASE}\\best_hyperparams_xgb.json") as f:
        xgb_params = json.load(f)

    # ---- CatBoost ----
    Xc_train, Xc_eval = X_train_full.copy(), X_eval_full.copy()
    for c in CAT_FEATURES:
        Xc_train[c] = Xc_train[c].astype(str)
        Xc_eval[c] = Xc_eval[c].astype(str)
    # no eval_set/early_stopping here -- Fold 4 is the held-out evaluation set,
    # and LightGBM/XGBoost below are fit the same way (fixed tuned iterations,
    # no peeking at this fold), so this keeps the three models comparable.
    cb_p = dict(loss_function="MAE", random_seed=42, bootstrap_type="Bayesian",
                verbose=False, **cb_params)
    train_pool = Pool(Xc_train, np.log1p(y_train_raw), cat_features=CAT_FEATURES)
    cb_model = CatBoostRegressor(**cb_p)
    cb_model.fit(train_pool)
    cb_pred = np.clip(np.expm1(cb_model.predict(Xc_eval)), 0, None)

    # ---- LightGBM / XGBoost ----
    Xg_train, Xg_eval = X_train_full.copy(), X_eval_full.copy()
    for c in CAT_FEATURES:
        Xg_train[c] = Xg_train[c].astype("category")
        Xg_eval[c] = Xg_eval[c].astype("category")

    lgb_p = dict(objective="regression_l1", random_state=42, verbosity=-1, **lgb_params)
    lgb_model = lgb.LGBMRegressor(**lgb_p)
    lgb_model.fit(Xg_train, np.log1p(y_train_raw))
    lgb_pred = np.clip(np.expm1(lgb_model.predict(Xg_eval)), 0, None)

    xgb_p = dict(tree_method="hist", enable_categorical=True, objective="reg:absoluteerror",
                 random_state=42, **xgb_params)
    xgb_model = xgb.XGBRegressor(**xgb_p)
    xgb_model.fit(Xg_train, np.log1p(y_train_raw))
    xgb_pred = np.clip(np.expm1(xgb_model.predict(Xg_eval)), 0, None)

    lag1_pred = eval_set["Lag1"].values
    rm2_pred = eval_set["RollingMean2"].values
    rm3_pred = eval_set["RollingMean3"].values

    cov = eval_set["RollingStd3"] / eval_set["RollingMean3"].replace(0, np.nan)
    stable_mask = (eval_set["RollingMean3"].fillna(-1) > VOLUME_THRESHOLD) | (
        (eval_set["RollingMean3"].fillna(-1) > 0) & (cov.fillna(np.inf) < COV_CUTOFF)
    )
    cb_blend = np.where(stable_mask, lag1_pred, cb_pred)
    lgb_blend = np.where(stable_mask, lag1_pred, lgb_pred)
    xgb_blend = np.where(stable_mask, lag1_pred, xgb_pred)

    print()
    print("=" * 78)
    print(f"FOLD 4 (May 2026, {len(eval_set)} overlap items only) -- Business Accuracy")
    print("=" * 78)
    model_preds = {
        "Lag1": lag1_pred, "RollingMean2": rm2_pred, "RollingMean3": rm3_pred,
        "CatBoost": cb_pred, "CatBoost_Blend": cb_blend,
        "LightGBM": lgb_pred, "LightGBM_Blend": lgb_blend,
        "XGBoost": xgb_pred, "XGBoost_Blend": xgb_blend,
    }
    results_rows = []
    sheet_frames = {}
    base_cols = eval_set[["Item", "Date", "TimeIndex"]].copy()
    base_cols["Fold"] = "Fold4_May2026_OverlapOnly"
    base_cols["Actual"] = y_eval_raw
    for name, pred in model_preds.items():
        correct = business_accuracy_mask(y_eval_raw, pred)
        ba = correct.mean() * 100
        mae = np.mean(np.abs(y_eval_raw - np.round(pred)))
        print(f"  {name:16s}  BusinessAcc={ba:6.2f}%  MAE={mae:8.3f}")
        results_rows.append({"Model": name, "BusinessAccuracy": round(ba, 2), "MAE": round(mae, 3),
                              "N_Items": len(eval_set)})
        d = base_cols.copy()
        d["Predicted"] = np.round(pred)
        d["AbsError"] = np.abs(d["Actual"] - d["Predicted"])
        d["Correct"] = correct
        sheet_frames[name] = d

    summary_df = pd.DataFrame(results_rows).sort_values("BusinessAccuracy", ascending=False)

    out_path = f"{BASE}\\Fold4_May2026_Analysis.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for name, d in sheet_frames.items():
            d.to_excel(writer, sheet_name=name[:31], index=False)
        # also include the full period-9 reconstruction (all May items, overlap flag) for transparency
        period9_export = period9[["Item", "InOverlapSet", "SalesTarget_actual", "Lag1", "RollingMean2",
                                   "RollingMean3", "Segment", "ItemAge"]]
        period9_export.to_excel(writer, sheet_name="May2026_AllItems", index=False)

    print()
    print("Saved:", out_path)
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
