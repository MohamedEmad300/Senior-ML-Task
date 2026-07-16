"""
Phase 13 -- Existence + Quantity hurdle model for the warehouse-allocation
stage of the hierarchical forecast (Phase 12).

Confirmed in Phase 12: Model 1 (item-level national demand) alone reaches
65.79% Business Accuracy -- well above the 59.21% Lag1 baseline. But
reconciling through a plain regression on warehouse Share falls back to
59.17%, roughly tied with the naive baseline. Share is itself ~55-58%
exact zero, so the plain regression has to learn existence-vs-magnitude
simultaneously, one level down from the original row-level problem.

Fix (per business direction): split the share stage into
  Model 2a (classifier): P(PositiveSales > 0 | Item, WH, Month)
  Model 2b (regression): E[Share | PositiveSales > 0], trained only on
    nonzero rows, keeping the hierarchical structure (Model 1 already
    supplies the national total, so this predicts conditional share, not
    raw units).
Combined two ways, both reported since the zero-handling in Business
Accuracy makes the choice matter empirically rather than obviously:
  Expectation form (as literally specified):  Final_share = P(sell) * Share_if_sell
  Threshold form: Final_share = 0 if P(sell) < 0.5 else Share_if_sell

Final_pred = round(clip(NationalTotal_pred (Model 1, from Phase 12) *
Final_share, 0, None)).

Same Fold 2 dev split as Phase 11/12 (train TimeIndex 1-6, test TimeIndex 7)
for direct comparability.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from business_accuracy_metrics import eval_metrics

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TRAIN_PERIODS = list(range(1, 7))
TEST_PERIOD = 7
LOSS = "MAE"
SHARE_CAT_FEATURES = ["Item", "WH", "Segment"]

SHARE_FEATURE_COLS = [
    "Item", "WH", "Segment", "Year", "Month", "Quarter", "MonthSin", "MonthCos",
    "SegmentSeverity", "HasAvailabilityRecord", "ItemAge", "CurrentSegmentDuration",
    "Outage", "IsExpFlagged",
    "Lag1", "Lag2", "RollingMean2", "RollingMean3",
    "HistoricalShare_Lag1", "WarehouseVolume_Lag1", "WarehouseAvgSales_Lag1",
    "WarehouseGrowth", "WarehouseItemCount_Lag1",
    "NationalSales_Lag1", "NationalGrowth",
]


def main():
    df = pd.read_csv(f"{BASE}\\Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "WH", "TimeIndex"]).reset_index(drop=True)

    df["_outaged"] = (df["Outage"] > 0).astype(int)
    df["NationalTotal_cur"] = df.groupby(["Item", "TimeIndex"])["PositiveSales"].transform("sum")
    df["Share"] = np.where(df["NationalTotal_cur"] > 0,
                            df["PositiveSales"] / df["NationalTotal_cur"], 0.0)
    df["HistoricalShare_Lag1"] = np.where(df["NationalSales_Lag1"].fillna(0) > 0,
                                           df["Lag1"] / df["NationalSales_Lag1"], 0.0)

    wh_item_count = (
        df[df["PositiveSales"] > 0].groupby(["WH", "TimeIndex"])["Item"].nunique()
        .reset_index(name="WarehouseItemCount")
    )
    full_wh_time = df[["WH", "TimeIndex"]].drop_duplicates()
    wh_item_count = full_wh_time.merge(wh_item_count, on=["WH", "TimeIndex"], how="left")
    wh_item_count["WarehouseItemCount"] = wh_item_count["WarehouseItemCount"].fillna(0)
    wh_item_count = wh_item_count.sort_values(["WH", "TimeIndex"])
    wh_item_count["WarehouseItemCount_Lag1"] = wh_item_count.groupby("WH")["WarehouseItemCount"].shift(1)
    df = df.merge(wh_item_count[["WH", "TimeIndex", "WarehouseItemCount_Lag1"]],
                   on=["WH", "TimeIndex"], how="left")

    train = df[df["TimeIndex"].isin(TRAIN_PERIODS)]
    test = df[df["TimeIndex"] == TEST_PERIOD]
    print(f"Train rows: {len(train)}  Test rows: {len(test)}  "
          f"Train zero-Share share: {(train['Share']==0).mean():.1%}")

    X_train = train[SHARE_FEATURE_COLS].copy()
    X_test = test[SHARE_FEATURE_COLS].copy()
    for c in SHARE_CAT_FEATURES:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    # ---- Model 2a: existence classifier ----
    y_exists_train = (train["PositiveSales"] > 0).astype(int).values
    y_exists_test = (test["PositiveSales"] > 0).astype(int).values

    print()
    print("Training Model 2a (existence classifier)...")
    pool_a_train = Pool(X_train, y_exists_train, cat_features=SHARE_CAT_FEATURES)
    pool_a_test = Pool(X_test, y_exists_test, cat_features=SHARE_CAT_FEATURES)
    model_a = CatBoostClassifier(loss_function="Logloss", random_seed=42, iterations=3000,
                                  early_stopping_rounds=50, verbose=False)
    model_a.fit(pool_a_train, eval_set=pool_a_test)
    p_sell = model_a.predict_proba(X_test)[:, 1]
    print(f"  Classifier test accuracy: {(model_a.predict(X_test).flatten()==y_exists_test).mean():.1%}  "
          f"AUC-ish base rate check -- actual sell rate: {y_exists_test.mean():.1%}")

    # ---- Model 2b: conditional share, nonzero rows only ----
    nz_train_mask = train["PositiveSales"].values > 0
    X_train_nz = X_train.loc[nz_train_mask]
    y_share_nz = train.loc[nz_train_mask, "Share"].values

    print()
    print(f"Training Model 2b (conditional share, {nz_train_mask.sum()} nonzero training rows)...")
    pool_b_train = Pool(X_train_nz, y_share_nz, cat_features=SHARE_CAT_FEATURES)
    model_b = CatBoostRegressor(loss_function=LOSS, random_seed=42, iterations=3000,
                                 verbose=False)
    model_b.fit(pool_b_train)
    share_if_sell = np.clip(model_b.predict(X_test), 0, 1)

    # ---- Combine two ways ----
    final_share_expectation = p_sell * share_if_sell
    final_share_threshold = np.where(p_sell >= 0.5, share_if_sell, 0.0)

    item_level_preds = pd.read_csv(f"{BASE}\\hierarchical_predictions_fold2.csv", dtype={"Item": str})
    national_lookup = item_level_preds[["Item", "TimeIndex", "NationalTotal_pred"]].drop_duplicates()

    test_out = test.copy()
    test_out["Share_expectation"] = final_share_expectation
    test_out["Share_threshold"] = final_share_threshold
    test_out = test_out.merge(national_lookup, on=["Item", "TimeIndex"], how="left")

    test_out["Final_pred_expectation"] = np.clip(
        test_out["NationalTotal_pred"] * test_out["Share_expectation"], 0, None
    )
    test_out["Final_pred_threshold"] = np.clip(
        test_out["NationalTotal_pred"] * test_out["Share_threshold"], 0, None
    )

    print()
    print("=" * 70)
    print("Reconciled hurdle-share forecast vs actual PositiveSales (test=TimeIndex 7)")
    print("=" * 70)
    eval_metrics(test_out["PositiveSales"], test_out["Final_pred_expectation"], "Hurdle_Expectation(P*Share)")
    eval_metrics(test_out["PositiveSales"], test_out["Final_pred_threshold"], "Hurdle_Threshold(0.5 cutoff)")

    print()
    print("For reference:")
    print("  Baseline1_Lag1              BusinessAcc= 59.21%  (from Phase 11a)")
    print("  Model1 alone (national)      BusinessAcc= 65.79%  (from Phase 12)")
    print("  Model1 x plain Share (Ph12)  BusinessAcc= 59.17%  (from Phase 12)")

    test_out.to_csv(f"{BASE}\\hurdle_predictions_fold2.csv", index=False)
    model_a.save_model(f"{BASE}\\catboost_model2a_exists.cbm")
    model_b.save_model(f"{BASE}\\catboost_model2b_share_conditional.cbm")
    print()
    print("Saved: hurdle_predictions_fold2.csv, catboost_model2a_exists.cbm, catboost_model2b_share_conditional.cbm")


if __name__ == "__main__":
    main()
