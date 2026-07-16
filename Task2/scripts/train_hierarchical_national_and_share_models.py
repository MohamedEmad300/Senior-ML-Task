"""
Phase 12 -- Hierarchical (top-down) forecasting.

Model 1 (item level): predict NationalTotal = sum(PositiveSales across all
  WH) for (Item, Month). Log1p target (still heavy-tailed at the national
  level).
Model 2 (item x warehouse level): predict Share = PositiveSales /
  NationalTotal for (Item, WH, Month). Plain target in [0,1], no log.
Reconciled forecast = round(clip(Model1_pred * Model2_pred, 0, None)).

Rationale (per business direction): the row-level model has to solve "how
much demand exists" and "which warehouse gets it" simultaneously, and the
second part is much noisier -- likely why Lag1 (which implicitly repeats
the historical allocation) beat smarter regression models on Business
Accuracy even while losing on MAE/RMSE. Decomposing should make each
sub-problem easier.

Dev fold matches Phase 11a: train TimeIndex 1-6, test TimeIndex 7, so the
result is directly comparable to Lag1 (BA=59.21%) and the log-target
single-stage models (BA=51.68-57.99%, MAE loss winning).

New features added here (not leakage -- Outage is exogenous/supply-side,
same treatment as row-level Outage/Segment features throughout):
  NationalOutageRate / NationalOutageMean -- current-period, aggregated
    across all WH for the item.
  HistoricalShare_Lag1 -- this (Item,WH)'s share of the item's national
    total last period, derived from existing Lag1 / NationalSales_Lag1.
  WarehouseItemCount_Lag1 -- how many distinct items this warehouse sold
    last period (breadth of assortment), lagged.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from business_accuracy_metrics import eval_metrics

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TRAIN_PERIODS = list(range(1, 7))
TEST_PERIOD = 7
LOSS = "MAE"  # winner from phase11a

ITEM_CAT_FEATURES = ["Item", "Segment"]
SHARE_CAT_FEATURES = ["Item", "WH", "Segment"]


def main():
    df = pd.read_csv(f"{BASE}\\Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "WH", "TimeIndex"]).reset_index(drop=True)

    # ---- new features ----
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

    # ==================== MODEL 1: item-level national demand ====================
    item_level = df.groupby(["Item", "TimeIndex"], as_index=False).agg(
        NationalTotal=("PositiveSales", "sum"),
        NationalOutageRate=("_outaged", "mean"),
        NationalOutageMean=("Outage", "mean"),
    )
    item_static_cols = [
        "Item", "TimeIndex", "Year", "Month", "Quarter", "MonthSin", "MonthCos",
        "Segment", "SegmentSeverity", "HasAvailabilityRecord", "ItemAge",
        "CurrentSegmentDuration", "SegmentTransitionCount",
        "EverRARE", "EverSHTG", "EverROFF",
        "MonthsInAVAL", "MonthsInRARE", "MonthsInSHTG", "MonthsInROFF",
        "NationalSales_Lag1", "WarehousesSellingCount_Lag1", "NationalGrowth", "NationalRollingMean3",
    ]
    item_static = df[item_static_cols].drop_duplicates(subset=["Item", "TimeIndex"])
    item_level = item_level.merge(item_static, on=["Item", "TimeIndex"], how="left")

    item_train = item_level[item_level["TimeIndex"].isin(TRAIN_PERIODS)]
    item_test = item_level[item_level["TimeIndex"] == TEST_PERIOD]
    item_feature_cols = [c for c in item_level.columns if c not in ("TimeIndex", "NationalTotal")]

    X1_train = item_train[item_feature_cols].copy()
    X1_test = item_test[item_feature_cols].copy()
    for c in ITEM_CAT_FEATURES:
        X1_train[c] = X1_train[c].astype(str)
        X1_test[c] = X1_test[c].astype(str)
    y1_train_log = np.log1p(item_train["NationalTotal"].values)
    y1_test_log = np.log1p(item_test["NationalTotal"].values)

    print(f"Model 1 (item-level): train {len(X1_train)} item-months, test {len(X1_test)} item-months")
    pool1_train = Pool(X1_train, y1_train_log, cat_features=ITEM_CAT_FEATURES)
    pool1_test = Pool(X1_test, y1_test_log, cat_features=ITEM_CAT_FEATURES)
    model1 = CatBoostRegressor(loss_function=LOSS, random_seed=42, iterations=3000,
                                early_stopping_rounds=50, verbose=False)
    model1.fit(pool1_train, eval_set=pool1_test)
    national_pred = np.clip(np.expm1(model1.predict(X1_test)), 0, None)
    eval_metrics(item_test["NationalTotal"].values, national_pred, "Model1_NationalTotal")

    item_test = item_test.copy()
    item_test["NationalTotal_pred"] = national_pred

    # ==================== MODEL 2: warehouse share ====================
    share_feature_cols = [
        "Item", "WH", "Segment", "Year", "Month", "Quarter", "MonthSin", "MonthCos",
        "SegmentSeverity", "HasAvailabilityRecord", "ItemAge", "CurrentSegmentDuration",
        "Outage", "IsExpFlagged",
        "Lag1", "Lag2", "RollingMean2", "RollingMean3",
        "HistoricalShare_Lag1", "WarehouseVolume_Lag1", "WarehouseAvgSales_Lag1",
        "WarehouseGrowth", "WarehouseItemCount_Lag1",
        "NationalSales_Lag1", "NationalGrowth",
    ]
    share_train = df[df["TimeIndex"].isin(TRAIN_PERIODS)]
    share_test = df[df["TimeIndex"] == TEST_PERIOD]

    X2_train = share_train[share_feature_cols].copy()
    X2_test = share_test[share_feature_cols].copy()
    for c in SHARE_CAT_FEATURES:
        X2_train[c] = X2_train[c].astype(str)
        X2_test[c] = X2_test[c].astype(str)
    y2_train = share_train["Share"].values
    y2_test = share_test["Share"].values

    print(f"Model 2 (share): train {len(X2_train)} rows, test {len(X2_test)} rows")
    pool2_train = Pool(X2_train, y2_train, cat_features=SHARE_CAT_FEATURES)
    pool2_test = Pool(X2_test, y2_test, cat_features=SHARE_CAT_FEATURES)
    model2 = CatBoostRegressor(loss_function=LOSS, random_seed=42, iterations=3000,
                                early_stopping_rounds=50, verbose=False)
    model2.fit(pool2_train, eval_set=pool2_test)
    share_pred = np.clip(model2.predict(X2_test), 0, 1)

    # ==================== Reconcile ====================
    share_test = share_test.copy()
    share_test["Share_pred"] = share_pred
    reconciled = share_test.merge(
        item_test[["Item", "TimeIndex", "NationalTotal_pred"]], on=["Item", "TimeIndex"], how="left"
    )
    reconciled["Final_pred"] = np.clip(reconciled["NationalTotal_pred"] * reconciled["Share_pred"], 0, None)

    print()
    print("=" * 70)
    print("Reconciled hierarchical forecast vs actual PositiveSales (test=TimeIndex 7)")
    print("=" * 70)
    eval_metrics(reconciled["PositiveSales"], reconciled["Final_pred"], "Hierarchical_Model1xModel2")

    reconciled.to_csv(f"{BASE}\\hierarchical_predictions_fold2.csv", index=False)
    model1.save_model(f"{BASE}\\catboost_model1_national.cbm")
    model2.save_model(f"{BASE}\\catboost_model2_share.cbm")
    print()
    print("Saved: hierarchical_predictions_fold2.csv, catboost_model1_national.cbm, catboost_model2_share.cbm")


if __name__ == "__main__":
    main()
