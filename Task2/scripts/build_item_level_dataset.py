"""
Phase 1-2 (new plan) -- Rebuild the dataset and features at (Item, Month)
grain instead of (Item, WH, Month).

Rationale: Phase 12 showed item-level national demand alone reaches 65.79%
Business Accuracy, well above anything achieved at the warehouse-allocation
level (all attempts landed at or below Lag1's 59.21%). Rather than keep
fighting the warehouse-split noise, forecast at item-month grain directly.

Warehouse information is not discarded -- it's folded into item-level
*supply* features (outage breadth/severity across warehouses), since that
describes a real supply-side constraint on national demand even though
per-warehouse allocation itself is dropped as a modeling target.

Reuses:
  - build_availability_lifecycle() and streak_len_ending_prev_period() from
    phase7_feature_engineering.py (both already operate per-Item; no change
    needed).
  - Item-level static columns (Segment, ItemAge, lifecycle counters) are
    pulled from the existing row-level Feature_Dataset.csv via
    drop_duplicates, since they're already identical across WH rows for a
    given (Item, Date) -- avoids recomputing and risking divergence.
"""
import numpy as np
import pandas as pd
from engineer_row_level_features import streak_len_ending_prev_period

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"


def main():
    df = pd.read_csv(f"{BASE}\\Feature_Dataset.csv", dtype={"Item": str})

    print("Aggregating to (Item, Month)...")
    item_static = df[[
        "Item", "Date", "TimeIndex", "Year", "Month", "Quarter", "MonthSin", "MonthCos",
        "Segment", "SegmentSeverity", "HasAvailabilityRecord", "ItemAge",
        "CurrentSegmentDuration", "SegmentTransitionCount",
        "EverRARE", "EverSHTG", "EverROFF",
        "MonthsInAVAL", "MonthsInRARE", "MonthsInSHTG", "MonthsInROFF",
    ]].drop_duplicates(subset=["Item", "Date"])

    item_agg = df.groupby(["Item", "Date"], as_index=False).agg(
        SalesTarget=("PositiveSales", "sum"),
        NetSalesTotal=("NetSales", "sum"),
        ReturnsTotal=("Returns", "sum"),
        TotalOutageDays=("Outage", "sum"),
        MeanOutage=("Outage", "mean"),
        MaxOutage=("Outage", "max"),
        NumWarehouses=("WH", "nunique"),
        NumWarehousesAffected=("Outage", lambda s: (s > 0).sum()),
        AnyExpFlagged=("IsExpFlagged", "max"),
    )
    item_agg["PctWarehousesAffected"] = item_agg["NumWarehousesAffected"] / item_agg["NumWarehouses"]

    item_df = item_agg.merge(item_static, on=["Item", "Date"], how="left")
    item_df = item_df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    print(f"Item-month rows: {len(item_df)}  Unique items: {item_df['Item'].nunique()}")

    g = item_df.groupby("Item", sort=False)

    print("Lag features...")
    item_df["Lag1"] = g["SalesTarget"].shift(1)
    item_df["Lag2"] = g["SalesTarget"].shift(2)
    item_df["Lag3"] = g["SalesTarget"].shift(3)
    item_df["Lag1_NetSales"] = g["NetSalesTotal"].shift(1)
    item_df["Lag1_Returns"] = g["ReturnsTotal"].shift(1)
    item_df["Lag1_Outage"] = g["MeanOutage"].shift(1)
    item_df["Lag2_Outage"] = g["MeanOutage"].shift(2)
    item_df["Lag3_Outage"] = g["MeanOutage"].shift(3)
    item_df["Lag1_PctWarehousesAffected"] = g["PctWarehousesAffected"].shift(1)

    print("Rolling statistics...")
    lag_block = item_df[["Lag1", "Lag2", "Lag3"]]
    item_df["RollingMean2"] = item_df[["Lag1", "Lag2"]].mean(axis=1)
    item_df["RollingMean3"] = lag_block.mean(axis=1)
    item_df["RollingStd2"] = item_df[["Lag1", "Lag2"]].std(axis=1)
    item_df["RollingStd3"] = lag_block.std(axis=1)
    item_df["RollingMax3"] = lag_block.max(axis=1)
    item_df["RollingMin3"] = lag_block.min(axis=1)
    item_df["RollingMedian3"] = lag_block.median(axis=1)

    print("Trend features...")
    item_df["Momentum"] = item_df["Lag1"] - item_df["Lag2"]
    item_df["GrowthRatio"] = item_df["Lag1"] / (item_df["Lag2"] + 1)
    item_df["RollingMeanDelta_3_2"] = item_df["RollingMean3"] - item_df["RollingMean2"]

    item_df["ConsecutiveZeroMonths"] = streak_len_ending_prev_period(
        item_df, ["Item"], item_df["SalesTarget"] == 0
    )
    item_df["ConsecutivePositiveMonths"] = streak_len_ending_prev_period(
        item_df, ["Item"], item_df["SalesTarget"] > 0
    )

    g = item_df.groupby("Item", sort=False)
    item_df["_sale_period"] = np.where(item_df["SalesTarget"] > 0, item_df["TimeIndex"], np.nan)
    item_df["_sale_period_shifted"] = g["_sale_period"].shift(1)
    item_df["_last_sale_period"] = item_df.groupby("Item")["_sale_period_shifted"].ffill()
    item_df["MonthsSinceLastSale"] = item_df["TimeIndex"] - item_df["_last_sale_period"]
    item_df.drop(columns=["_sale_period", "_sale_period_shifted", "_last_sale_period"], inplace=True)

    print("Outage features...")
    item_df["RollingOutage2"] = item_df[["Lag1_Outage", "Lag2_Outage"]].mean(axis=1)
    item_df["RollingOutage3"] = item_df[["Lag1_Outage", "Lag2_Outage", "Lag3_Outage"]].mean(axis=1)

    item_df["_outage_flag"] = (item_df["MeanOutage"] > 0).astype(int)
    g = item_df.groupby("Item", sort=False)
    item_df["_outage_flag_shifted"] = g["_outage_flag"].shift(1)
    item_df["_periods_observed_prior"] = g.cumcount()
    item_df["_outage_flag_shifted_filled"] = item_df["_outage_flag_shifted"].fillna(0)

    g = item_df.groupby("Item", sort=False)
    cum_count = g["_outage_flag_shifted_filled"].cumsum()
    item_df["OutageFrequency"] = cum_count / item_df["_periods_observed_prior"].replace(0, np.nan)
    item_df["ConsecutiveOutageMonths"] = streak_len_ending_prev_period(
        item_df, ["Item"], item_df["MeanOutage"] > 0
    )
    g = item_df.groupby("Item", sort=False)
    item_df["EverOutaged"] = g["_outage_flag_shifted_filled"].cummax().astype(int)
    item_df.drop(columns=[
        "_outage_flag", "_outage_flag_shifted", "_periods_observed_prior", "_outage_flag_shifted_filled",
    ], inplace=True)

    print("Interaction features...")
    item_df["Outage_x_Segment"] = item_df["Lag1_Outage"] * item_df["SegmentSeverity"]
    item_df["Sales_x_Outage"] = item_df["Lag1"] * item_df["Lag1_Outage"]
    item_df["ItemAge_x_Segment"] = item_df["ItemAge"] * item_df["SegmentSeverity"]
    item_df["HasAvail_x_Segment"] = item_df["HasAvailabilityRecord"] * item_df["SegmentSeverity"]

    print()
    print("Final item-month feature dataset shape:", item_df.shape)
    print("Columns:", list(item_df.columns))
    print()
    print("Zero-share of SalesTarget:", (item_df["SalesTarget"] == 0).mean())

    out_path = f"{BASE}\\Item_Feature_Dataset.csv"
    item_df.to_csv(out_path, index=False)
    print()
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
