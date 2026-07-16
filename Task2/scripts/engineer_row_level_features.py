"""
Feature Engineering (Stages A-H) on top of Master_Dataset.csv.
Writes Feature_Dataset.csv. Master_Dataset.csv is untouched.

Sequencing note: the panel has a calendar gap between 2025-04 and 2026-01
(9 sales months total, not consecutive calendar months). All lag/rolling/
streak features are computed on TimeIndex (1..9, Stage H) -- the row order
within each (Item, WH) group -- not on the raw Date, so "last month" always
means "the previous of the 9 observed periods" as intended.

Leakage discipline: every feature that describes "current state" (lags,
rolling stats, streaks, outage frequency, national/warehouse aggregates) is
built from data strictly BEFORE the row's own period (shift(1) or earlier).
The only features that use the current period are lifecycle/availability
features (Stage D) and TimeIndex/calendar features (Stage H), which
describe externally-known state (product age, segment) rather than the
sales target itself, plus IsExpFlagged/Outage which are also exogenous
supply-side signals rather than the sales target.
"""
import numpy as np
import pandas as pd

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"


def month_index(yyyymm):
    return (yyyymm // 100) * 12 + (yyyymm % 100)


def idx_to_yyyymm(idx):
    y, m = divmod(idx - 1, 12)
    return y * 100 + (m + 1)


def streak_len_ending_prev_period(df, group_cols, flag):
    """Length of the run of `flag`==True ending at the PREVIOUS period
    (i.e. excludes the current row -- avoids leakage)."""
    flag_int = flag.astype(int)
    g = df.groupby(group_cols, sort=False)
    flag_shift = g[flag_int.name].shift(1) if flag_int.name in df.columns else None
    # use a temp column so groupby can see it
    tmp_col = "_tmp_flag"
    df[tmp_col] = flag_int
    g = df.groupby(group_cols, sort=False)
    flag_shift = g[tmp_col].shift(1)
    new_group = g.cumcount() == 0
    streak_break = (df[tmp_col] != flag_shift) | new_group
    streak_id = streak_break.cumsum()
    streak_len = df.groupby(streak_id).cumcount() + 1
    consecutive = np.where(df[tmp_col] == 1, streak_len, 0)
    df["_tmp_consec"] = consecutive
    result = df.groupby(group_cols, sort=False)["_tmp_consec"].shift(1)
    df.drop(columns=[tmp_col, "_tmp_consec"], inplace=True)
    return result


def build_availability_lifecycle(items, base):
    avail = pd.read_csv(f"{base}\\AvailabilityHistory_clean.csv", dtype={"ITEM_CODE": str})
    avail = avail.rename(columns={"ITEM_CODE": "Item", "Date": "Date"})
    avail = avail[avail["Item"].isin(items)][["Item", "Date", "Segment"]].drop_duplicates()

    start_idx = month_index(avail["Date"].min())
    end_idx = month_index(avail["Date"].max())
    full_months = [idx_to_yyyymm(i) for i in range(start_idx, end_idx + 1)]

    grid = pd.DataFrame({"Item": sorted(items)}).merge(
        pd.DataFrame({"Date": full_months}), how="cross"
    )
    grid = grid.merge(avail, on=["Item", "Date"], how="left")
    grid["MonthIdx"] = grid["Date"].map(month_index)
    grid = grid.sort_values(["Item", "MonthIdx"]).reset_index(drop=True)

    # LOCF is safe here: this grid is a genuinely consecutive monthly
    # calendar (no gap), unlike the sales panel.
    grid["Segment_filled"] = grid.groupby("Item")["Segment"].ffill()
    grid["HasSegment"] = grid["Segment_filled"].notna()

    first_seen = grid.loc[grid["HasSegment"]].groupby("Item")["MonthIdx"].min()
    grid = grid.merge(first_seen.rename("FirstSeenIdx"), on="Item", how="left")
    grid["ItemAge"] = grid["MonthIdx"] - grid["FirstSeenIdx"]
    grid.loc[grid["ItemAge"] < 0, "ItemAge"] = np.nan

    seg = grid["Segment_filled"]
    seg_shift = grid.groupby("Item")["Segment_filled"].shift(1)
    new_item_start = grid.groupby("Item").cumcount() == 0
    seg_break = (seg != seg_shift) | new_item_start
    grid["_seg_streak_id"] = seg_break.cumsum()
    grid["CurrentSegmentDuration"] = grid.groupby("_seg_streak_id").cumcount() + 1
    grid.loc[~grid["HasSegment"], "CurrentSegmentDuration"] = np.nan

    transition = ((seg != seg_shift) & seg_shift.notna() & seg.notna()).astype(int)
    grid["SegmentTransitionCount"] = transition.groupby(grid["Item"]).cumsum()

    for name in ["RARE", "SHTG", "ROFF"]:
        flag = (grid["Segment_filled"] == name).astype(int)
        grid[f"Ever{name}"] = flag.groupby(grid["Item"]).cummax()

    for name in ["AVAL", "RARE", "SHTG", "ROFF"]:
        flag = (grid["Segment_filled"] == name).astype(int)
        grid[f"MonthsIn{name}"] = flag.groupby(grid["Item"]).cumsum()

    keep = [
        "Item", "Date", "ItemAge", "CurrentSegmentDuration", "SegmentTransitionCount",
        "EverRARE", "EverSHTG", "EverROFF",
        "MonthsInAVAL", "MonthsInRARE", "MonthsInSHTG", "MonthsInROFF",
    ]
    return grid[keep]


def main():
    df = pd.read_csv(f"{BASE}\\Master_Dataset.csv", dtype={"Item": str})

    # ---- Stage H (TimeIndex computed first; needed for correct ordering
    #      across the Apr-2025 -> Jan-2026 calendar gap) ----
    months = sorted(df["Date"].unique())
    time_index_map = {m: i + 1 for i, m in enumerate(months)}
    df["TimeIndex"] = df["Date"].map(time_index_map)

    df = df.sort_values(["Item", "WH", "TimeIndex"]).reset_index(drop=True)
    g = df.groupby(["Item", "WH"], sort=False)

    print("Stage A: lag features...")
    df["Lag1"] = g["PositiveSales"].shift(1)
    df["Lag2"] = g["PositiveSales"].shift(2)
    df["Lag3"] = g["PositiveSales"].shift(3)
    df["Lag1_NetSales"] = g["NetSales"].shift(1)
    df["Lag1_Returns"] = g["Returns"].shift(1)
    df["Lag1_Outage"] = g["Outage"].shift(1)
    df["Lag2_Outage"] = g["Outage"].shift(2)
    df["Lag3_Outage"] = g["Outage"].shift(3)  # supports RollingOutage3 in Stage E

    print("Stage B: rolling statistics (built from lags -> no leakage)...")
    lag_block = df[["Lag1", "Lag2", "Lag3"]]
    df["RollingMean2"] = df[["Lag1", "Lag2"]].mean(axis=1)
    df["RollingMean3"] = lag_block.mean(axis=1)
    df["RollingStd2"] = df[["Lag1", "Lag2"]].std(axis=1)
    df["RollingStd3"] = lag_block.std(axis=1)
    df["RollingMax3"] = lag_block.max(axis=1)
    df["RollingMin3"] = lag_block.min(axis=1)
    df["RollingMedian3"] = lag_block.median(axis=1)

    print("Stage C: trend features...")
    df["Momentum"] = df["Lag1"] - df["Lag2"]
    df["GrowthRatio"] = df["Lag1"] / (df["Lag2"] + 1)
    df["RollingMeanDelta_3_2"] = df["RollingMean3"] - df["RollingMean2"]

    df["ConsecutiveZeroMonths"] = streak_len_ending_prev_period(
        df, ["Item", "WH"], df["PositiveSales"] == 0
    )
    df["ConsecutivePositiveMonths"] = streak_len_ending_prev_period(
        df, ["Item", "WH"], df["PositiveSales"] > 0
    )

    g = df.groupby(["Item", "WH"], sort=False)  # rebuild after temp col churn
    df["_sale_period"] = np.where(df["PositiveSales"] > 0, df["TimeIndex"], np.nan)
    df["_sale_period_shifted"] = g["_sale_period"].shift(1)
    df["_last_sale_period"] = df.groupby(["Item", "WH"])["_sale_period_shifted"].ffill()
    df["MonthsSinceLastSale"] = df["TimeIndex"] - df["_last_sale_period"]
    df.drop(columns=["_sale_period", "_sale_period_shifted", "_last_sale_period"], inplace=True)

    print("Stage D: availability lifecycle features (full 36-month history)...")
    lifecycle = build_availability_lifecycle(set(df["Item"].unique()), BASE)
    df = df.merge(lifecycle, on=["Item", "Date"], how="left")

    print("Stage E: outage features...")
    g = df.groupby(["Item", "WH"], sort=False)
    df["RollingOutage2"] = df[["Lag1_Outage", "Lag2_Outage"]].mean(axis=1)
    df["RollingOutage3"] = df[["Lag1_Outage", "Lag2_Outage", "Lag3_Outage"]].mean(axis=1)

    df["_outage_flag"] = (df["Outage"] > 0).astype(int)
    g = df.groupby(["Item", "WH"], sort=False)
    df["_outage_flag_shifted"] = g["_outage_flag"].shift(1)
    df["_periods_observed_prior"] = g.cumcount()
    df["_outage_flag_shifted_filled"] = df["_outage_flag_shifted"].fillna(0)

    g = df.groupby(["Item", "WH"], sort=False)
    cum_count = g["_outage_flag_shifted_filled"].cumsum()
    df["OutageFrequency"] = cum_count / df["_periods_observed_prior"].replace(0, np.nan)

    df["ConsecutiveOutageMonths"] = streak_len_ending_prev_period(
        df, ["Item", "WH"], df["Outage"] > 0
    )

    g = df.groupby(["Item", "WH"], sort=False)
    df["EverOutaged"] = g["_outage_flag_shifted_filled"].cummax().astype(int)

    df.drop(columns=[
        "_outage_flag", "_outage_flag_shifted", "_periods_observed_prior",
        "_outage_flag_shifted_filled",
    ], inplace=True)
    # IsExpFlagged stays as-is (already in Master_Dataset)

    print("Stage F: national (item) and warehouse aggregate features (lagged)...")
    item_agg = df.groupby(["Item", "TimeIndex"], as_index=False).agg(
        NationalSales=("PositiveSales", "sum"),
        WarehousesSellingCount=("PositiveSales", lambda s: (s > 0).sum()),
    ).sort_values(["Item", "TimeIndex"])
    gi = item_agg.groupby("Item")
    item_agg["NationalSales_Lag1"] = gi["NationalSales"].shift(1)
    item_agg["NationalSales_Lag2"] = gi["NationalSales"].shift(2)
    item_agg["NationalSales_Lag3"] = gi["NationalSales"].shift(3)
    item_agg["WarehousesSellingCount_Lag1"] = gi["WarehousesSellingCount"].shift(1)
    item_agg["NationalGrowth"] = (
        (item_agg["NationalSales_Lag1"] - item_agg["NationalSales_Lag2"])
        / (item_agg["NationalSales_Lag2"] + 1)
    )
    item_agg["NationalRollingMean3"] = item_agg[
        ["NationalSales_Lag1", "NationalSales_Lag2", "NationalSales_Lag3"]
    ].mean(axis=1)

    df = df.merge(
        item_agg[[
            "Item", "TimeIndex", "NationalSales_Lag1", "WarehousesSellingCount_Lag1",
            "NationalGrowth", "NationalRollingMean3",
        ]],
        on=["Item", "TimeIndex"], how="left",
    )

    wh_agg = df.groupby(["WH", "TimeIndex"], as_index=False).agg(
        WarehouseVolume=("PositiveSales", "sum"),
        WarehouseAvgSales=("PositiveSales", "mean"),
    ).sort_values(["WH", "TimeIndex"])
    gw = wh_agg.groupby("WH")
    wh_agg["WarehouseVolume_Lag1"] = gw["WarehouseVolume"].shift(1)
    wh_agg["WarehouseVolume_Lag2"] = gw["WarehouseVolume"].shift(2)
    wh_agg["WarehouseAvgSales_Lag1"] = gw["WarehouseAvgSales"].shift(1)
    wh_agg["WarehouseGrowth"] = (
        (wh_agg["WarehouseVolume_Lag1"] - wh_agg["WarehouseVolume_Lag2"])
        / (wh_agg["WarehouseVolume_Lag2"] + 1)
    )
    df = df.merge(
        wh_agg[["WH", "TimeIndex", "WarehouseVolume_Lag1", "WarehouseGrowth", "WarehouseAvgSales_Lag1"]],
        on=["WH", "TimeIndex"], how="left",
    )

    print("Stage G: interaction features...")
    severity_map = {"AVAL": 0, "NEW": 1, "SHTG": 2, "RARE": 3, "ROFF": 4}
    df["SegmentSeverity"] = df["Segment"].map(severity_map)  # UNKNOWN -> NaN

    df["Outage_x_Segment"] = df["Lag1_Outage"] * df["SegmentSeverity"]
    df["Sales_x_Outage"] = df["Lag1"] * df["Lag1_Outage"]
    df["ItemAge_x_Segment"] = df["ItemAge"] * df["SegmentSeverity"]
    df["National_x_Warehouse"] = df["NationalSales_Lag1"] * df["WarehouseVolume_Lag1"]
    df["HasAvail_x_Segment"] = df["HasAvailabilityRecord"] * df["SegmentSeverity"]

    print("Stage H: finalize temporal features (TimeIndex already built)...")
    # Month/Quarter/MonthSin/MonthCos already present from Master_Dataset

    df = df.sort_values(["Item", "WH", "TimeIndex"]).reset_index(drop=True)

    print()
    print("Final feature dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print()
    print("NaN counts (top 20 by count):")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    out_path = f"{BASE}\\Feature_Dataset.csv"
    df.to_csv(out_path, index=False)
    print()
    print("Wrote feature dataset to:", out_path)


if __name__ == "__main__":
    main()
