"""
Phase 3 -- Build the Master Dataset
Phase 4 -- Fill missing (Item, WH, Month) combinations with explicit zeros
Phase 5 -- Time axis features

Builds a full monthly panel indexed by (DateMonth, Item_Code, WH) and merges
Sales, Outage, and Availability onto it -- rather than starting from Sales
rows and left-joining onto them (which would silently drop any
item/warehouse/month combo that had zero sales, e.g. because of a 100%
outage, and corrupt lag features downstream).

Population of (Item_Code, WH) pairs:
  Union of pairs seen in Sales AND Outage (not Sales alone). An item can be
  fully out of stock at a warehouse for the whole modeling window (zero
  sales) yet still appear in Outage -- that combo is a real, trackable
  entity and its zero-sales months are genuine zeros, not "doesn't exist".

Modeling window:
  All DateMonths present in the cleaned Sales files EXCEPT 202605, which is
  a truncated/incomplete extract (1,788 raw sales rows vs ~140-148k for
  every other month -- confirmed during Phase 8-10 model evaluation, where
  it produced a test set that was 99.4% zero and made every baseline and
  CatBoost look broken). Modeling window = 202501-202504, 202601-202604
  (8 complete months). Outage/Availability months outside this window are
  irrelevant since we are forecasting sales.
"""
EXCLUDED_MONTHS = [202605]
import numpy as np
import pandas as pd

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"


def main():
    print("Loading cleaned files...")
    s25 = pd.read_csv(f"{BASE}\\Sales2025_clean.csv", dtype={"Item_Code": str})
    s26 = pd.read_csv(f"{BASE}\\Sales2026_clean.csv", dtype={"Item_Code": str})
    sales = pd.concat([s25, s26], ignore_index=True)[
        ["DateMonth", "Item_Code", "WH", "NetSales", "Returns", "PositiveSales"]
    ]
    sales = sales[~sales["DateMonth"].isin(EXCLUDED_MONTHS)]

    o25 = pd.read_csv(f"{BASE}\\Outage2025_clean.csv", dtype={"Item_Code": str})
    o26 = pd.read_csv(f"{BASE}\\Outage2026_clean.csv", dtype={"Item_Code": str})
    outage = pd.concat([o25, o26], ignore_index=True)[
        ["DateMonth", "Item_Code", "WH", "Outage", "IsExpFlagged"]
    ]
    # collapse any duplicate (Date,Item,WH) that could arise from EXP re-keying
    # colliding with an existing numeric row for the same period
    outage = (
        outage.groupby(["DateMonth", "Item_Code", "WH"], as_index=False)
        .agg(Outage=("Outage", "max"), IsExpFlagged=("IsExpFlagged", "max"))
    )

    avail = pd.read_csv(f"{BASE}\\AvailabilityHistory_clean.csv", dtype={"ITEM_CODE": str})
    avail = avail.rename(columns={"ITEM_CODE": "Item_Code", "Date": "DateMonth"})

    # ---- modeling window: months actually present in Sales ----
    months = sorted(sales["DateMonth"].unique())
    print("Modeling window months:", months)

    # ---- entity population: (Item_Code, WH) seen in Sales OR Outage ----
    sales_pairs = sales[["Item_Code", "WH"]].drop_duplicates()
    outage_pairs = outage[["Item_Code", "WH"]].drop_duplicates()
    pairs = pd.concat([sales_pairs, outage_pairs], ignore_index=True).drop_duplicates()
    print("Unique (Item_Code, WH) entities:", len(pairs))

    months_df = pd.DataFrame({"DateMonth": months})
    print("Building full panel grid (entities x months)...")
    master = pairs.merge(months_df, how="cross")
    print("Master grid rows:", len(master))

    # ---- 1. Sales: base demand ----
    master = master.merge(sales, on=["DateMonth", "Item_Code", "WH"], how="left")
    for col in ["NetSales", "Returns", "PositiveSales"]:
        master[col] = master[col].fillna(0).astype(int)

    # ---- 2. Outage: left join, missing -> 0 ----
    master = master.merge(outage, on=["DateMonth", "Item_Code", "WH"], how="left")
    master["Outage"] = master["Outage"].fillna(0).astype(int)
    master["IsExpFlagged"] = master["IsExpFlagged"].astype("boolean").fillna(False).astype(bool)

    # ---- 3. Availability: join on (DateMonth, Item_Code) only ----
    #     no WH in Availability -> every warehouse gets the same state
    avail_small = avail[["DateMonth", "Item_Code", "Segment"]].drop_duplicates()
    master = master.merge(avail_small, on=["DateMonth", "Item_Code"], how="left")
    master["Segment"] = master["Segment"].fillna("UNKNOWN")

    # ---- Phase 5: time axis ----
    master["Year"] = master["DateMonth"] // 100
    master["Month"] = master["DateMonth"] % 100
    master["Quarter"] = ((master["Month"] - 1) // 3) + 1
    master["MonthSin"] = np.sin(2 * np.pi * master["Month"] / 12)
    master["MonthCos"] = np.cos(2 * np.pi * master["Month"] / 12)

    master = master.rename(columns={"DateMonth": "Date", "Item_Code": "Item"})
    master = master.sort_values(["Item", "WH", "Date"]).reset_index(drop=True)

    col_order = [
        "Date", "Year", "Month", "Quarter", "MonthSin", "MonthCos",
        "Item", "WH", "NetSales", "PositiveSales", "Returns",
        "Outage", "IsExpFlagged", "Segment",
    ]
    master = master[col_order]

    print()
    print("Master dataset shape:", master.shape)
    print(master.dtypes)
    print()
    print("Missing values:")
    print(master.isna().sum())
    print()
    print("Segment distribution (incl. UNKNOWN = item absent from AvailabilityHistory):")
    print(master["Segment"].value_counts())
    print()
    rows_with_zero_fill = (master["NetSales"] == 0).sum()
    print("Rows where NetSales had to be filled/observed as 0:", rows_with_zero_fill,
          f"({rows_with_zero_fill / len(master):.1%} of panel)")

    # spot check the example from the brief: item 516965, WH 11
    check = master[(master["Item"] == "516965") & (master["WH"] == 11)]
    print()
    print("Spot check Item 516965 / WH 11 across all panel months:")
    print(check[["Date", "NetSales", "PositiveSales", "Outage", "Segment"]].to_string(index=False))

    out_path = f"{BASE}\\Master_Dataset.csv"
    master.to_csv(out_path, index=False)
    print()
    print("Wrote master dataset to:", out_path)


if __name__ == "__main__":
    main()
