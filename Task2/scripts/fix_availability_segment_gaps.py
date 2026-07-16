"""
Investigate the 761,051 UNKNOWN Segment rows in Master_Dataset.csv.

For every UNKNOWN (Item, Date) cell, classify it using the FULL
AvailabilityHistory (202307-202606, wider than the 9-month modeling
window) rather than just what's inside the panel:

  Case 1 - Not yet launched: no Availability record on/before this Date,
           but a record exists AFTER it. UNKNOWN is correct, no fix.
  Case 2 - Disappeared/gap: a record exists on/before this Date (and
           possibly after). UNKNOWN is very likely wrong -> carry the
           last known Segment forward (LOCF).
  Case 3 - Never tracked: item has no Availability record at all, ever.
           UNKNOWN is the only honest label, no fix.

This patches Master_Dataset.csv in place (Segment column) and adds one
new column, HasAvailabilityRecord (1 = this exact Item/Date had a real
Availability row before any imputation, 0 = imputed or genuinely
missing) so the model can use the missingness itself as a signal.
"""
import pandas as pd

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"


def main():
    print("Loading Master_Dataset.csv and full AvailabilityHistory_clean.csv...")
    master = pd.read_csv(f"{BASE}\\Master_Dataset.csv", dtype={"Item": str})
    avail = pd.read_csv(f"{BASE}\\AvailabilityHistory_clean.csv", dtype={"ITEM_CODE": str})
    avail = avail.rename(columns={"ITEM_CODE": "Item", "Date": "AvailDate"})
    avail = avail[["Item", "AvailDate", "Segment"]].drop_duplicates()

    # HasAvailabilityRecord: real record existed for this exact (Item, Date)
    # BEFORE any imputation -- must be computed before Segment is touched.
    master["HasAvailabilityRecord"] = (master["Segment"] != "UNKNOWN").astype(int)

    n_unknown_before = (master["Segment"] == "UNKNOWN").sum()
    print(f"UNKNOWN rows before investigation: {n_unknown_before} "
          f"({n_unknown_before / len(master):.1%} of panel)")

    # merge_asof requires the "on" column globally monotonic (not grouped by
    # "by" first) -- sort by Date only, keep original row position to map back.
    master_sorted = master.reset_index(drop=True)
    left = master_sorted[["Item", "Date"]].reset_index().rename(columns={"index": "orig_idx"})
    left = left.sort_values("Date").reset_index(drop=True)
    avail_sorted = avail.rename(columns={"AvailDate": "Date"}).sort_values("Date").reset_index(drop=True)

    backward_sorted = pd.merge_asof(
        left, avail_sorted, on="Date", by="Item", direction="backward", allow_exact_matches=True,
    )
    forward_sorted = pd.merge_asof(
        left, avail_sorted, on="Date", by="Item", direction="forward", allow_exact_matches=True,
    )

    backward = backward_sorted.set_index("orig_idx").sort_index()["Segment"]
    forward = forward_sorted.set_index("orig_idx").sort_index()["Segment"]

    is_unknown = master_sorted["Segment"] == "UNKNOWN"
    case2_mask = is_unknown & backward.notna()
    case1_mask = is_unknown & backward.isna() & forward.notna()
    case3_mask = is_unknown & backward.isna() & forward.isna()

    print()
    print("Breakdown of UNKNOWN rows:")
    print(f"  Case 1 (not yet launched, no fix needed):  {case1_mask.sum():>9,}")
    print(f"  Case 2 (disappeared, carry-forward fixed): {case2_mask.sum():>9,}")
    print(f"  Case 3 (never tracked, no fix possible):   {case3_mask.sum():>9,}")
    print(f"  Total UNKNOWN accounted for:               {case1_mask.sum() + case2_mask.sum() + case3_mask.sum():>9,}")

    master_sorted.loc[case2_mask, "Segment"] = backward[case2_mask].values

    n_unknown_after = (master_sorted["Segment"] == "UNKNOWN").sum()
    print()
    print(f"UNKNOWN rows after LOCF fix: {n_unknown_after} "
          f"({n_unknown_after / len(master_sorted):.1%} of panel)")
    print()
    print("Segment distribution after fix:")
    print(master_sorted["Segment"].value_counts())
    print()
    print("HasAvailabilityRecord distribution:")
    print(master_sorted["HasAvailabilityRecord"].value_counts())

    # restore original column order + new column at the end
    col_order = [
        "Date", "Year", "Month", "Quarter", "MonthSin", "MonthCos",
        "Item", "WH", "NetSales", "PositiveSales", "Returns",
        "Outage", "IsExpFlagged", "Segment", "HasAvailabilityRecord",
    ]
    master_sorted = master_sorted[col_order]

    master_sorted.to_csv(f"{BASE}\\Master_Dataset.csv", index=False)
    print()
    print("Master_Dataset.csv patched in place.")


if __name__ == "__main__":
    main()
