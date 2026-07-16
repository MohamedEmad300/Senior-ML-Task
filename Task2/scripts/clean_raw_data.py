"""
Phase 2 -- Cleaning

Produces cleaned versions of the five raw files:
  Sales2025_clean.csv, Sales2026_clean.csv
  Outage2025_clean.csv, Outage2026_clean.csv
  AvailabilityHistory_clean.csv

Rules applied (per Phase 2 decisions):

AvailabilityHistory
  - drop exact duplicate rows
  - strip whitespace from Segment (ROFF is KEPT -- it is informative, likely
    Removed/Retired/Off-market, and is a real demand-relevant state)

Outage
  - Item cast to string (not int) since it is a code, not a quantity
  - Whitespace stripped from Item
  - "EXPxxxxxx" codes: prefix stripped to test whether the underlying numeric
    code is a real, sold item (appears in Sales2025/2026). If yes, it is
    treated as the same item with an outage caused by an expiry flag
    (IsExpFlagged=1) and re-keyed to the numeric Item_Code so it joins to
    Sales/Availability. If the numeric code never appears in Sales, the row
    is dropped from the modeling set (kept in a *_excluded.csv for audit).
  - "Y90006" and similar non-numeric, non-EXP codes: never appear in Sales
    under any interpretation -> excluded from modeling (audited separately).

Sales
  - Item_Code cast to string to match Outage/Availability join keys
  - Negative NET_QTY rows are NOT removed or clipped in place. Instead:
      NetSales      = NET_QTY (unchanged, can be negative)
      Returns       = RETURNS_QTY (unchanged, <= 0)
      PositiveSales = max(NET_QTY, 0)   -- demand-only signal for models
        that require non-negative targets; NetSales retains the sign so
        return-heavy months (potential precursor to zero future demand)
        are not thrown away.
"""
import re
import pandas as pd

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"


def load_sales(fname):
    df = pd.read_csv(f"{BASE}\\{fname}", dtype={"Item_Code": str})
    df["Item_Code"] = df["Item_Code"].str.strip()
    df["NetSales"] = df["NET_QTY"]
    df["Returns"] = df["RETURNS_QTY"]
    df["PositiveSales"] = df["NET_QTY"].clip(lower=0)
    return df


def load_outage(fname, sales_items):
    df = pd.read_csv(f"{BASE}\\{fname}", dtype={"Item": str})
    df["Item"] = df["Item"].str.strip()

    is_numeric = df["Item"].str.match(r"^\d+$")
    is_exp = df["Item"].str.match(r"^EXP\d+$")

    df["IsExpFlagged"] = False
    resolved_item = df["Item"].copy()

    exp_mask = is_exp
    exp_suffix = df.loc[exp_mask, "Item"].str.replace("^EXP", "", regex=True)
    exp_matches_sales = exp_suffix.isin(sales_items)

    # Re-key EXP rows whose numeric suffix is a real sold item
    matched_idx = exp_suffix.index[exp_matches_sales]
    resolved_item.loc[matched_idx] = exp_suffix.loc[matched_idx]
    df.loc[matched_idx, "IsExpFlagged"] = True

    keep_mask = is_numeric | (exp_mask & exp_matches_sales.reindex(df.index, fill_value=False))

    df["Item_Code"] = resolved_item
    clean = df.loc[keep_mask].copy()
    excluded = df.loc[~keep_mask].copy()
    return clean, excluded


def load_availability(fname):
    df = pd.read_csv(f"{BASE}\\{fname}")
    df["Segment"] = df["Segment"].str.strip()
    before = len(df)
    df = df.drop_duplicates()
    print(f"  AvailabilityHistory: dropped {before - len(df)} exact duplicate rows")
    df["ITEM_CODE"] = df["ITEM_CODE"].astype(str)
    return df


def main():
    print("Loading Sales files...")
    s25 = load_sales("Sales2025.csv")
    s26 = load_sales("Sales2026.csv")
    sales_items = set(s25["Item_Code"]) | set(s26["Item_Code"])

    print("Loading Outage files...")
    o25, o25_excl = load_outage("Outage2025.csv", sales_items)
    o26, o26_excl = load_outage("Outage2026.csv", sales_items)

    print("Loading AvailabilityHistory...")
    av = load_availability("AvailabilityHistory.csv")

    print()
    print("Outage2025: kept", len(o25), "/ excluded", len(o25_excl))
    print("Outage2026: kept", len(o26), "/ excluded", len(o26_excl))
    print("EXP-flagged rows retained (re-keyed to numeric item):",
          int(o25["IsExpFlagged"].sum()), "(2025) +", int(o26["IsExpFlagged"].sum()), "(2026)")
    print()
    print("Segment distribution after cleaning:")
    print(av["Segment"].value_counts())
    print()
    print("Sales2025 sample of derived columns:")
    print(s25[s25["NetSales"] < 0][["DateMonth", "Item_Code", "WH", "NetSales", "Returns", "PositiveSales"]].head())

    # Write cleaned outputs
    s25.to_csv(f"{BASE}\\Sales2025_clean.csv", index=False)
    s26.to_csv(f"{BASE}\\Sales2026_clean.csv", index=False)
    o25.drop(columns=["Item"]).to_csv(f"{BASE}\\Outage2025_clean.csv", index=False)
    o26.drop(columns=["Item"]).to_csv(f"{BASE}\\Outage2026_clean.csv", index=False)
    av.to_csv(f"{BASE}\\AvailabilityHistory_clean.csv", index=False)

    # Audit trail of excluded outage rows (never appear in Sales under any code interpretation)
    if len(o25_excl):
        o25_excl.to_csv(f"{BASE}\\Outage2025_excluded.csv", index=False)
    if len(o26_excl):
        o26_excl.to_csv(f"{BASE}\\Outage2026_excluded.csv", index=False)

    print()
    print("Wrote cleaned files to:", BASE)


if __name__ == "__main__":
    main()
