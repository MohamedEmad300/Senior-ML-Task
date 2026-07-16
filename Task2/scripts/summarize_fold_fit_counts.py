"""
Counts, per model and per walk-forward fold, how many item-months fit
within +/-20% of actual ("Correct" per the Business Accuracy rule in
forecast_utils.py) vs. how many don't.

Reads the per-row predictions already saved in Model_Predictions_AllFolds.xlsx
(one sheet per model, each with a boolean "Correct" column and a "Fold"
column) and tallies Fits_+-20pct / Incorrect / Total / Pct_Fit per
(Model, Fold). Writes the result both as a standalone CSV and as an
additional "FoldFitCounts" sheet appended into the same workbook.
"""
import pandas as pd

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast"
WORKBOOK = f"{BASE}\\Model_Predictions_AllFolds.xlsx"
MODELS = [
    "Lag1", "RollingMean2", "RollingMean3",
    "CatBoost", "CatBoost_Blend", "LightGBM", "LightGBM_Blend", "XGBoost", "XGBoost_Blend",
]


def main():
    print("Reading per-model prediction sheets...")
    rows = []
    for model in MODELS:
        sheet = pd.read_excel(WORKBOOK, sheet_name=model)
        g = sheet.groupby("Fold")["Correct"].agg(["sum", "count"])
        g["Incorrect"] = g["count"] - g["sum"]
        g = g.rename(columns={"sum": "Fits_+-20pct", "count": "Total"})
        g["Model"] = model
        g = g.reset_index()
        rows.append(g[["Model", "Fold", "Fits_+-20pct", "Incorrect", "Total"]])

    result = pd.concat(rows, ignore_index=True)
    result["Pct_Fit"] = (result["Fits_+-20pct"] / result["Total"] * 100).round(2)

    print(result.to_string(index=False))

    csv_path = f"{BASE}\\fold_fit_counts.csv"
    result.to_csv(csv_path, index=False)
    print()
    print("Saved:", csv_path)

    with pd.ExcelWriter(WORKBOOK, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        result.to_excel(writer, sheet_name="FoldFitCounts", index=False)
    print(f"Added/updated 'FoldFitCounts' sheet in {WORKBOOK}")


if __name__ == "__main__":
    main()
