"""
Phase 3-5 (new plan) -- Lag1 baseline, default CatBoost, at (Item, Month)
grain. Every experiment is judged by Business Accuracy, not MAE.

Walk-forward folds, same scheme as the warehouse-level experiments:
  Fold 1: train TimeIndex 1-5, test TimeIndex 6
  Fold 2: train TimeIndex 1-6, test TimeIndex 7
  Fold 3: train TimeIndex 1-7, test TimeIndex 8

Current-period Outage/Segment/Availability aggregates are kept as features
(exogenous supply-side facts, same treatment as the row-level pipeline).
Current-period NetSalesTotal/ReturnsTotal are dropped -- same-period sales
byproducts, contemporaneous with the target, so only their Lag1 versions
are used.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from business_accuracy_metrics import eval_metrics

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "SalesTarget"
CAT_FEATURES = ["Item", "Segment"]
DROP_COLS = ["Date", "NetSalesTotal", "ReturnsTotal", TARGET]
FOLDS = [(list(range(1, 6)), 6), (list(range(1, 7)), 7), (list(range(1, 8)), 8)]


def main():
    df = pd.read_csv(f"{BASE}\\Item_Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    fold_summaries = []
    for fold_i, (train_periods, test_period) in enumerate(FOLDS, start=1):
        print()
        print("=" * 78)
        print(f"FOLD {fold_i}: train TimeIndex {train_periods[0]}-{train_periods[-1]}, "
              f"test TimeIndex {test_period}")
        print("=" * 78)

        train = df[df["TimeIndex"].isin(train_periods)]
        test = df[df["TimeIndex"] == test_period]
        print(f"Train item-months: {len(train)}  Test item-months: {len(test)}  "
              f"Test zero-share: {(test[TARGET]==0).mean():.1%}")

        fold_result = {"fold": fold_i}
        fold_result["Lag1"] = eval_metrics(test[TARGET], test["Lag1"], "Baseline_Lag1")
        fold_result["RollMean2"] = eval_metrics(test[TARGET], test["RollingMean2"], "Baseline_RollMean2")
        fold_result["RollMean3"] = eval_metrics(test[TARGET], test["RollingMean3"], "Baseline_RollMean3")

        X_train, y_train = train[feature_cols].copy(), train[TARGET]
        X_test, y_test = test[feature_cols].copy(), test[TARGET]
        for c in CAT_FEATURES:
            X_train[c] = X_train[c].astype(str)
            X_test[c] = X_test[c].astype(str)

        train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
        test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)
        model = CatBoostRegressor(loss_function="MAE", random_seed=42, iterations=3000,
                                   early_stopping_rounds=50, verbose=False)
        model.fit(train_pool, eval_set=test_pool)
        preds = np.clip(model.predict(X_test), 0, None)
        fold_result["CatBoost"] = eval_metrics(y_test, preds, "CatBoost_ItemLevel")
        fold_summaries.append(fold_result)

    print()
    print("=" * 78)
    print("SUMMARY -- mean Business Accuracy across 3 folds")
    print("=" * 78)
    for name in ["Lag1", "RollMean2", "RollMean3", "CatBoost"]:
        mean_ba = np.mean([f[name]["BusinessAccuracy"] for f in fold_summaries])
        mean_mae = np.mean([f[name]["MAE"] for f in fold_summaries])
        print(f"  {name:15s}  BusinessAcc={mean_ba:6.2f}%  MAE={mean_mae:8.3f}")


if __name__ == "__main__":
    main()
