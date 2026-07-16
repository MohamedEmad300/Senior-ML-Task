"""
Phase 11a -- Log-target transform + loss function comparison.

Dev split for this comparison: Fold 2 (train TimeIndex 1-6, test TimeIndex 7)
from the walk-forward scheme used in Phase 8-10. Chosen (not Fold 3) so a
truly held-out fold (3, test=TimeIndex 8) remains available later for
confirming the final tuned config hasn't overfit to this comparison.

Model selection criterion per business direction: Business Accuracy
(share of forecasts within +/-20% of actual, with actual=0 handled per the
documented assumption in forecast_utils.py) -- not MAE/RMSE.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from business_accuracy_metrics import eval_metrics

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "PositiveSales"
CAT_FEATURES = ["Item", "WH", "Segment"]
DROP_COLS = ["Date", "NetSales", "Returns", TARGET]

TRAIN_PERIODS = list(range(1, 7))
TEST_PERIOD = 7


def main():
    df = pd.read_csv(f"{BASE}\\Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "WH", "TimeIndex"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    train = df[df["TimeIndex"].isin(TRAIN_PERIODS)]
    test = df[df["TimeIndex"] == TEST_PERIOD]
    print(f"Dev fold: train TimeIndex {TRAIN_PERIODS[0]}-{TRAIN_PERIODS[-1]} "
          f"({len(train)} rows), test TimeIndex {TEST_PERIOD} ({len(test)} rows)")

    X_train, y_train_raw = train[feature_cols].copy(), train[TARGET].values
    X_test, y_test_raw = test[feature_cols].copy(), test[TARGET].values
    for c in CAT_FEATURES:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    print()
    print("Baselines (for reference, incl. Business Accuracy):")
    eval_metrics(y_test_raw, test["Lag1"], "Baseline1_Lag1")
    eval_metrics(y_test_raw, test["RollingMean2"], "Baseline2_RollMean2")
    eval_metrics(y_test_raw, test["RollingMean3"], "Baseline3_RollMean3")

    y_train_log = np.log1p(y_train_raw)
    y_test_log = np.log1p(y_test_raw)

    loss_functions = ["RMSE", "MAE", "Huber:delta=1.0"]
    results = {}

    print()
    print("=" * 70)
    print("Log-target CatBoost, comparing loss functions (defaults otherwise)")
    print("=" * 70)
    for loss in loss_functions:
        train_pool = Pool(X_train, y_train_log, cat_features=CAT_FEATURES)
        test_pool = Pool(X_test, y_test_log, cat_features=CAT_FEATURES)
        model = CatBoostRegressor(
            loss_function=loss, random_seed=42, iterations=3000,
            early_stopping_rounds=50, verbose=False,
        )
        model.fit(train_pool, eval_set=test_pool)
        pred_log = model.predict(X_test)
        pred = np.clip(np.expm1(pred_log), 0, None)
        m = eval_metrics(y_test_raw, pred, f"LogTarget_{loss}")
        m["best_iteration"] = model.get_best_iteration()
        results[loss] = m

    print()
    best_loss = max(results, key=lambda k: results[k]["BusinessAccuracy"])
    print(f"Best loss function by Business Accuracy: {best_loss}")
    for loss, m in results.items():
        print(f"  {loss:20s} BusinessAcc={m['BusinessAccuracy']:.2f}%  "
              f"MAE={m['MAE']:.3f}  best_iter={m['best_iteration']}")


if __name__ == "__main__":
    main()
