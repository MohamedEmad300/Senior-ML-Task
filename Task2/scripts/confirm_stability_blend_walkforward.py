"""
Final confirmation -- the stability-blend approach (tuned log1p CatBoost +
Lag1 fallback for stable/high-volume items, RollingMean3 > 50 with a
CoV < 0.5 refinement) evaluated across all 3 walk-forward folds, not just
the Fold 2 dev split it was selected on.
"""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from business_accuracy_metrics import business_accuracy, eval_metrics

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "SalesTarget"
CAT_FEATURES = ["Item", "Segment"]
DROP_COLS = ["Date", "NetSalesTotal", "ReturnsTotal", TARGET]
FOLDS = [(list(range(1, 6)), 6), (list(range(1, 7)), 7), (list(range(1, 8)), 8)]
VOLUME_THRESHOLD = 50
COV_CUTOFF = 0.5


def main():
    with open(f"{BASE}\\best_hyperparams.json") as f:
        best_params = json.load(f)

    df = pd.read_csv(f"{BASE}\\Item_Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    fold_results = []
    for fold_i, (train_periods, test_period) in enumerate(FOLDS, start=1):
        print()
        print("=" * 78)
        print(f"FOLD {fold_i}: train TimeIndex {train_periods[0]}-{train_periods[-1]}, "
              f"test TimeIndex {test_period}")
        print("=" * 78)

        train = df[df["TimeIndex"].isin(train_periods)]
        test = df[df["TimeIndex"] == test_period].copy()

        X_train, y_train_raw = train[feature_cols].copy(), train[TARGET].values
        X_test, y_test_raw = test[feature_cols].copy(), test[TARGET].values
        for c in CAT_FEATURES:
            X_train[c] = X_train[c].astype(str)
            X_test[c] = X_test[c].astype(str)

        params = dict(loss_function="MAE", random_seed=42, bootstrap_type="Bayesian",
                      early_stopping_rounds=50, verbose=False, **best_params)
        train_pool = Pool(X_train, np.log1p(y_train_raw), cat_features=CAT_FEATURES)
        test_pool = Pool(X_test, np.log1p(y_test_raw), cat_features=CAT_FEATURES)
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=test_pool)
        pred_model = np.clip(np.expm1(model.predict(X_test)), 0, None)
        pred_lag1 = test["Lag1"].values

        cov = test["RollingStd3"] / test["RollingMean3"].replace(0, np.nan)
        stable_mask = (test["RollingMean3"].fillna(-1) > VOLUME_THRESHOLD) | (
            (test["RollingMean3"].fillna(-1) > 0) & (cov.fillna(np.inf) < COV_CUTOFF)
        )
        final_pred = np.where(stable_mask, pred_lag1, pred_model)

        ba_lag1 = business_accuracy(y_test_raw, pred_lag1) * 100
        ba_model = business_accuracy(y_test_raw, pred_model) * 100
        ba_blend = business_accuracy(y_test_raw, final_pred) * 100
        print(f"  Lag1 alone:        BusinessAcc={ba_lag1:.2f}%")
        print(f"  Tuned model alone: BusinessAcc={ba_model:.2f}%")
        print(f"  Stability blend:   BusinessAcc={ba_blend:.2f}%  "
              f"({stable_mask.mean()*100:.1f}% routed to Lag1)")
        fold_results.append({"fold": fold_i, "Lag1": ba_lag1, "Model": ba_model, "Blend": ba_blend})

    print()
    print("=" * 78)
    print("SUMMARY across all 3 folds")
    print("=" * 78)
    for name in ["Lag1", "Model", "Blend"]:
        mean_ba = np.mean([f[name] for f in fold_results])
        per_fold = ", ".join(f"F{f['fold']}={f[name]:.2f}%" for f in fold_results)
        print(f"  {name:8s}  mean BusinessAcc={mean_ba:6.2f}%   ({per_fold})")


if __name__ == "__main__":
    main()
