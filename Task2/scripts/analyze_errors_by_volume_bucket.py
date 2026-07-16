"""
Phase 8 (new plan) -- Error analysis by actual-value bucket.

Compares the tuned log1p CatBoost against Lag1 within each bucket of
actual SalesTarget, to see exactly where the model wins or loses --
rather than jumping to another architecture change on an aggregate number.
"""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from business_accuracy_metrics import business_accuracy

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "SalesTarget"
CAT_FEATURES = ["Item", "Segment"]
DROP_COLS = ["Date", "NetSalesTotal", "ReturnsTotal", TARGET]
TRAIN_PERIODS = list(range(1, 7))
TEST_PERIOD = 7

BUCKETS = [(-0.5, 0.5, "0"), (0.5, 5.5, "1-5"), (5.5, 20.5, "6-20"),
           (20.5, 100.5, "21-100"), (100.5, np.inf, "100+")]


def bucketize(values):
    labels = np.full(len(values), "", dtype=object)
    for lo, hi, name in BUCKETS:
        mask = (values > lo) & (values <= hi)
        labels[mask] = name
    return labels


def main():
    with open(f"{BASE}\\best_hyperparams.json") as f:
        best_params = json.load(f)

    df = pd.read_csv(f"{BASE}\\Item_Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    train = df[df["TimeIndex"].isin(TRAIN_PERIODS)]
    test = df[df["TimeIndex"] == TEST_PERIOD].copy()

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

    test["Bucket"] = bucketize(y_test_raw)
    test["Pred_Model"] = pred_model
    test["Pred_Lag1"] = test["Lag1"]

    print()
    print(f"{'Bucket':10s} {'N':>8s} {'%ofRows':>8s} {'BA_Lag1':>10s} {'BA_Model':>10s} {'Delta':>8s}")
    order = [b[2] for b in BUCKETS]
    for bucket in order:
        sub = test[test["Bucket"] == bucket]
        if len(sub) == 0:
            continue
        ba_lag1 = business_accuracy(sub[TARGET].values, sub["Pred_Lag1"].values) * 100
        ba_model = business_accuracy(sub[TARGET].values, sub["Pred_Model"].values) * 100
        pct = len(sub) / len(test) * 100
        print(f"{bucket:10s} {len(sub):8d} {pct:7.1f}% {ba_lag1:9.2f}% {ba_model:9.2f}% "
              f"{ba_model-ba_lag1:+7.2f}pp")

    print()
    overall_lag1 = business_accuracy(test[TARGET].values, test["Pred_Lag1"].values) * 100
    overall_model = business_accuracy(test[TARGET].values, test["Pred_Model"].values) * 100
    print(f"{'OVERALL':10s} {len(test):8d} {'100.0%':>8s} {overall_lag1:9.2f}% {overall_model:9.2f}% "
          f"{overall_model-overall_lag1:+7.2f}pp")

    test.to_csv(f"{BASE}\\error_analysis_by_bucket.csv", index=False)
    print()
    print("Saved: error_analysis_by_bucket.csv")


if __name__ == "__main__":
    main()
