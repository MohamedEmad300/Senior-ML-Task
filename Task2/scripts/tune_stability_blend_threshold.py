"""
Phase 9 (refined) -- Blend Lag1 and the tuned model based on a stability
proxy, not a raw volume cutoff.

Framing (per business direction): the error analysis only proved that
high-volume items currently do better under Lag1 -- it didn't prove volume
is the underlying cause. Volume is used here only because it's a cheap,
available proxy for demand stability (RollingMean3, already computed).
Report language should say "persistent/stable medicines" not "high-volume
medicines were treated specially."

Rule: for each candidate threshold T, if RollingMean3 > T, use Lag1
(the item is assumed high-volume/stable enough that persistence is safer
than the model); otherwise use the tuned log1p CatBoost. Rows with no
RollingMean3 yet (insufficient history) fall back to the model, since
stability can't be confirmed.

Thresholds swept: 50, 100, 200, 500 (as specified), picked by Business
Accuracy on the Fold 2 dev split (train TimeIndex 1-6, test TimeIndex 7).
A bonus low-volatility condition (coefficient of variation) is also
checked on top of the winning threshold.
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
TRAIN_PERIODS = list(range(1, 7))
TEST_PERIOD = 7
THRESHOLDS = [50, 100, 200, 500]


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
    test["Pred_Model"] = np.clip(np.expm1(model.predict(X_test)), 0, None)
    test["Pred_Lag1"] = test["Lag1"]

    print("Threshold sweep (stability proxy = RollingMean3, avg of last 3 months):")
    print(f"{'Threshold':>10s} {'%StableRows':>12s} {'BusinessAcc':>12s}")
    results = []
    for t in THRESHOLDS:
        stable_mask = test["RollingMean3"].fillna(-1) > t
        final_pred = np.where(stable_mask, test["Pred_Lag1"], test["Pred_Model"])
        ba = business_accuracy(test[TARGET].values, final_pred) * 100
        pct_stable = stable_mask.mean() * 100
        print(f"{t:>10d} {pct_stable:>11.1f}% {ba:>11.2f}%")
        results.append((t, ba, pct_stable))

    best_t, best_ba, best_pct = max(results, key=lambda r: r[1])
    print()
    print(f"Best threshold: RollingMean3 > {best_t}  (BusinessAcc={best_ba:.2f}%, "
          f"{best_pct:.1f}% of rows routed to Lag1)")

    stable_mask = test["RollingMean3"].fillna(-1) > best_t
    final_pred_best = np.where(stable_mask, test["Pred_Lag1"], test["Pred_Model"])
    test["Final_pred_blend"] = final_pred_best

    print()
    print("Reference:")
    print(f"  Lag1 alone:            BusinessAcc=65.10%")
    print(f"  Tuned log1p model alone: BusinessAcc=62.99%")
    print(f"  Blend (threshold={best_t}):  BusinessAcc={best_ba:.2f}%")

    print()
    print("Bonus: add low-volatility condition (coefficient of variation) on top of best threshold...")
    cov = test["RollingStd3"] / test["RollingMean3"].replace(0, np.nan)
    for cov_cutoff in [0.3, 0.5, 0.75, 1.0]:
        stable_mask2 = (test["RollingMean3"].fillna(-1) > best_t) | (
            (test["RollingMean3"].fillna(-1) > 0) & (cov.fillna(np.inf) < cov_cutoff)
        )
        final_pred2 = np.where(stable_mask2, test["Pred_Lag1"], test["Pred_Model"])
        ba2 = business_accuracy(test[TARGET].values, final_pred2) * 100
        print(f"  + CoV < {cov_cutoff}: BusinessAcc={ba2:.2f}%  ({stable_mask2.mean()*100:.1f}% routed to Lag1)")

    test.to_csv(f"{BASE}\\stability_blend_predictions_fold2.csv", index=False)
    print()
    print("Saved: stability_blend_predictions_fold2.csv")


if __name__ == "__main__":
    main()
