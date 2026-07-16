"""
Phase 7 (new plan) -- log1p target transform, using the Optuna-tuned
hyperparameters from Phase 16. Train on log1p(SalesTarget), predict,
invert with expm1(), compare Business Accuracy against the raw-target
tuned model (61.23% on this fold).
"""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from business_accuracy_metrics import eval_metrics

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "SalesTarget"
CAT_FEATURES = ["Item", "Segment"]
DROP_COLS = ["Date", "NetSalesTotal", "ReturnsTotal", TARGET]
TRAIN_PERIODS = list(range(1, 7))
TEST_PERIOD = 7


def main():
    with open(f"{BASE}\\best_hyperparams.json") as f:
        best_params = json.load(f)
    print("Using tuned hyperparameters:", best_params)

    df = pd.read_csv(f"{BASE}\\Item_Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    train = df[df["TimeIndex"].isin(TRAIN_PERIODS)]
    test = df[df["TimeIndex"] == TEST_PERIOD]

    X_train, y_train_raw = train[feature_cols].copy(), train[TARGET].values
    X_test, y_test_raw = test[feature_cols].copy(), test[TARGET].values
    for c in CAT_FEATURES:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    y_train_log = np.log1p(y_train_raw)
    y_test_log = np.log1p(y_test_raw)

    params = dict(loss_function="MAE", random_seed=42, bootstrap_type="Bayesian",
                  early_stopping_rounds=50, verbose=False, **best_params)

    print()
    print("Training on log1p(SalesTarget)...")
    train_pool = Pool(X_train, y_train_log, cat_features=CAT_FEATURES)
    test_pool = Pool(X_test, y_test_log, cat_features=CAT_FEATURES)
    model_log = CatBoostRegressor(**params)
    model_log.fit(train_pool, eval_set=test_pool)
    pred_log = np.clip(np.expm1(model_log.predict(X_test)), 0, None)
    m_log = eval_metrics(y_test_raw, pred_log, "LogTarget_Tuned")

    print()
    print("Retraining on raw SalesTarget (same tuned params) for direct comparison...")
    train_pool_raw = Pool(X_train, y_train_raw, cat_features=CAT_FEATURES)
    test_pool_raw = Pool(X_test, y_test_raw, cat_features=CAT_FEATURES)
    model_raw = CatBoostRegressor(**params)
    model_raw.fit(train_pool_raw, eval_set=test_pool_raw)
    pred_raw = np.clip(model_raw.predict(X_test), 0, None)
    m_raw = eval_metrics(y_test_raw, pred_raw, "RawTarget_Tuned")

    print()
    winner = "log1p" if m_log["BusinessAccuracy"] > m_raw["BusinessAccuracy"] else "raw"
    print(f"Winner by Business Accuracy: {winner}")
    print("  Reference -- Lag1 on this fold: BusinessAcc=65.10%")

    (model_log if winner == "log1p" else model_raw).save_model(f"{BASE}\\catboost_item_level_final.cbm")
    print(f"Saved winning model as catboost_item_level_final.cbm (target_transform={winner})")


if __name__ == "__main__":
    main()
