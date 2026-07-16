"""
Phase 8  -- Naive baselines (Lag1, RollingMean2, RollingMean3, Simple Exp Smoothing)
Phase 9  -- CatBoost Model #1, default hyperparameters, no tuning
Phase 10 -- Feature importance (built-in + SHAP)

Walk-forward validation over the 8 complete modeling months (202605 was
dropped -- see phase3_master_dataset.py -- it was a truncated extract that
made every model look broken). Expanding-window folds:
  Fold 1: train TimeIndex 1-5, test TimeIndex 6
  Fold 2: train TimeIndex 1-6, test TimeIndex 7
  Fold 3: train TimeIndex 1-7, test TimeIndex 8
Each fold retrains both the baselines' comparison and a fresh CatBoost
model, so no fold's test period ever leaks into an earlier fold's training
window. The Fold 3 model (most training data) is reused for Phase 10
feature importance / SHAP rather than training a redundant 4th model.

Target: PositiveSales.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "PositiveSales"
SES_ALPHA = 0.3

DROP_COLS = ["Date", "NetSales", "Returns", TARGET, "SES_Baseline"]
CAT_FEATURES = ["Item", "WH", "Segment"]
FOLDS = [(list(range(1, 6)), 6), (list(range(1, 7)), 7), (list(range(1, 8)), 8)]


def metrics(y_true, y_pred, name):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  {name:25s}  MAE={mae:8.3f}  RMSE={rmse:8.3f}  wMAPE={wmape:7.2f}%  R2={r2:7.3f}")
    return {"name": name, "MAE": mae, "RMSE": rmse, "wMAPE": wmape, "R2": r2}


def main():
    print("Loading Feature_Dataset.csv...")
    df = pd.read_csv(f"{BASE}\\Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "WH", "TimeIndex"]).reset_index(drop=True)

    shifted = df.groupby(["Item", "WH"], sort=False)["PositiveSales"].shift(1)
    df["_shifted_possales"] = shifted
    ses = df.groupby(["Item", "WH"], sort=False)["_shifted_possales"].ewm(
        alpha=SES_ALPHA, adjust=False
    ).mean()
    df["SES_Baseline"] = ses.reset_index(level=[0, 1], drop=True)
    df.drop(columns=["_shifted_possales"], inplace=True)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    all_fold_results = []
    last_model, last_feature_cols, last_X_test = None, None, None

    for fold_i, (train_periods, test_period) in enumerate(FOLDS, start=1):
        print()
        print("=" * 78)
        print(f"FOLD {fold_i}: train TimeIndex {train_periods[0]}-{train_periods[-1]}, "
              f"test TimeIndex {test_period}")
        print("=" * 78)

        train = df[df["TimeIndex"].isin(train_periods)]
        test = df[df["TimeIndex"] == test_period]
        print(f"Train rows: {len(train)}  Test rows: {len(test)}  "
              f"Test zero-share: {(test[TARGET]==0).mean():.1%}")

        print("Baselines:")
        fold_results = {"fold": fold_i}
        fold_results["Baseline1_Lag1"] = metrics(test[TARGET], test["Lag1"], "Baseline1_Lag1")
        fold_results["Baseline2_RollMean2"] = metrics(test[TARGET], test["RollingMean2"], "Baseline2_RollMean2")
        fold_results["Baseline3_RollMean3"] = metrics(test[TARGET], test["RollingMean3"], "Baseline3_RollMean3")
        fold_results["Baseline4_SES"] = metrics(test[TARGET], test["SES_Baseline"], f"Baseline4_SES(a={SES_ALPHA})")

        X_train, y_train = train[feature_cols].copy(), train[TARGET]
        X_test, y_test = test[feature_cols].copy(), test[TARGET]
        for c in CAT_FEATURES:
            X_train[c] = X_train[c].astype(str)
            X_test[c] = X_test[c].astype(str)

        train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
        test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)

        print("Training CatBoost (defaults)...")
        model = CatBoostRegressor(loss_function="RMSE", random_seed=42, verbose=False)
        model.fit(train_pool, eval_set=test_pool)

        preds = np.clip(model.predict(X_test), 0, None)
        print("CatBoost:")
        fold_results["CatBoost"] = metrics(y_test, preds, "CatBoost_Model1")

        all_fold_results.append(fold_results)
        last_model, last_feature_cols, last_X_test = model, feature_cols, X_test

    print()
    print("=" * 78)
    print("SUMMARY -- mean metrics across the 3 walk-forward folds")
    print("=" * 78)
    model_names = ["Baseline1_Lag1", "Baseline2_RollMean2", "Baseline3_RollMean3", "Baseline4_SES", "CatBoost"]
    summary = {}
    for name in model_names:
        maes = [f[name]["MAE"] for f in all_fold_results]
        rmses = [f[name]["RMSE"] for f in all_fold_results]
        wmapes = [f[name]["wMAPE"] for f in all_fold_results]
        summary[name] = {"MAE": np.mean(maes), "RMSE": np.mean(rmses), "wMAPE": np.mean(wmapes)}
        print(f"  {name:25s}  MAE={summary[name]['MAE']:8.3f}  RMSE={summary[name]['RMSE']:8.3f}  "
              f"wMAPE={summary[name]['wMAPE']:7.2f}%")

    print()
    print("Improvement of CatBoost over each baseline (mean MAE across folds):")
    cb_mae = summary["CatBoost"]["MAE"]
    for name in model_names[:-1]:
        improvement = (summary[name]["MAE"] - cb_mae) / summary[name]["MAE"] * 100
        print(f"  vs {name:25s}: {improvement:6.2f}% MAE reduction")

    # ---- Phase 10: feature importance from the Fold 3 model (most training data) ----
    print()
    print("=" * 78)
    print("PHASE 10 -- Feature Importance (Fold 3 model: train 1-7, test 8)")
    print("=" * 78)

    fi = last_model.get_feature_importance(type="FeatureImportance")
    fi_df = pd.DataFrame({"feature": last_feature_cols, "importance": fi}).sort_values(
        "importance", ascending=False
    )
    print()
    print("Built-in feature importance (PredictionValuesChange), top 20:")
    print(fi_df.head(20).to_string(index=False))

    print()
    print("Computing SHAP values on a 20,000-row sample of the Fold 3 test set...")
    shap_sample = last_X_test.sample(n=min(20000, len(last_X_test)), random_state=42)
    shap_pool = Pool(shap_sample, cat_features=CAT_FEATURES)
    shap_values = last_model.get_feature_importance(shap_pool, type="ShapValues")
    mean_abs_shap = np.abs(shap_values[:, :-1]).mean(axis=0)
    shap_df = pd.DataFrame({"feature": last_feature_cols, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    )
    print()
    print("SHAP mean |value|, top 20:")
    print(shap_df.head(20).to_string(index=False))

    fi_df.to_csv(f"{BASE}\\feature_importance_builtin.csv", index=False)
    shap_df.to_csv(f"{BASE}\\feature_importance_shap.csv", index=False)
    last_model.save_model(f"{BASE}\\catboost_model1.cbm")
    print()
    print("Saved: feature_importance_builtin.csv, feature_importance_shap.csv, catboost_model1.cbm")


if __name__ == "__main__":
    main()
