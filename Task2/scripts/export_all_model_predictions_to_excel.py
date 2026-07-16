"""
Generates Model_Predictions_AllFolds.xlsx: one sheet per model (Lag1,
RollingMean2, RollingMean3, CatBoost, CatBoost_Blend, LightGBM,
LightGBM_Blend, XGBoost, XGBoost_Blend), each containing per-row
predictions across all 3 walk-forward folds, plus a Summary sheet with
Business Accuracy per model.

Retrains CatBoost/LightGBM/XGBoost per fold using the Optuna-tuned
hyperparameters saved in best_hyperparams*.json (Phases 16, 21, 22), since
earlier phases printed aggregate metrics without persisting full per-row
predictions for every fold.
"""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "SalesTarget"
CAT_FEATURES = ["Item", "Segment"]
DROP_COLS = ["Date", "NetSalesTotal", "ReturnsTotal", TARGET]
FOLDS = [(list(range(1, 6)), 6), (list(range(1, 7)), 7), (list(range(1, 8)), 8)]
VOLUME_THRESHOLD = 50
COV_CUTOFF = 0.5


def business_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.round(np.asarray(y_pred, dtype=float))
    zero_mask = y_true == 0
    correct = np.empty(len(y_true), dtype=bool)
    correct[zero_mask] = pred[zero_mask] == 0
    nz = ~zero_mask
    correct[nz] = np.abs(pred[nz] - y_true[nz]) <= 0.2 * y_true[nz]
    return correct


def main():
    with open(f"{BASE}\\best_hyperparams.json") as f:
        cb_params = json.load(f)
    with open(f"{BASE}\\best_hyperparams_lgbm.json") as f:
        lgb_params = json.load(f)
    with open(f"{BASE}\\best_hyperparams_xgb.json") as f:
        xgb_params = json.load(f)

    df = pd.read_csv(f"{BASE}\\Item_Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    all_predictions = {name: [] for name in [
        "Lag1", "RollingMean2", "RollingMean3",
        "CatBoost", "CatBoost_Blend", "LightGBM", "LightGBM_Blend", "XGBoost", "XGBoost_Blend",
    ]}

    for fold_i, (train_periods, test_period) in enumerate(FOLDS, start=1):
        print(f"Fold {fold_i}: train TimeIndex {train_periods[0]}-{train_periods[-1]}, test {test_period}")
        train = df[df["TimeIndex"].isin(train_periods)]
        test = df[df["TimeIndex"] == test_period].copy()
        y_train_raw = train[TARGET].values
        y_test_raw = test[TARGET].values

        # ---- CatBoost (str categoricals via Pool) ----
        Xc_train, Xc_test = train[feature_cols].copy(), test[feature_cols].copy()
        for c in CAT_FEATURES:
            Xc_train[c] = Xc_train[c].astype(str)
            Xc_test[c] = Xc_test[c].astype(str)
        cb_p = dict(loss_function="MAE", random_seed=42, bootstrap_type="Bayesian",
                    early_stopping_rounds=50, verbose=False, **cb_params)
        train_pool = Pool(Xc_train, np.log1p(y_train_raw), cat_features=CAT_FEATURES)
        test_pool = Pool(Xc_test, np.log1p(y_test_raw), cat_features=CAT_FEATURES)
        cb_model = CatBoostRegressor(**cb_p)
        cb_model.fit(train_pool, eval_set=test_pool)
        cb_pred = np.clip(np.expm1(cb_model.predict(Xc_test)), 0, None)

        # ---- LightGBM / XGBoost (category dtype) ----
        Xg_train, Xg_test = train[feature_cols].copy(), test[feature_cols].copy()
        for c in CAT_FEATURES:
            Xg_train[c] = Xg_train[c].astype("category")
            Xg_test[c] = Xg_test[c].astype("category")

        lgb_p = dict(objective="regression_l1", random_state=42, verbosity=-1, **lgb_params)
        lgb_model = lgb.LGBMRegressor(**lgb_p)
        lgb_model.fit(Xg_train, np.log1p(y_train_raw),
                      eval_set=[(Xg_test, np.log1p(y_test_raw))],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        lgb_pred = np.clip(np.expm1(lgb_model.predict(Xg_test)), 0, None)

        xgb_p = dict(tree_method="hist", enable_categorical=True, objective="reg:absoluteerror",
                     random_state=42, early_stopping_rounds=50, **xgb_params)
        xgb_model = xgb.XGBRegressor(**xgb_p)
        xgb_model.fit(Xg_train, np.log1p(y_train_raw), eval_set=[(Xg_test, np.log1p(y_test_raw))], verbose=False)
        xgb_pred = np.clip(np.expm1(xgb_model.predict(Xg_test)), 0, None)

        lag1_pred = test["Lag1"].values
        rm2_pred = test["RollingMean2"].values
        rm3_pred = test["RollingMean3"].values

        cov = test["RollingStd3"] / test["RollingMean3"].replace(0, np.nan)
        stable_mask = (test["RollingMean3"].fillna(-1) > VOLUME_THRESHOLD) | (
            (test["RollingMean3"].fillna(-1) > 0) & (cov.fillna(np.inf) < COV_CUTOFF)
        )
        cb_blend = np.where(stable_mask, lag1_pred, cb_pred)
        lgb_blend = np.where(stable_mask, lag1_pred, lgb_pred)
        xgb_blend = np.where(stable_mask, lag1_pred, xgb_pred)

        base = test[["Item", "Date", "TimeIndex"]].copy()
        base["Fold"] = fold_i
        base["Actual"] = y_test_raw

        model_preds = {
            "Lag1": lag1_pred, "RollingMean2": rm2_pred, "RollingMean3": rm3_pred,
            "CatBoost": cb_pred, "CatBoost_Blend": cb_blend,
            "LightGBM": lgb_pred, "LightGBM_Blend": lgb_blend,
            "XGBoost": xgb_pred, "XGBoost_Blend": xgb_blend,
        }
        for name, pred in model_preds.items():
            d = base.copy()
            pred_arr = np.asarray(pred, dtype=float)
            d["Predicted"] = np.round(pred_arr)
            d["AbsError"] = np.abs(d["Actual"] - d["Predicted"])
            d["Correct"] = business_accuracy(d["Actual"].values, pred_arr)
            all_predictions[name].append(d)

    print()
    print("Writing Excel workbook...")
    summary_rows = []
    full_frames = {}
    for name, frames in all_predictions.items():
        full = pd.concat(frames, ignore_index=True)
        full_frames[name] = full
        ba_overall = full["Correct"].mean() * 100
        ba_by_fold = full.groupby("Fold")["Correct"].mean() * 100
        summary_rows.append({
            "Model": name,
            "BusinessAccuracy_Fold1": round(ba_by_fold.get(1, float("nan")), 2),
            "BusinessAccuracy_Fold2": round(ba_by_fold.get(2, float("nan")), 2),
            "BusinessAccuracy_Fold3": round(ba_by_fold.get(3, float("nan")), 2),
            "BusinessAccuracy_Mean": round(ba_by_fold.mean(), 2),
            "N_Predictions": len(full),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("BusinessAccuracy_Mean", ascending=False)

    out_path = f"{BASE}\\Model_Predictions_AllFolds.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for name, full in full_frames.items():
            full.to_excel(writer, sheet_name=name[:31], index=False)

    print("Wrote:", out_path)
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
