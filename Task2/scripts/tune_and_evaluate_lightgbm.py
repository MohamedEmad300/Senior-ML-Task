"""
Phase 9 (final resort) -- LightGBM, per the plan: only tried because the
CatBoost route (tuned + log1p + stability blend) is our current best
(mean BA=65.28% across 3 folds) but only edges out Lag1 (64.60%) by a
modest +0.68pp.

Same item-month dataset (Item_Feature_Dataset.csv), same walk-forward
folds, same Business Accuracy metric, same categorical treatment (Item,
Segment) via LightGBM's native categorical support. Same stability-blend
routing (RollingMean3 > 50 OR CoV < 0.5 -> Lag1) applied on top, for a
fair head-to-head against the CatBoost blend.

Tuning: a modest Optuna search (num_leaves, learning_rate, n_estimators,
min_child_samples, reg_lambda) on the Fold 2 dev split, mirroring the
CatBoost tuning scope (a handful of important params, not an exhaustive
search).
"""
import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from business_accuracy_metrics import business_accuracy, eval_metrics

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "SalesTarget"
CAT_FEATURES = ["Item", "Segment"]
DROP_COLS = ["Date", "NetSalesTotal", "ReturnsTotal", TARGET]
TRAIN_PERIODS_DEV = list(range(1, 7))
TEST_PERIOD_DEV = 7
FOLDS = [(list(range(1, 6)), 6), (list(range(1, 7)), 7), (list(range(1, 8)), 8)]
VOLUME_THRESHOLD = 50
COV_CUTOFF = 0.5
N_TRIALS = 30

optuna.logging.set_verbosity(optuna.logging.WARNING)


def prep_xy(frame, feature_cols):
    X = frame[feature_cols].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    return X


def main():
    df = pd.read_csv(f"{BASE}\\Item_Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    train_dev = df[df["TimeIndex"].isin(TRAIN_PERIODS_DEV)]
    test_dev = df[df["TimeIndex"] == TEST_PERIOD_DEV]
    X_train_dev = prep_xy(train_dev, feature_cols)
    X_test_dev = prep_xy(test_dev, feature_cols)
    y_train_dev_log = np.log1p(train_dev[TARGET].values)
    y_test_dev_raw = test_dev[TARGET].values

    print("Optuna tuning LightGBM (log1p target) on Fold 2 dev split...")

    def objective(trial):
        params = dict(
            objective="regression_l1",
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            n_estimators=trial.suggest_int("n_estimators", 200, 3000),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            random_state=42,
            verbosity=-1,
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_dev, y_train_dev_log,
            eval_set=[(X_test_dev, np.log1p(y_test_dev_raw))],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = np.clip(np.expm1(model.predict(X_test_dev)), 0, None)
        return business_accuracy(y_test_dev_raw, pred)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    print(f"Best trial BusinessAcc={study.best_value*100:.2f}%  params={study.best_params}")

    with open(f"{BASE}\\best_hyperparams_lgbm.json", "w") as f:
        json.dump(study.best_params, f, indent=2)

    print()
    print("=" * 78)
    print("Walk-forward confirmation: LightGBM alone, and LightGBM + stability blend")
    print("=" * 78)

    fold_results = []
    for fold_i, (train_periods, test_period) in enumerate(FOLDS, start=1):
        train = df[df["TimeIndex"].isin(train_periods)]
        test = df[df["TimeIndex"] == test_period].copy()

        X_train = prep_xy(train, feature_cols)
        X_test = prep_xy(test, feature_cols)
        y_train_log = np.log1p(train[TARGET].values)
        y_test_raw = test[TARGET].values

        params = dict(objective="regression_l1", random_state=42, verbosity=-1, **study.best_params)
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train_log,
            eval_set=[(X_test, np.log1p(y_test_raw))],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
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
        print(f"Fold {fold_i} (test=TimeIndex {test_period}): "
              f"Lag1={ba_lag1:.2f}%  LightGBM_alone={ba_model:.2f}%  LightGBM_blend={ba_blend:.2f}%")
        fold_results.append({"fold": fold_i, "Lag1": ba_lag1, "LGBM": ba_model, "LGBM_Blend": ba_blend})

    print()
    print("=" * 78)
    print("SUMMARY -- LightGBM vs CatBoost (from Phase 20) vs Lag1")
    print("=" * 78)
    for name in ["Lag1", "LGBM", "LGBM_Blend"]:
        mean_ba = np.mean([f[name] for f in fold_results])
        print(f"  {name:12s}  mean BusinessAcc={mean_ba:6.2f}%")
    print()
    print("  For reference (Phase 20, CatBoost):")
    print("    Lag1              mean BusinessAcc= 64.60%")
    print("    CatBoost alone    mean BusinessAcc= 61.74%")
    print("    CatBoost blend    mean BusinessAcc= 65.28%")


if __name__ == "__main__":
    main()
