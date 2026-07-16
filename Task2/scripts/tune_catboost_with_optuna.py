"""
Phase 6 (new plan) -- Optuna hyperparameter tuning for the item-level model.

Tuned: depth, learning_rate, iterations (ceiling, with early stopping),
l2_leaf_reg, random_strength, bagging_temperature (requires
bootstrap_type="Bayesian" to take effect).

Objective: maximize Business Accuracy directly (Optuna is black-box, so
there's no need for a differentiable proxy -- BA is computed post-hoc after
each trial's fit, same as any other metric).

Tuning dev split: Fold 2 (train TimeIndex 1-6, test TimeIndex 7), matching
every prior comparison in this project. Fold 1 and Fold 3 stay unseen by
the search itself; the best params get a walk-forward confirmation run
across all 3 folds afterward.
"""
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor, Pool
from business_accuracy_metrics import eval_metrics, business_accuracy

BASE = r"D:\Ibn SIna\tasks\Task 2- Sales Forecast\Task2_Deliverable\data"
TARGET = "SalesTarget"
CAT_FEATURES = ["Item", "Segment"]
DROP_COLS = ["Date", "NetSalesTotal", "ReturnsTotal", TARGET]
TRAIN_PERIODS = list(range(1, 7))
TEST_PERIOD = 7
N_TRIALS = 40

optuna.logging.set_verbosity(optuna.logging.WARNING)


def main():
    df = pd.read_csv(f"{BASE}\\Item_Feature_Dataset.csv", dtype={"Item": str})
    df = df.sort_values(["Item", "TimeIndex"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]

    train = df[df["TimeIndex"].isin(TRAIN_PERIODS)]
    test = df[df["TimeIndex"] == TEST_PERIOD]

    X_train, y_train = train[feature_cols].copy(), train[TARGET]
    X_test, y_test = test[feature_cols].copy(), test[TARGET]
    for c in CAT_FEATURES:
        X_train[c] = X_train[c].astype(str)
        X_test[c] = X_test[c].astype(str)

    train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
    test_pool = Pool(X_test, y_test, cat_features=CAT_FEATURES)

    def objective(trial):
        params = dict(
            loss_function="MAE",
            random_seed=42,
            iterations=trial.suggest_int("iterations", 300, 3000),
            depth=trial.suggest_int("depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            bootstrap_type="Bayesian",
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            early_stopping_rounds=50,
            verbose=False,
        )
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=test_pool)
        preds = np.clip(model.predict(X_test), 0, None)
        return business_accuracy(y_test.values, preds)

    print(f"Running Optuna search ({N_TRIALS} trials, maximizing Business Accuracy)...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    print()
    print("Best trial:")
    print(f"  Business Accuracy: {study.best_value*100:.2f}%")
    print(f"  Params: {study.best_params}")

    print()
    print("Top 5 trials:")
    trials_df = study.trials_dataframe().sort_values("value", ascending=False)
    print(trials_df[["number", "value"] + [f"params_{k}" for k in study.best_params]].head(5).to_string(index=False))

    best_params = dict(
        loss_function="MAE", random_seed=42, bootstrap_type="Bayesian",
        early_stopping_rounds=50, verbose=False, **study.best_params,
    )
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(train_pool, eval_set=test_pool)
    preds = np.clip(final_model.predict(X_test), 0, None)
    print()
    eval_metrics(y_test, preds, "Tuned_CatBoost_Fold2")
    print("  (Lag1 on this fold: BusinessAcc=65.10%, from Phase 15)")

    final_model.save_model(f"{BASE}\\catboost_item_level_tuned.cbm")
    import json
    with open(f"{BASE}\\best_hyperparams.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print()
    print("Saved: catboost_item_level_tuned.cbm, best_hyperparams.json")


if __name__ == "__main__":
    main()
