"""
Shared evaluation utilities for Phase 11+.

Documented assumption (business accuracy zero-handling):
  The spec does not define how rows with actual sales of zero should be
  evaluated under the +/-20% rule -- a percentage tolerance around zero is
  undefined (anything/0 is infinite or undefined). In this work, a forecast
  is treated as correct for an actual-zero row if and only if the rounded
  prediction is exactly 0. No invented slack (e.g. "actual=0, prediction<=5
  counts as correct") is applied, since that would make the reported
  accuracy incomparable to a plain reading of "+/-20%". This should be
  confirmed with stakeholders before deployment.

Documented assumption (rounding):
  PositiveSales is a whole-unit quantity. Continuous model output is
  rounded to the nearest integer before both the exact-zero check above and
  the +/-20% check on nonzero actuals -- this is a data-type correction,
  not a business tolerance.
"""
import numpy as np


def business_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    pred = np.round(np.asarray(y_pred, dtype=float))
    zero_mask = y_true == 0
    correct = np.empty(len(y_true), dtype=bool)
    correct[zero_mask] = pred[zero_mask] == 0
    nz = ~zero_mask
    correct[nz] = np.abs(pred[nz] - y_true[nz]) <= 0.2 * y_true[nz]
    return correct.mean()


def eval_metrics(y_true, y_pred, name=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    ba = business_accuracy(y_true, y_pred) * 100
    result = {"MAE": mae, "RMSE": rmse, "WAPE": wape, "BusinessAccuracy": ba}
    if name:
        print(f"  {name:28s}  MAE={mae:8.3f}  RMSE={rmse:8.3f}  WAPE={wape:7.2f}%  "
              f"BusinessAcc={ba:6.2f}%")
    return result
