# Count-level evaluation metrics.

import numpy as np


def compute_metrics(y_true, y_pred):
    """
    Compute step-count evaluation metrics.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth step counts per window.
    y_pred : array-like, shape (n_samples,)
        Predicted step counts per window.

    Returns
    -------
    dict
        MAE, undercount rate, overcount rate, error rate, count ratio, and
        mean relative accuracy.
    """
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    diff = y_pred - y_true

    mae = float(np.mean(np.abs(diff)))
    uc = float(np.mean(diff < 0) * 100)
    oc = float(np.mean(diff > 0) * 100)
    er_mean = float(np.mean(diff / np.clip(y_true, 1, None)) * 100)
    rca = float(np.mean(y_pred / np.clip(y_true, 1, None)))
    acc_mean = float(
        np.mean((1 - np.abs(diff / np.clip(y_true, 1, None))) * 100)
    )

    return dict(
        MAE=mae,
        UC=uc,
        OC=oc,
        ER_mean=er_mean,
        RCA=rca,
        ACC_mean=acc_mean
    )
