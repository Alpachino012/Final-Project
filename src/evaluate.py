"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH, NumPy only).

scikit-learn is used ONLY in tests/test_sanity.py to sanity-check these
numbers, and for the baseline models in run_all.py.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


def regression_report(y_true_log, y_pred_log) -> dict:
    """Task A metrics on log(price) and, after exp(), on price in AZN."""
    price_true, price_pred = np.exp(y_true_log), np.exp(y_pred_log)
    return {
        "RMSE_log": rmse(y_true_log, y_pred_log),
        "MAE_log": mae(y_true_log, y_pred_log),
        "R2_log": r2(y_true_log, y_pred_log),
        "RMSE_AZN": rmse(price_true, price_pred),
        "MAE_AZN": mae(price_true, price_pred),
        "R2_AZN": r2(price_true, price_pred),
    }


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall == 0:
        return float(precision), float(recall), 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney U): the chance that a random positive gets a
    higher score than a random negative. Tied scores get their average rank.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)

    # average ranks for ties (trees give many identical probabilities)
    values, counts = np.unique(scores, return_counts=True)
    for value in values[counts > 1]:
        tied = scores == value
        ranks[tied] = ranks[tied].mean()

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_report(y_true, y_pred, scores) -> dict:
    """Task B metrics: accuracy, precision, recall, F1, ROC-AUC."""
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc(y_true, scores),
    }


# --- Comparison helpers -----------------------------------------------------
def kfold_indices(n: int, k: int = 5, seed: int = 42):
    """Shuffled k-fold split: list of (train_idx, val_idx) pairs."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    pairs = []
    for i in range(k):
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        pairs.append((train_idx, folds[i]))
    return pairs


def mean_std(values) -> str:
    """Format a list of fold scores as 'mean +/- std'."""
    return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"


def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: our models vs the scikit-learn baselines.
    name_to_metrics = {"model name": {"metric": value, ...}, ...}
    The SAME preprocessing/split is used for every row (fair comparison).
    """
    columns = []
    for metrics in name_to_metrics.values():
        for key in metrics:
            if key not in columns:
                columns.append(key)

    def fmt(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    name_width = max(len(name) for name in name_to_metrics) + 2
    widths = []
    for col in columns:
        cells = [fmt(m.get(col, "-")) for m in name_to_metrics.values()]
        widths.append(max(len(col), max(len(c) for c in cells)) + 2)

    header = "Model".ljust(name_width) + "".join(c.rjust(w) for c, w in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for name, metrics in name_to_metrics.items():
        row = name.ljust(name_width)
        for col, w in zip(columns, widths):
            row += fmt(metrics.get(col, "-")).rjust(w)
        print(row)
