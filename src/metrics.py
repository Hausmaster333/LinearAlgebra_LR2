from __future__ import annotations
import numpy as np

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    counts = confusion_counts(y_true, y_pred)
    denom = counts["tp"] + counts["fp"]
    return 0.0 if denom == 0 else counts["tp"] / denom

def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    counts = confusion_counts(y_true, y_pred)
    denom = counts["tp"] + counts["fn"]
    return 0.0 if denom == 0 else counts["tp"] / denom

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)

def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    positives = np.sum(y_sorted == 1)
    negatives = np.sum(y_sorted == 0)
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tpr = [0.0]
    fpr = [0.0]
    tp = 0
    fp = 0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return np.array(fpr), np.array(tpr)

def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = roc_curve_points(y_true, scores)
    return float(np.trapezoid(tpr, fpr))

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
    values = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if scores is not None:
        values["roc_auc"] = roc_auc_score(y_true, scores)
    return values
