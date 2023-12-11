from sklearn.metrics import precision_recall_curve, confusion_matrix
from scipy.interpolate import interp1d
from typing import Tuple, List
import numpy as np


def pr_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float) -> Tuple[float, float]:
    # Сортировка массивов
    # sort_indices = np.argsort(-y_prob)  # Сортировка по убыванию
    y_true_sorted = y_true#[sort_indices]
    y_prob_sorted = y_prob#[sort_indices]

    # Расчет накопительных сумм
    tp_cumsum = np.cumsum(y_true_sorted)  # True Positives
    fp_cumsum = np.arange(1, len(y_true) + 1) - tp_cumsum  # False Positives

    # Расчет Precision и Recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / np.sum(y_true)

    # Поиск порога, соответствующего минимальной точности
    valid_indices = np.where(precision >= min_precision)[0]
    if valid_indices.size == 0:
        return 0, 0  # Нет подходящего порога
    best_index = valid_indices[np.argmax(recall[valid_indices])]

    return y_prob_sorted[best_index], recall[best_index]

def sr_threshold(y_true: np.ndarray,
                 y_prob: np.ndarray,
                 min_specificity: float) -> Tuple[float, float]:
    """Returns threshold and recall (from Specificity-Recall Curve)"""
    # calculate True Positive as a cumulative amount
    tp = np.cumsum(y_true)
    # calculate recall
    recall = tp / tp[-1]
    # calculate the cumulative amount of negative class
    negatives = np.cumsum(y_true == 0)
    # set left border
    left = -1
    # calculate the length of the dataset and right border
    right = y_true.shape[0]
    # through the loop compare the specificity
    while left + 1 < right:
        # calculate the middle
        middle = (left + right) // 2
        # calculate specificity
        specificity = 1 - negatives[middle] / negatives[-1]
        # compare the specifics at the middle point
        if specificity < min_specificity:
            right = middle
        elif specificity > min_specificity:
            left = middle
        else:
            left = middle
            break
    # override threshold_proba and max_recall
    threshold_proba = y_prob[left]
    max_recall = recall[left]

    return threshold_proba, max_recall


def bootstrap_pr(y_true: np.ndarray, y_probs: np.ndarray):
    indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
    y_true_bootstrap = y_true[indices]
    y_probs_bootstrap = y_probs[indices]

    tp_cumsum = np.cumsum(y_true_bootstrap)  # True Positives
    fp_cumsum = np.arange(1, len(y_true_bootstrap) + 1) - tp_cumsum  # False Positives

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / np.sum(y_true_bootstrap)

    return precision, recall

def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    tp_cumsum = np.cumsum(y_true)  # True Positives
    fp_cumsum = np.arange(1, len(y_true) + 1) - tp_cumsum  # False Positives

    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / np.sum(y_true)

    bootstrapped_precisions = []
    for _ in range(n_bootstrap):
        p, _ = bootstrap_pr(y_true, y_prob)
        bootstrapped_precisions.append(p)

    bootstrapped_precisions = np.array(bootstrapped_precisions)
    alpha = (1.0 - conf) / 2.0
    precision_lcb = np.percentile(bootstrapped_precisions, alpha * 100, axis=0)
    precision_ucb = np.percentile(bootstrapped_precisions, (1 - alpha) * 100, axis=0)

    return recall, precision, precision_lcb, precision_ucb


def bootstrap_sr(y_true: np.ndarray, y_probs: np.ndarray):
    indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
    y_true_bootstrap = y_true[indices]
    y_probs_bootstrap = y_probs[indices]

    # tp_cumsum = np.cumsum(y_true_bootstrap)  # True Positives
    # fp_cumsum = np.arange(1, len(y_true_bootstrap) + 1) - tp_cumsum  # False Positives

    # tn = len(y_true_bootstrap) - np.sum(y_true_bootstrap) - fp_cumsum  # True Negatives

    # specificity = tn / (tn + fp_cumsum)
    # recall = tp_cumsum / np.sum(y_true_bootstrap)

    tp = np.cumsum(y_true_bootstrap)
    # calculate recall
    recall = tp / tp[-1]
    # calculate the cumulative amount of negative class
    negatives = np.cumsum(y_true_bootstrap == 0)
    # set left border
    specificity = 1 - negatives / negatives[-1]



    return specificity, recall

def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # tp_cumsum = np.cumsum(y_true)  # True Positives
    # fp_cumsum = np.arange(1, len(y_true) + 1) - tp_cumsum  # False Positives
    # tn = len(y_true) - np.sum(y_true) - fp_cumsum  # True Negatives

    # specificity = tn / (tn + fp_cumsum)
    # recall = tp_cumsum / np.sum(y_true)
    tp = np.cumsum(y_true)
    # calculate recall
    recall = tp / tp[-1]
    # calculate the cumulative amount of negative class
    negatives = np.cumsum(y_true == 0)
    # set left border
    specificity = 1 - negatives / negatives[-1]


    bootstrapped_specificities = []
    for _ in range(n_bootstrap):
        s, _ = bootstrap_sr(y_true, y_prob)
        bootstrapped_specificities.append(s)

    bootstrapped_specificities = np.array(bootstrapped_specificities)
    alpha = (1.0 - conf) / 2.0
    specificity_lcb = np.percentile(bootstrapped_specificities, alpha * 100, axis=0)
    specificity_ucb = np.percentile(bootstrapped_specificities, (1 - alpha) * 100, axis=0)

    return recall, specificity, specificity_lcb, specificity_ucb



