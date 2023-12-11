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


def bootstrap_pr(y_true: np.ndarray,
                 y_probs: np.ndarray,
                 n_bootstrap: int = 10_000) -> Tuple[List[float], List[float]]:
    all_precisions, all_recalls = [], []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_bootstrap = y_true[indices]
        y_probs_bootstrap = y_probs[indices]

        # precision, recall, _ = precision_recall_curve(y_true_bootstrap, y_probs_bootstrap)
        # Расчет накопительных сумм
        tp_cumsum = np.cumsum(y_true_bootstrap)  # True Positives
        fp_cumsum = np.arange(1, len(y_true_bootstrap) + 1) - tp_cumsum  # False Positives

        # Расчет Precision и Recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)
        recall = tp_cumsum / np.sum(y_true_bootstrap)

        all_precisions.append(np.mean(precision))
        all_recalls.append(np.mean(recall))

    return all_precisions, all_recalls

def pr_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000
) -> Tuple[float, float, float, float]:
    alpha = (1 - conf) / 2
    all_precisions, all_recalls = bootstrap_pr(y_true, y_probs, n_bootstrap)

    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)

    print(alpha * 100, (1 - alpha) * 100)

    # Использование bootstrap для вычисления доверительных интервалов:
    precision_ci_lower = np.percentile(all_precisions, alpha * 100)
    precision_ci_upper = np.percentile(all_precisions, (1 - alpha) * 100)

    return mean_recall, mean_precision, precision_ci_lower, precision_ci_upper




def sr_curve(y_true: np.ndarray, y_prob: np.ndarray, conf: float = 0.95, n_bootstrap: int = 10_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    return y_true, y_prob, y_true, y_prob


