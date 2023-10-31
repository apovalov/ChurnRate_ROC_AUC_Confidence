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

    # Использование bootstrap для вычисления доверительных интервалов:
    precision_ci_lower = np.percentile(all_precisions, alpha * 100)
    precision_ci_upper = np.percentile(all_precisions, (1 - alpha) * 100)

    return mean_recall, mean_precision, precision_ci_lower, precision_ci_upper




def sr_curve(y_true: np.ndarray, y_prob: np.ndarray, conf: float = 0.95, n_bootstrap: int = 10_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # threshold_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    # sorted_probs = y_prob[threshold_indices]
    # sorted_true = y_true[threshold_indices]

    # recall = []
    # specificity = []
    # for threshold in sorted_probs:
    #     predicted = (y_prob >= threshold).astype(int)
    #     tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
    #     recall.append(tp / (tp + fn))
    #     specificity.append(tn / (tn + fp))

    # specificity_bootstrap_list = []

    # for _ in range(n_bootstrap):
    #     indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
    #     y_true_bootstrap = y_true[indices]
    #     y_prob_bootstrap = y_prob[indices]

    #     threshold_indices_bootstrap = np.argsort(y_prob_bootstrap, kind="mergesort")[::-1]
    #     sorted_probs_bootstrap = y_prob_bootstrap[threshold_indices_bootstrap]
    #     sorted_true_bootstrap = y_true_bootstrap[threshold_indices_bootstrap]

    #     recall_bootstrap = []
    #     specificity_bootstrap = []
    #     for threshold in sorted_probs_bootstrap:
    #         predicted = (y_prob_bootstrap >= threshold).astype(int)
    #         tn, fp, fn, tp = confusion_matrix(y_true_bootstrap, predicted).ravel()
    #         recall_bootstrap.append(tp / (tp + fn))
    #         specificity_bootstrap.append(tn / (tn + fp))

    #     interpolated_specificity = np.interp(recall, recall_bootstrap, specificity_bootstrap)
    #     specificity_bootstrap_list.append(interpolated_specificity)

    # specificity_bootstrap_array = np.array(specificity_bootstrap_list)
    # specificity_lcb = np.percentile(specificity_bootstrap_array, (1 - conf) / 2 * 100, axis=0)
    # specificity_ucb = np.percentile(specificity_bootstrap_array, (1 + conf) / 2 * 100, axis=0)

    # return np.array(recall), np.array(specificity), specificity_lcb, specificity_ucb

    return y_true, y_prob, y_true, y_prob


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from sklearn.datasets import make_classification

# # [Вставьте здесь определения всех четырех функций]

# # Генерация данных
# y_true, y_prob = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.5, 0.5], random_state=42)
# # y_prob = y_prob[:, 1]  # Вероятности для класса 1

# # 1. pr_threshold
# threshold_proba, max_recall = pr_threshold(y_true, y_prob, 0.8)
# print(f'PR Threshold: {threshold_proba:.3f}, Max Recall: {max_recall:.3f}')

# # 2. sr_threshold
# threshold_proba, max_recall = sr_threshold(y_true, y_prob, 0.8)
# print(f'SR Threshold: {threshold_proba:.3f}, Max Recall: {max_recall:.3f}')

# # 3. pr_curve
# recall, precision, precision_lcb, precision_ucb = pr_curve(y_true, y_prob)

# plt.figure(figsize=(10, 6))
# plt.plot(recall, precision, label="Precision-Recall Curve")
# plt.fill_between(recall, precision_lcb, precision_ucb, color='skyblue', alpha=0.4, label=f'{0.95*100:.0f}% Confidence Interval')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend()
# plt.title("Precision-Recall Curve with Confidence Interval")
# plt.show()

# # 4. sr_curve
# recall, spec, spec_lcb, spec_ucb = sr_curve(y_true, y_prob)

# plt.figure(figsize=(10, 6))
# plt.plot(recall, spec, label="Sensitivity-Specificity Curve")
# plt.fill_between(recall, spec_lcb, spec_ucb, color='lightgreen', alpha=0.4, label=f'{0.95*100:.0f}% Confidence Interval')
# plt.xlabel('Recall (Sensitivity)')
# plt.ylabel('Specificity')
# plt.legend()
# plt.title("Sensitivity-Specificity Curve with Confidence Interval")
# plt.show()
