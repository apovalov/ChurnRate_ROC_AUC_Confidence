from sklearn.metrics import precision_recall_curve, confusion_matrix
from scipy.interpolate import interp1d
from typing import Tuple
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


# def binary_search(arr, target, func):
#     low, high = 0, len(arr) - 1
#     best_val = 0
#     while low <= high:
#         mid = (low + high) // 2
#         val = func(arr, mid)
#         if val < target:
#             high = mid - 1
#         else:
#             best_val = mid
#             low = mid + 1
#     return best_val



# def sr_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_specificity: float) -> Tuple[float, float]:
#     # sort_indices = np.argsort(-y_prob)
#     y_true_sorted = y_true#[sort_indices]
#     y_prob_sorted = y_prob#[sort_indices]

#     # Предварительные вычисления
#     tp_cumsum = np.cumsum(y_true_sorted == 1)  # Кумулятивная сумма истинных положительных результатов
#     total_positive = tp_cumsum[-1]  # Общее количество положительных классов

#     threshold_index = binary_search(y_prob_sorted, min_specificity,
#                                     lambda arr, idx: specificity_at_threshold(y_true_sorted, arr, idx))
#     threshold = y_prob_sorted[threshold_index]
#     recall = tp_cumsum[threshold_index] / total_positive  # Вычисление полноты

#     return threshold, recall



# def specificity_at_threshold(y_true, y_prob, index):
#     threshold = y_prob[index]
#     y_pred = (y_prob >= threshold).astype(int)
#     tn = np.count_nonzero((y_pred == 0) & (y_true == 0))
#     fp = np.count_nonzero((y_pred == 1) & (y_true == 0))
    # return tn / (tn + fp)

def sr_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_specificity: float) -> Tuple[float, float]:
    # Сортировка массивов не требуется, так как y_prob уже отсортирован
    unique_prob = np.unique(y_prob)  # Уникальные значения вероятности
    left, right = 0, len(unique_prob) - 1  # Инициализация границ бинарного поиска

    best_threshold = 0
    max_recall = 0

    total_positives = np.sum(y_true)  # Общее количество положительных примеров
    total_negatives = len(y_true) - total_positives  # Общее количество отрицательных примеров

    while left <= right:
        mid = (left + right) // 2
        threshold = unique_prob[mid]

        # Определение индексов для текущего порога
        indices = np.where(y_prob >= threshold)[0]

        # Вычисление TN, FP, TP и FN
        tp = np.sum(y_true[indices])
        fp = len(indices) - tp
        tn = total_negatives - fp
        fn = total_positives - tp

        # Вычисление Specificity и Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Проверка условия минимальной специфичности
        if specificity >= min_specificity:
            # Обновление лучшего порога и максимальной полноты, если необходимо
            if recall > max_recall:
                best_threshold = threshold
                max_recall = recall

            # Перемещение правой границы бинарного поиска
            right = mid - 1
        else:
            # Перемещение левой границы бинарного поиска
            left = mid + 1

    return best_threshold, max_recall


def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # precision, recall, _ = precision_recall_curve(y_true, y_prob)

    # precisions_bootstrap = []

    # positive_indices = np.where(y_true == 1)[0]
    # negative_indices = np.where(y_true == 0)[0]

    # for _ in range(n_bootstrap):
    #     positive_bootstrap_indices = np.random.choice(positive_indices, len(positive_indices), replace=True)
    #     negative_bootstrap_indices = np.random.choice(negative_indices, len(negative_indices), replace=True)

    #     bootstrap_indices = np.concatenate([positive_bootstrap_indices, negative_bootstrap_indices])
    #     y_true_bootstrap = y_true[bootstrap_indices]
    #     y_prob_bootstrap = y_prob[bootstrap_indices]

    #     precision_bootstrap, recall_bootstrap, _ = precision_recall_curve(y_true_bootstrap, y_prob_bootstrap)
    #     interp_func = interp1d(recall_bootstrap, precision_bootstrap, bounds_error=False, fill_value=(1., 0.), assume_sorted=True)
    #     interpolated_precision = interp_func(recall)

    #     precisions_bootstrap.append(interpolated_precision)

    # precisions_bootstrap = np.array(precisions_bootstrap)
    # # precision_lcb = np.quantile(precisions_bootstrap, (1 - conf) / 2 * 100, axis=0)
    # # precision_ucb = np.quantile(precisions_bootstrap, (1 + conf) / 2 * 100, axis=0)

    # precision_lcb = np.quantile(precisions_bootstrap, (1 - conf) / 2, axis=0)
    # precision_ucb = np.quantile(precisions_bootstrap, (1 + conf) / 2, axis=0)


    # return recall, precision, precision_lcb, precision_ucb


    return y_true, y_prob, y_true, y_prob

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
