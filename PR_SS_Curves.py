from sklearn.metrics import precision_recall_curve, confusion_matrix
import numpy as np
from typing import Tuple

def pr_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1)

    index = np.where(precision >= min_precision)[0]
    max_recall = recall[index].max()
    threshold_proba = thresholds[index[np.where(recall == max_recall)][0].max()]

    return threshold_proba, max_recall

def sr_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_specificity: float) -> Tuple[float, float]:
    threshold_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    sorted_probs = y_prob[threshold_indices]
    sorted_true = y_true[threshold_indices]

    recall = []
    specificity = []
    for threshold in sorted_probs:
        predicted = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
        recall.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))

    specificity = np.array(specificity)
    recall = np.array(recall)
    index = np.where(specificity >= min_specificity)[0]
    max_recall = recall[index].max()
    threshold_proba = sorted_probs[index[np.where(recall == max_recall)][0].max()]

    return threshold_proba, max_recall


def pr_curve(y_true: np.ndarray, y_prob: np.ndarray, conf: float = 0.95, n_bootstrap: int = 10_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    precisions_bootstrap = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
        y_true_bootstrap = y_true[indices]
        y_prob_bootstrap = y_prob[indices]

        precision_bootstrap, recall_bootstrap, _ = precision_recall_curve(y_true_bootstrap, y_prob_bootstrap)
        interpolated_precision = np.interp(recall, recall_bootstrap, precision_bootstrap)
        precisions_bootstrap.append(interpolated_precision)

    precisions_bootstrap = np.array(precisions_bootstrap)
    precision_lcb = np.percentile(precisions_bootstrap, (1 - conf) / 2 * 100, axis=0)
    precision_ucb = np.percentile(precisions_bootstrap, (1 + conf) / 2 * 100, axis=0)

    return recall, precision, precision_lcb, precision_ucb

def sr_curve(y_true: np.ndarray, y_prob: np.ndarray, conf: float = 0.95, n_bootstrap: int = 10_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    threshold_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    sorted_probs = y_prob[threshold_indices]
    sorted_true = y_true[threshold_indices]

    recall = []
    specificity = []
    for threshold in sorted_probs:
        predicted = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
        recall.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))

    specificity_bootstrap_list = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(range(len(y_true)), len(y_true), replace=True)
        y_true_bootstrap = y_true[indices]
        y_prob_bootstrap = y_prob[indices]

        threshold_indices_bootstrap = np.argsort(y_prob_bootstrap, kind="mergesort")[::-1]
        sorted_probs_bootstrap = y_prob_bootstrap[threshold_indices_bootstrap]
        sorted_true_bootstrap = y_true_bootstrap[threshold_indices_bootstrap]

        recall_bootstrap = []
        specificity_bootstrap = []
        for threshold in sorted_probs_bootstrap:
            predicted = (y_prob_bootstrap >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_bootstrap, predicted).ravel()
            recall_bootstrap.append(tp / (tp + fn))
            specificity_bootstrap.append(tn / (tn + fp))

        interpolated_specificity = np.interp(recall, recall_bootstrap, specificity_bootstrap)
        specificity_bootstrap_list.append(interpolated_specificity)

    specificity_bootstrap_array = np.array(specificity_bootstrap_list)
    specificity_lcb = np.percentile(specificity_bootstrap_array, (1 - conf) / 2 * 100, axis=0)
    specificity_ucb = np.percentile(specificity_bootstrap_array, (1 + conf) / 2 * 100, axis=0)

    return np.array(recall), np.array(specificity), specificity_lcb, specificity_ucb


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
