from typing import Tuple
import numpy as np

def bootstrap_sr(y_true: np.ndarray, y_probs: np.ndarray):
    indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
    y_true_bootstrap = y_true[indices]
    y_probs_bootstrap = y_probs[indices]

    tp_cumsum = np.cumsum(y_true_bootstrap)  # True Positives
    fp_cumsum = np.arange(1, len(y_true_bootstrap) + 1) - tp_cumsum  # False Positives
    tn = len(y_true_bootstrap) - np.sum(y_true_bootstrap) - fp_cumsum  # True Negatives

    specificity = tn / (tn + fp_cumsum)
    recall = tp_cumsum / np.sum(y_true_bootstrap)

    return specificity, recall

def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    tp_cumsum = np.cumsum(y_true)  # True Positives
    fp_cumsum = np.arange(1, len(y_true) + 1) - tp_cumsum  # False Positives
    tn = len(y_true) - np.sum(y_true) - fp_cumsum  # True Negatives

    specificity = tn / (tn + fp_cumsum)
    recall = tp_cumsum / np.sum(y_true)

    bootstrapped_specificities = []
    for _ in range(n_bootstrap):
        s, _ = bootstrap_sr(y_true, y_prob)
        bootstrapped_specificities.append(s)

    bootstrapped_specificities = np.array(bootstrapped_specificities)
    alpha = (1.0 - conf) / 2.0
    specificity_lcb = np.percentile(bootstrapped_specificities, alpha * 100, axis=0)
    specificity_ucb = np.percentile(bootstrapped_specificities, (1 - alpha) * 100, axis=0)

    return recall, specificity, specificity_lcb, specificity_ucb








import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
\
# [Вставьте здесь определения всех четырех функций]

# Генерация данных
X, y = make_classification(n_samples=5000, n_features=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем классификатор
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_probs = clf.predict_proba(X_test)[:, 1]

# # 1. pr_threshold
# threshold_proba, max_recall = pr_threshold(y_true, y_prob, 0.8)
# print(f'PR Threshold: {threshold_proba:.3f}, Max Recall: {max_recall:.3f}')

# # 2. sr_threshold
# threshold_proba, max_recall = sr_threshold(y_true, y_prob, 0.8)
# print(f'SR Threshold: {threshold_proba:.3f}, Max Recall: {max_recall:.3f}')

# 3. pr_curve
recall, precision, precision_lcb, precision_ucb = sr_curve(y_test, y_probs)

print('recall', recall)
print('precision', precision)
print('precision_lcb', precision_lcb)
print('precision_ucb', precision_ucb)


# plt.figure(figsize=(10, 6))
# plt.plot(recall, precision, label="Precision-Recall Curve")
# plt.fill_between(recall, precision_lcb, precision_ucb, color='skyblue', alpha=0.4, label=f'{0.95*100:.0f}% Confidence Interval')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend()
# plt.title("Precision-Recall Curve with Confidence Interval")
# plt.show()

# 4. sr_curve
# recall, spec, spec_lcb, spec_ucb = sr_curve(y_true, y_prob)

# plt.figure(figsize=(10, 6))
# plt.plot(recall, spec, label="Sensitivity-Specificity Curve")
# plt.fill_between(recall, spec_lcb, spec_ucb, color='lightgreen', alpha=0.4, label=f'{0.95*100:.0f}% Confidence Interval')
# plt.xlabel('Recall (Sensitivity)')
# plt.ylabel('Specificity')
# plt.legend()
# plt.title("Sensitivity-Specificity Curve with Confidence Interval")
# plt.show()
