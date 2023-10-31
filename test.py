from sklearn.metrics import precision_recall_curve
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


from sklearn.metrics import precision_recall_curve
from joblib import Parallel, delayed
import numpy as np
from typing import Tuple, List

def bootstrap_pr_iteration(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
    y_true_bootstrap = y_true[indices]
    y_probs_bootstrap = y_probs[indices]
    precision, recall, thresholds = precision_recall_curve(y_true_bootstrap, y_probs_bootstrap)
    return precision, recall

def pr_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha = (1 - conf) / 2
    all_precisions, all_recalls = zip(*Parallel(n_jobs=-1)(
        delayed(bootstrap_pr_iteration)(y_true, y_probs) for _ in range(n_bootstrap)
    ))

    # Interpolate precision and recall values to a common set of thresholds
    common_thresholds = np.linspace(0, 1, len(y_true) + 1)
    interpolated_precisions = np.array([np.interp(common_thresholds, np.linspace(0, 1, len(p)), p) for p in all_precisions])
    interpolated_recalls = np.array([np.interp(common_thresholds, np.linspace(0, 1, len(r)), r) for r in all_recalls])

    # Calculate means and confidence intervals
    precision_mean = np.mean(interpolated_precisions, axis=0)
    recall_mean = np.mean(interpolated_recalls, axis=0)
    precision_lcb = np.percentile(interpolated_precisions, alpha * 100, axis=0)
    precision_ucb = np.percentile(interpolated_precisions, (1 - alpha) * 100, axis=0)

    return recall_mean, precision_mean, precision_lcb, precision_ucb

# Генерируем синтетические данные
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем классификатор
clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_probs = clf.predict_proba(X_test)[:, 1]

# Получаем значения кривой точность-полнота и доверительные интервалы
recall, precision, precision_lcb, precision_ucb = pr_curve(y_test, y_probs)

# print('y_test_len', len(y_test))
# print('recall', recall)
# print('precision', precision)
# print('precision_lcb', precision_lcb)
# print('precision_ucb', precision_ucb)

# Визуализируем результаты
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.fill_between(recall, precision_lcb, precision_ucb, color='blue', alpha=0.2, label='95% CI')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve with 95% Confidence Interval')
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True)
plt.show()
