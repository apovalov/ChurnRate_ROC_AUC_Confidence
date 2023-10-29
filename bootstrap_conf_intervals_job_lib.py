from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import numpy as np

def bootstrap_iteration(y_true: np.ndarray, y_probs: np.ndarray):
    n = len(y_true)
    bootstrap_indices = np.random.choice(np.arange(n), size=n, replace=True)
    y_bootstrap = y_true[bootstrap_indices]
    y_probs_bootstrap = y_probs[bootstrap_indices]

    while len(np.unique(y_bootstrap)) < 2:
        bootstrap_indices = np.random.choice(np.arange(n), size=n, replace=True)
        y_bootstrap = y_true[bootstrap_indices]
        y_probs_bootstrap = y_probs[bootstrap_indices]

    return roc_auc_score(y_bootstrap, y_probs_bootstrap)

def roc_auc_ci(classifier: ClassifierMixin, X: np.ndarray, y: np.ndarray, conf: float = 0.95, n_bootstraps: int = 10_000):
    alpha = (1.0 - conf) / 2.0

    # Предварительно вычисляем вероятности для всей тестовой выборки
    y_probs = classifier.predict_proba(X)[:, 1]

    bootstrapped_scores = Parallel(n_jobs=-1)(
        delayed(bootstrap_iteration)(y, y_probs) for _ in range(n_bootstraps)
    )

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lcb = sorted_scores[int(alpha * n_bootstraps)]
    ucb = sorted_scores[int((1 - alpha) * n_bootstraps)]

    return (lcb, ucb)




################################################################
################################################################
################################################################

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# # Создаем искусственные данные
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.7, 0.3], random_state=42)

# # Разделяем данные на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Обучаем классификатор
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Вычисляем доверительный интервал для ROC-AUC
# lcb, ucb = roc_auc_ci(clf, X_test, y_test)

# print(f"ROC-AUC confidence interval: ({lcb:.3f}, {ucb:.3f})")


# # Подсчет количества примеров каждого класса
# class_counts = np.bincount(y)

# # Вычисление процентного соотношения
# class_0_pct = (class_counts[0] / len(y)) * 100
# class_1_pct = (class_counts[1] / len(y)) * 100

# print(f"Class 0: {class_counts[0]} ({class_0_pct:.2f}%)")
# print(f"Class 1: {class_counts[1]} ({class_1_pct:.2f}%)")