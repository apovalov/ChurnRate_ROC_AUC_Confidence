from typing import Tuple
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score

def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:

    n = len(y)
    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        # Bootstrap sampling with replacement
        indices = np.random.choice(range(n), size=n, replace=True)
        y_bootstrap = y[indices]
        X_bootstrap = X[indices]

        # Check for presence of both classes
        if len(np.unique(y_bootstrap)) < 2:
            # Skip this iteration if only one class is present
            continue

        y_pred = classifier.predict_proba(X_bootstrap)[:, 1]
        auc = roc_auc_score(y_bootstrap, y_pred)
        bootstrapped_scores.append(auc)

    # Ensure that bootstrapped_scores has the desired length
    # by adding the original ROC AUC or repeating the last computed AUC
    while len(bootstrapped_scores) < n_bootstraps:
        original_auc = roc_auc_score(y, classifier.predict_proba(X)[:, 1])
        bootstrapped_scores.append(original_auc)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    alpha = (1.0 - conf) / 2.0
    lcb = sorted_scores[int(round(alpha * n_bootstraps))]
    ucb = sorted_scores[int(round((1.0 - alpha) * n_bootstraps))]

    return (lcb, ucb)



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Создаем искусственные данные
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.7, 0.3], random_state=42)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем классификатор
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Вычисляем доверительный интервал для ROC-AUC
lcb, ucb = roc_auc_ci(clf, X_test, y_test)

print(f"ROC-AUC confidence interval: ({lcb:.3f}, {ucb:.3f})")
