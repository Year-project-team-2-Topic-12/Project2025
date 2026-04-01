from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.5, positive_class=1):
        self.model = model
        self.threshold = threshold
        self.positive_class = positive_class

    def fit(self, X, y):
        self.model_ = clone(self.model)
        self.model_.fit(X, y)
        self.classes_ = getattr(self.model_, "classes_", None)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        check_is_fitted(self, "model_")
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(X)
        # если у базовой модели нет decision_function — вернём вероятность positive_class
        proba = self.predict_proba(X)
        cls_idx = list(self.model_.classes_).index(self.positive_class)
        return proba[:, cls_idx]

    def predict(self, X):
        scores = self.decision_function(X)
        # если это probability, threshold логичный; если это logit-margin — лучше use_decision_function=False
        return (scores >= self.threshold).astype(int)