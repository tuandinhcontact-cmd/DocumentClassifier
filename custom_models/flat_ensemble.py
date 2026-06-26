import numpy as np
import copy


class CustomOneVsRestClassifier:
    """OVR wrapper cho các binary classifier."""
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.models = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = (y == cls).astype(int)
            model = copy.deepcopy(self.base_estimator)
            model.fit(X, y_binary)
            self.models[cls] = model
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.zeros((n, len(self.classes)))
        for i, cls in enumerate(self.classes):
            probs[:, i] = self.models[cls].predict_proba(X)[:, 1]
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]


class FlatSoftVotingClassifier:
    """Soft Voting ensemble: trung bình có trọng số xác suất của các model thành phần."""
    def __init__(self, estimators, weights=None):
        self.estimators = estimators  # [(name, model), ...]
        self.weights = weights        # [w1, w2, ...]
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        for name, est in self.estimators:
            est.fit(X, y)
        return self

    def predict_proba(self, X):
        probs_list = [est.predict_proba(X) for _, est in self.estimators]
        if self.weights is None:
            return np.mean(probs_list, axis=0)
        else:
            w_arr = np.array(self.weights)
            w_normalized = w_arr / np.sum(w_arr)
            weighted_probs = np.zeros_like(probs_list[0])
            for w, probs in zip(w_normalized, probs_list):
                weighted_probs += w * probs
            return weighted_probs

    def predict(self, X):
        probs = self.predict_proba(X)
        cls_arr = next(
            (e.classes for _, e in self.estimators if hasattr(e, 'classes')),
            self.classes
        )
        return cls_arr[np.argmax(probs, axis=1)]
