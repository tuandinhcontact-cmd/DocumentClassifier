import numpy as np
import copy

class CustomOneVsRestClassifier:
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
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))
        
        for idx, cls in enumerate(self.classes):
            prob = self.models[cls].predict_proba(X)[:, 1]
            probs[:, idx] = prob
            
        # L1 Normalization
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
        
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

class CustomMultiClassVotingClassifier:
    def __init__(self, estimators):
        self.estimators = estimators # List of tuples: (name, estimator_object)
        
    def fit(self, X, y):
        for name, est in self.estimators:
            print(f"    -> Training {name} for MultiClassVoting...")
            est.fit(X, y)
        self.classes = self.estimators[0][1].classes
        return self
        
    def predict_proba(self, X):
        all_probs = []
        for name, est in self.estimators:
            all_probs.append(est.predict_proba(X))
        return np.mean(all_probs, axis=0)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
