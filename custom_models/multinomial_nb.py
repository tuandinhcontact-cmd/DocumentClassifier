import numpy as np
from scipy.sparse import issparse

class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            class_counts[idx] = X_c.shape[0]
            # Sum of features for class c
            if issparse(X_c):
                feature_counts[idx, :] = np.array(X_c.sum(axis=0)).ravel()
            else:
                feature_counts[idx, :] = np.sum(X_c, axis=0)
                
        self.class_log_prior_ = np.log(class_counts / n_samples)
        
        # Laplace smoothing
        smoothed_fc = feature_counts + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)
        return self
        
    def predict_proba(self, X):
        # log_posterior = log_prior + X * log_prob^T
        if issparse(X):
            # X.dot works for scipy sparse matrices dot product
            log_likelihood = X.dot(self.feature_log_prob_.T)
        else:
            log_likelihood = np.dot(X, self.feature_log_prob_.T)
            
        log_posterior = log_likelihood + self.class_log_prior_
        
        # Softmax with log-sum-exp trick to avoid numerical issues
        max_log = np.max(log_posterior, axis=1, keepdims=True)
        exp_posterior = np.exp(log_posterior - max_log)
        return exp_posterior / np.sum(exp_posterior, axis=1, keepdims=True)
        
    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
