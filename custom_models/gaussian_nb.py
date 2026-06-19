import numpy as np
from scipy.sparse import issparse

class CustomGaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.class_prior = None
        self.means = {}
        self.variances = {}
        
    def fit(self, X, y):
        y = np.array(y)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        self.class_prior = []
        
        # Tính toán phương sai tổng thể của đặc trưng để tính epsilon làm mịn
        if issparse(X):
            # Tính toán phương sai cho ma trận thưa: Var = E[X^2] - (E[X])^2
            mean_global = np.array(X.mean(axis=0)).ravel()
            mean_sq_global = np.array(X.power(2).mean(axis=0)).ravel()
            var_global = mean_sq_global - mean_global**2
            epsilon = self.var_smoothing * var_global.max()
        else:
            X = np.array(X)
            epsilon = self.var_smoothing * np.var(X, axis=0).max()
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_prior.append(X_c.shape[0] / n_samples)
            
            # Tính trung bình và phương sai cho từng lớp
            if issparse(X_c):
                mean = np.array(X_c.mean(axis=0)).ravel()
                mean_sq = np.array(X_c.power(2).mean(axis=0)).ravel()
                var = mean_sq - mean**2 + epsilon
            else:
                mean = np.mean(X_c, axis=0)
                var = np.var(X_c, axis=0) + epsilon
                
            self.means[c] = mean
            self.variances[c] = var
            
        return self
        
    def _calculate_log_likelihood(self, class_val, X):
        mean = self.means[class_val]
        var = self.variances[class_val]
        
        if issparse(X):
            # Công thức vectorized log-likelihood dành riêng cho ma trận thưa để tránh OOM:
            # Log PDF = sum( -0.5 * log(2 * pi * var) - mean^2 / 2var ) + X * (mean/var) - X^2 * (0.5/var)
            log_pdf_const = -0.5 * np.log(2 * np.pi * var)
            term_const = np.sum(log_pdf_const) - np.sum((mean ** 2) / (2 * var))
            
            # Tích vô hướng tuyến tính và bình phương tuyến tính trên ma trận thưa
            term_linear = X.dot(mean / var)
            term_quadratic = X.power(2).dot(0.5 / var)
            
            return term_const + term_linear - term_quadratic
        else:
            X = np.array(X)
            log_pdf = -0.5 * np.log(2 * np.pi * var) - ((X - mean) ** 2) / (2 * var)
            return np.sum(log_pdf, axis=1)
        
    def predict_proba(self, X):
        n_samples = X.shape[0]
        log_posteriors = np.zeros((n_samples, len(self.classes)))
        
        for idx, c in enumerate(self.classes):
            log_prior = np.log(self.class_prior[idx])
            log_likelihood = self._calculate_log_likelihood(c, X)
            log_posteriors[:, idx] = log_prior + log_likelihood
            
        # Dùng Softmax với log-sum-exp trick để bảo vệ độ chính xác số học
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors - max_log)
        
        probabilities = exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)
        return probabilities
        
    def predict(self, X):
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
