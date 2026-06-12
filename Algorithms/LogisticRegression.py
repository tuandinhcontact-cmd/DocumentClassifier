import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.w = np.zeros((n_classes, n_features))
        self.b = np.zeros(n_classes)

        for class_index, class_label in enumerate(self.classes_):
            y_binary = (y == class_label).astype(float)

            w = np.zeros(n_features)
            b = 0.0

            for _ in range(self.n_iters):
                linear_model = np.dot(X, w) + b
                linear_model = np.clip(linear_model, -500, 500)

                y_pred = 1 / (1 + np.exp(-linear_model))

                error = y_pred - y_binary

                dw = np.dot(X.T, error) / n_samples
                db = np.mean(error)

                w -= self.lr * dw
                b -= self.lr * db

            self.w[class_index] = w
            self.b[class_index] = b

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        linear_model = np.dot(X, self.w.T) + self.b
        linear_model = np.clip(linear_model, -500, 500)

        probabilities = 1 / (1 + np.exp(-linear_model))

        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1

        probabilities = probabilities / row_sums

        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)

        return self.classes_[class_indices]