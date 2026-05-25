import math

class LogisticRegression:
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.w = [0.0 for _ in range(n_features)]
        loss = 0.0

        for _ in range(self.n_iters):
            dw = [0.0] * n_features
            db = 0.0
            for (xi, yi) in zip(X, y):
                linear_model = sum(wi * xi_j for wi, xi_j in zip(self.w, xi)) + self.b
                y_pred = 1 / (1 + math.exp(-linear_model))

                error = y_pred - yi
                for j in range(n_features):
                    dw[j] += (1 / n_samples) * error * xi[j]

                db += (1/n_samples) * error
                loss += -(yi * math.log(y_pred) + (1 - yi) * math.log(1 - y_pred))

            for j in range(n_features):
                self.w[j] -= self.lr * dw[j]

            loss /= n_samples
            print(loss)
            self.b -= self.lr * db

    def predict(self, X):
        predictions = []

        for xi in X:
            linear_model = sum(wi * xi_j for wi, xi_j in zip(self.w, xi)) + self.b
            y_pred = 1 / (1 + math.exp(-linear_model))

            if (y_pred > 0.5): predictions.append(1)
            else: predictions.append(0)

        return predictions
