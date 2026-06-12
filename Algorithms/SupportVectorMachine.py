class SupportVectorMachine:
    def __init__(self, lr=0.001, n_iters=1000, C=1.0):
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.w = None
        self.b = 0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.w = [0.0 for _ in range(n_features)]
        self.b = 0.0

        for _ in range(self.n_iters):
            dw = [0.0] * n_features
            db = 0.0

            for xi, yi in zip(X, y):
                linear_model = sum(wi * xi_j for wi, xi_j in zip(self.w, xi)) + self.b
                condition = yi * linear_model

                if condition < 1:
                    for j in range(n_features):
                        dw[j] += self.C * (-yi * xi[j])
                    db += self.C * (-yi)

            for j in range(n_features):
                dw[j] += 2 * self.w[j]

            # update
            for j in range(n_features):
                self.w[j] -= self.lr * (dw[j] / n_samples)

            self.b -= self.lr * (db / n_samples)

    def predict(self, X):
        preds = [
            sum(wi * xi_j for wi, xi_j in zip(self.w, xi)) + self.b
            for xi in X
        ]
        return [1 if p >= 0 else -1 for p in preds]