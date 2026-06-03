import math
from collections import defaultdict


class GaussianNB:
    def __init__(self):
        self.classes = []
        self.priors = {}
        self.mean = {}
        self.var = {}

    def fit(self, X, y):
        n_samples = len(X)

        self.classes = sorted(set(y))

        data = defaultdict(list)
        for xi, yi in zip(X, y):
            data[yi].append(xi)

        for c, samples in data.items():
            self.priors[c] = len(samples) / n_samples

            features = list(zip(*samples))

            self.mean[c] = [
                sum(f) / len(f)
                for f in features
            ]

            self.var[c] = [
                sum((val - m) ** 2 for val in f) / len(f)
                for f, m in zip(features, self.mean[c])
            ]

    def _log_pdf(self, x, mean, var):
        eps = 1e-9
        var = var + eps

        return -0.5 * math.log(2.0 * math.pi * var) - ((x - mean) ** 2) / (2.0 * var)

    def predict(self, X):
        preds = []

        for xi in X:
            posteriors = {}

            for c in self.classes:
                log_prob = math.log(self.priors[c])

                for i, x_val in enumerate(xi):
                    log_prob += self._log_pdf(
                        x_val,
                        self.mean[c][i],
                        self.var[c][i]
                    )

                posteriors[c] = log_prob

            preds.append(max(posteriors, key=posteriors.get))

        return preds