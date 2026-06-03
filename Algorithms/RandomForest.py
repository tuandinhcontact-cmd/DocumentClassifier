import random
from collections import Counter
from sklearn.tree import DecisionTreeClassifier as DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.sample_size = sample_size
        self.trees = []

    def _sample(self, X, y):
        n = len(X) if self.sample_size is None else self.sample_size
        idxs = [random.randrange(len(X)) for _ in range(n)]
        return [X[i] for i in idxs], [y[i] for i in idxs]

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            X_samp, y_samp = self._sample(X, y)

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features="sqrt"
            )

            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = [tree.predict(X) for tree in self.trees]

        preds = []
        for i in range(len(X)):
            votes = [tree_preds[t][i] for t in range(self.n_trees)]
            final_vote = Counter(votes).most_common(1)[0][0]
            preds.append(final_vote)

        return preds