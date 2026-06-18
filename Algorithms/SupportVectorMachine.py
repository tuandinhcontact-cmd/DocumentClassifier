import numpy as np


class SupportVectorMachine:
    """
    Multiclass Linear SVM dùng mini-batch gradient descent.
    Train 1 model duy nhất cho nhiều class.
    """

    def __init__(
        self,
        lr=0.01,
        n_iters=100,
        C=1.0,
        reg_strength=0.0001,
        batch_size=512,
        verbose=True
    ):
        self.lr = lr
        self.n_iters = n_iters
        self.C = C
        self.reg_strength = reg_strength
        self.batch_size = batch_size
        self.verbose = verbose

        self.W = None
        self.b = None
        self.classes = None
        self.class_to_index = None
        self.index_to_class = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.class_to_index = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        self.index_to_class = {
            idx: cls for idx, cls in enumerate(self.classes)
        }

        y_idx = np.array([self.class_to_index[label] for label in y])

        self.W = np.zeros((n_classes, n_features), dtype=np.float64)
        self.b = np.zeros(n_classes, dtype=np.float64)

        for epoch in range(self.n_iters):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                X_batch = X[batch_idx]
                y_batch = y_idx[batch_idx]

                batch_size_actual = X_batch.shape[0]

                scores = np.dot(X_batch, self.W.T) + self.b

                correct_scores = scores[np.arange(batch_size_actual), y_batch].reshape(-1, 1)

                margins = scores - correct_scores + 1
                margins[np.arange(batch_size_actual), y_batch] = 0
                margins = np.maximum(0, margins)

                data_loss = np.mean(np.sum(margins, axis=1))
                reg_loss = self.reg_strength * np.sum(self.W * self.W)
                loss = self.C * data_loss + reg_loss

                epoch_loss += loss
                n_batches += 1

                mask = (margins > 0).astype(float)
                row_sum = np.sum(mask, axis=1)
                mask[np.arange(batch_size_actual), y_batch] = -row_sum

                dW = np.dot(mask.T, X_batch) / batch_size_actual
                db = np.mean(mask, axis=0)

                dW = self.C * dW + 2 * self.reg_strength * self.W
                db = self.C * db

                self.W -= self.lr * dW
                self.b -= self.lr * db

            avg_epoch_loss = epoch_loss / n_batches

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_iters}, Loss: {avg_epoch_loss:.4f}")

        return self

    def decision_function(self, X):
        X = np.array(X, dtype=np.float64)
        return np.dot(X, self.W.T) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        pred_indices = np.argmax(scores, axis=1)
        return np.array([self.index_to_class[idx] for idx in pred_indices])