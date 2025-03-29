import numpy as np
from itertools import repeat


class SoftmaxRegression:
    def __init__(self, lr=0.1, n_iters=100):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))


        # Initialize weights & bias
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        # One-hot encode y
        y_one_hot = np.eye(n_classes)[y]  


        # Gradient Descent
        for _ in repeat(None, self.n_iters):
            logits = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(logits)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_one_hot))
            db = np.mean(y_pred - y_one_hot, axis=0)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # compute final logits and their probabilities
        logits = np.dot(X, self.weights) + self.bias
        y_pred = self._softmax(logits)
        return np.argmax(y_pred, axis=1)
    
    def _softmax(self, z):
        # Calculating the softmax function for each row of the logit z
        z -= np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
