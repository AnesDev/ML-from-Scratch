import numpy as np
from itertools import repeat

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Compute mean and standard deviation"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        """Scale data using stored mean & std deviation"""
        return (X - self.mean) / (self.std)

    def fit_transform(self, X):
        """Fit and transform data"""
        self.fit(X)
        return self.transform(X)
    

class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
         """init parameters"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
         """gradient descent"""
        for _ in repeat(None, self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = 1/(n_samples) * np.dot(X.T, (y_predicted - y))
            db = np.mean(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db


    def predict(self, X):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            return (y_predicted >= 0.5).astype(int)

    
    def _sigmoid(self, x):
        return 1/ (1 + np.exp(-x))
