import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fig = plt.figure(figsize=(8, 6))
#plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
#plt.show()

from algorithm import LinearRegression

regressor1 = LinearRegression(lr = 0.001)
regressor2 = LinearRegression(lr = 0.01)
regressor3 = LinearRegression(lr = 0.1)

regressor1.fit(X_train, y_train)
regressor2.fit(X_train, y_train)
regressor3.fit(X_train, y_train)

predicted = regressor1.predict(X_test)

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mse(y_test, predicted)

print(f"MSE: {mse_value}")

y_pred_line1 = regressor1.predict(X)
y_pred_line2 = regressor2.predict(X)
y_pred_line3 = regressor3.predict(X)

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line1, color='red', linewidth=2, label="Prediction (lr=0.001)")
plt.plot(X, y_pred_line2, color='green', linewidth=2, label="Prediction (lr=0.01)")
plt.plot(X, y_pred_line3, color='blue', linewidth=2, label="Prediction (lr=0.1)")
plt.legend()
plt.show()