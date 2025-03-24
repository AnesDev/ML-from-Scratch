import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from algorithm import SoftmaxRegression

mnist = fetch_openml('mnist_784', version=1, as_frame=True)

X, y = mnist.data, mnist.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train / 255.0
X_test = X_test / 255.0

softmax = SoftmaxRegression()
softmax.fit(X_train, y_train)

y_pred = softmax.predict(X_test)
accuracy = (y_pred == y_test).mean()

print(f"Accuracy: {accuracy * 100:.2f}%")