import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = load_boston()
X = data.data[:, 5]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mean_x = np.mean(X_train)
mean_y = np.mean(y_train)

n = len(X_train)
numer = 0
denom = 0
for i in range(n):
    numer += (X_train[i] - mean_x) * (y_train[i] - mean_y)
    denom += (X_train[i] - mean_x) ** 2

b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

def plot_regression_line(X, y, b0, b1):
    plt.scatter(X, y, color='b', marker='o', s=30)
    y_pred = b0 + b1 * X
    plt.plot(X, y_pred, color='r')
    plt.xlabel('Average Number of Rooms')
    plt.ylabel('Median House Price (in $1000s)')
    plt.title('Linear Regression')
    plt.show()
print("Estimated coefficients:\nIntercept:", b0, "\nSlope:", b1)

plot_regression_line(X_train, y_train, b0, b1)
y_pred = b0 + b1 * X_test
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
