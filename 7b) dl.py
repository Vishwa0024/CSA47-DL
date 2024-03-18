import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_modified = X**2

X_b = np.c_[np.ones((100, 1)), X_modified]

iterations = 1000
learning_rate = 0.1
stopping_threshold = 0.0001

np.random.seed(42)
theta = np.random.randn(2, 1)

for i in range(iterations):
    gradients = 2/len(X) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    loss = np.mean((X_b.dot(theta) - y)**2)
    if np.linalg.norm(gradients) < stopping_threshold:
        break

print("Optimal parameters:", theta.flatten())
plt.scatter(X, y)
plt.scatter(X, X_b.dot(theta), color='red', marker='x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.show()
