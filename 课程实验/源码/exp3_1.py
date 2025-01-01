import numpy as np
import mindspore as ms
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]

# Plot data
plt.scatter(X, y, marker='x', c='r')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Scatter plot of training data')
plt.show()

def compute_cost(X, y, theta):
    m = len(y)
    J = (1 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)
    return J

# Add a column of ones to X
X = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros(2)

# Compute initial cost
initial_cost = compute_cost(X, y, theta)
print(f'初始代价: {initial_cost}')

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))
        J_history.append(compute_cost(X, y, theta))

    return theta, J_history

# Parameters for gradient descent
alpha = 0.01
num_iters = 1500

# Perform gradient descent
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
print(f'参数向量: {theta}')

# Plot the linear fit
plt.scatter(X[:, 1], y, marker='x', c='r')
plt.plot(X[:, 1], X.dot(theta), '-')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.title('Linear regression fit')
plt.show()

def predict(theta, population):
    return np.dot([1, population], theta)

# Predict values for population sizes of 35,000 and 70,000
predict1 = predict(theta, 3.5)
predict2 = predict(theta, 7.0)

print(f'人口为35000时，收益为 {predict1 * 10000}')
print(f'人口为70000时，收益为 {predict2 * 10000}')