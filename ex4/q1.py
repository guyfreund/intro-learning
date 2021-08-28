import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def make_graph(training_set_sizes, lambdas):
    plt.plot(training_set_sizes, lambdas, marker='o')
    plt.xticks(training_set_sizes)
    plt.yticks(lambdas)
    plt.xlabel('Training set size - m')
    plt.ylabel('\u03BB that minimizes the MSE')
    plt.show()


def mean_squared_error(w, n, x_test, y_test):
    # returns mse = 1/n * (norm(x'w - y))^2
    mse_rooted = np.linalg.norm((np.transpose(x_test) @ w) - y_test)
    return (1 / n) * (mse_rooted ** 2)


def ridge_regression(x_train, y_train, lambda_val, d):
    # returns w = inv(xx' + lambda * Idxd) * x * y
    return np.linalg.inv(x_train @ np.transpose(x_train) + lambda_val * np.eye(d)) @ x_train @ y_train


def get_data():
    data = loadmat('regdata.mat')
    # x_train columns are examples
    x_train = data.get('X', {})
    y_train = data.get('Y', {})
    # x_test columns are examples
    x_test = data.get('Xtest', {})
    y_test = data.get('Ytest', {})
    d = np.size(x_train, 0)
    m = np.size(x_train, 1)
    n = np.size(x_test, 1)
    return x_train, y_train, x_test, y_test, m, n, d


def main():
    x_train, y_train, x_test, y_test, m, n, d = get_data()
    training_set_sizes = np.arange(10, 101, 10)
    lambdas = np.empty((0, 10))
    for ts in range(10, 101, 10):
        # getting the examples/labels in the first ts indexes
        x_train_set, y_train_set = x_train[:, 0:ts], y_train[0:ts, :]
        min_mse, min_l = 100, 0
        for l in range(0, 31):
            w = ridge_regression(x_train_set, y_train_set, l, d)
            mse = mean_squared_error(w, n, x_test, y_test)
            if min_mse > mse:
                min_mse = mse
                min_l = l
        lambdas = np.append(lambdas, min_l)
    make_graph(training_set_sizes, lambdas)


if __name__ == "__main__":
    main()
