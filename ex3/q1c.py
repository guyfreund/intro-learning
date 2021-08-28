import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def make_fast_gauss_x_vector(sample_x, x, sigma):
    gv = np.subtract(sample_x, x)
    gv = np.square(gv)
    gv = np.sum(gv, 1)
    gv = -gv / (2 * sigma)
    gv = np.exp(gv)
    return gv


def make_gauss_x_vector(sample_x, x, sigma, m):
    gv = np.zeros((1, m))
    for j in np.arange(m):
        gv[0, j] = np.linalg.norm(x - sample_x[j, :])
    gv = np.square(gv)  # element-wise square
    gv = -1 * (1 / (2 * sigma)) * gv
    gv = np.exp(gv)  # exp g element-wise
    return gv


def get_data():
    data = loadmat('EX3q1_data.mat')
    x_train = data.get('Xtrain', {})
    y_train = data.get('Ytrain', {})
    x_test = data.get('Xtest', {})
    y_test = data.get('Ytest', {})
    return x_train, y_train, x_test, y_test


def get_alpha(i):
    if i not in range(1, 5):
        return []
    with open(f'alpha{i}.txt') as f:
        alpha = f.read()
        return np.array([float(i.strip()[:-1]) for i in alpha.split('alpha = ')[1].splitlines() if i.strip()[:-1]])


def predict(alpha, gauss_x_vector):
    return np.sign(np.inner(alpha, gauss_x_vector))


def prediction_to_color(label):
    if label >= 0:
        return 'green'
    return 'red'


def get_heatmap_details(alpha_index, sigma, sample_x):
    x1, x2 = np.mgrid[-10:11:0.01, -10:11:0.01]
    # 3D matrix where each point in 2D matrix is a vector [x1,x2]
    examples = np.stack((x1, x2), axis=2)
    # flatten one depth
    examples = examples.reshape((np.size(examples, 0) * np.size(examples, 1), 2))
    alpha = get_alpha(alpha_index)
    colors = []
    for x in examples:
        gauss_vector = make_fast_gauss_x_vector(sample_x, x, sigma)
        colors.append(prediction_to_color(predict(alpha, gauss_vector)))
    return np.matrix.flatten(x1), np.matrix.flatten(x2), colors


def main():
    x_train, y_train, x_test, y_test = get_data()
    # m = np.size(x_train, 0)
    sigmas = [0.01, 0.05, 1, 2]
    binaries = ['00', '01', '10', '11']
    fig, axs = plt.subplots(2, 2)
    for i in range(len(sigmas)):
        x, y, colors = get_heatmap_details(i + 1, sigmas[i], x_train)
        axs[int(binaries[i][0]), int(binaries[i][1])].scatter(x, y, c=colors)
        axs[int(binaries[i][0]), int(binaries[i][1])].set_title(f'\u03C3 = {sigmas[i]}')
    plt.show()


if __name__ == "__main__":
    main()
