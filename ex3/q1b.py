import numpy as np
import matplotlib.pyplot as plt


def main():
    e_cross_val = np.array([0.0800, 0.0635, 0.0680, 0.0830, 0.0800, 0.0630, 0.0830, 0.1330, 0.0800, 0.0630, 0.0835, 0.1395], dtype=float)
    e_test = np.array([0.0600, 0.0400, 0.0500, 0.0550, 0.0600, 0.0400, 0.0600, 0.1450, 0.0600, 0.0400, 0.0600, 0.1500], dtype=float)
    abs_diff = np.abs(e_cross_val - e_test)
    lambda_sigma_tuples = []
    for l in [0.01, 0.1, 1]:
        for s in [0.01, 0.05, 1, 2]:
            lambda_sigma_tuples.append(f'\u03BB: {l}, \u03C3: {s}')
    fig, ax = plt.subplots()
    index = np.arange(12)
    bar_width = 0.2
    opacity = 0.8

    cv = plt.bar(index, e_cross_val, bar_width, alpha=opacity, color='g', label='error cross validation')
    test = plt.bar(index + bar_width, e_test, bar_width, alpha=opacity, color='b', label='error test')
    diff = plt.bar(index + 2*bar_width, abs_diff, bar_width, alpha=opacity, color='r', label='difference')

    plt.xticks(index + bar_width, lambda_sigma_tuples)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
