from numba import cuda, njit
import numpy as np
import math


MU = 0.0
THETA = 1.0


@njit
def gauss2d(x, y):
    grid = np.empty_like(x)

    a = 1.0 / (THETA * np.sqrt(2 * math.pi))

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i, j] = a * np.exp(-(x[i, j]**2 / (2 * THETA) + y[i, j]**2
                                      / (2 * THETA)))

    return grid


X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(X, Y)

z = gauss2d(x, y)
