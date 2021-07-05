from numba import njit
import numpy as np
import math


THETA = 1.0


@njit
def gauss2d(x, y):
    grid = np.empty_like(x)

    a = 1.0 / np.sqrt(2 * math.pi)

    for i in range(grid.shape[0]):
        pass

    return grid


X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(X, Y)

z = gauss2d(x, y)
