from numba import cuda, njit
import numpy as np
import math
import pylab


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

pylab.imshow(z)
pylab.show()

# CPU

z0 = z.copy()
z1 = np.zeros_like(z0)


@njit
def smooth_jit(x0, x1):
    for i in range(1, x0.shape[0] - 1):
        for j in range(1, x0.shape[1] - 1):
            x1[i, j] = 0.25 * (x0[i, j - 1] + x0[i, j + 1] +
                               x0[i - 1, j] + x0[i + 1, j])


for i in range(2000):
    if (i % 2) == 0:
        smooth_jit(z0, z1)
    else:
        smooth_jit(z1, z0)

z_cpu = z0

pylab.imshow(z_cpu)
pylab.show()


# GPU

@cuda.jit
def smooth_cuda(x0, x1):
    i, j = cuda.grid(2)

    i_in_bounds = (i > 0) and (i < (x0.shape[0] - 1))
    j_in_bounds = (j > 0) and (j < (x1.shape[1] - 1))

    if i_in_bounds and j_in_bounds:
        x1[i, j] = 0.25 * (x0[i, j - 1] + x0[i, j + 1] +
                           x0[i - 1, j] + x0[i + 1, j])


z0 = cuda.to_device(z)
z1 = cuda.device_array_like(np.zeros_like(z))

for i in range(2000):
    if (i % 2) == 0:
        smooth_cuda[(16, 16), (16, 16)](z0, z1)
    else:
        smooth_cuda[(16, 16), (16, 16)](z1, z0)


z_cuda = z0.copy_to_host()

pylab.imshow(z_cuda)
pylab.show()

np.testing.assert_allclose(z_cpu, z_cuda)
