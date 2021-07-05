from numba import cuda, njit
import numpy as np
import math
import pylab
from time import perf_counter


ITERATIONS = 2
POINTS = 550

@njit
def gauss2d(x, y):
    grid = np.empty_like(x).astype(np.float32)

    a = 1.0 / np.sqrt(2 * math.pi)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i, j] = a * np.exp(-(x[i, j]**2 / 2 + y[i, j]**2
                                      / 2))

    return grid


X = np.linspace(-5, 5, POINTS)
Y = np.linspace(-5, 5, POINTS)
x, y = np.meshgrid(X, Y)

z = gauss2d(x, y)

pylab.imshow(z)
pylab.show()

# Python

z0 = z.copy()
z1 = np.zeros_like(z0)


def smooth(x0, x1):
    for i in range(1, x0.shape[0] - 1):
        for j in range(1, x0.shape[1] - 1):
            x1[i, j] = 0.25 * (x0[i, j - 1] + x0[i, j + 1] +
                               x0[i - 1, j] + x0[i + 1, j])


start = perf_counter()

#for i in range(2000):
#    if (i % 2) == 0:
#        smooth(z0, z1)
#    else:
#        smooth(z1, z0)

end = perf_counter()

z_python = z0
time_python = end - start

pylab.imshow(z_python)
pylab.show()


# CPU JIT

z0 = z.copy()
z1 = np.zeros_like(z0)


@njit
def smooth_jit(x0, x1):
    for i in range(1, x0.shape[0] - 1):
        for j in range(1, x0.shape[1] - 1):
            x1[i, j] = 0.25 * (x0[i, j - 1] + x0[i, j + 1] +
                               x0[i - 1, j] + x0[i + 1, j])


# Warm up JIT
smooth_jit(z0, z1)

start = perf_counter()

for i in range(ITERATIONS):
    if (i % 2) == 0:
        smooth_jit(z0, z1)
    else:
        smooth_jit(z1, z0)

end = perf_counter()

z_cpu = z0
time_cpu = end - start

pylab.imshow(z_cpu)
pylab.show()


# GPU

@cuda.jit
def smooth_cuda(x0, x1):
    i, j = cuda.grid(2)

    i_in_bounds = (i > 0) and (i < (x0.shape[0] - 1))
    j_in_bounds = (j > 0) and (j < (x0.shape[1] - 1))

    if i_in_bounds and j_in_bounds:
        x1[i, j] = 0.25 * (x0[i, j - 1] + x0[i, j + 1] +
                           x0[i - 1, j] + x0[i + 1, j])


# Copy to device and warm up JIT

z0 = cuda.to_device(z)
z1 = cuda.device_array_like(np.zeros_like(z))
#smooth_cuda[(16, 16), (16, 16)](z0, z1)

start = perf_counter()

for i in range(ITERATIONS):
    if (i % 2) == 0:
        smooth_cuda[(16, 16), (16, 16)](z0, z1)
    else:
        smooth_cuda[(16, 16), (16, 16)](z1, z0)

# Make sure the GPU is finished before we stop timing
cuda.synchronize()
end = perf_counter()

z_cuda = z0.copy_to_host()
time_cuda = end - start

pylab.imshow(z_cuda)
pylab.show()

#np.testing.assert_allclose(z_python, z_cpu)
#np.testing.assert_allclose(z_python, z_cuda)
#np.testing.assert_allclose(z_cpu, z_cuda)

res_diff = np.abs(z_cpu - z_cuda)
breakpoint()
print(np.argmax(res_diff))
#print(res_diff[255, 255])

pylab.imshow(res_diff)
pylab.show()

#print(f"Python time: {time_python}")
print(f"CPU time: {time_cpu}")
print(f"CUDA time: {time_cuda}")
