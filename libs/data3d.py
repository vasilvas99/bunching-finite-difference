import numpy as np


def create_3d_surface_from_level_lines(u_array, x_coords, h0, samples=300):
    n_levels, n_x = u_array.shape
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = u_array.min(), u_array.max()
    x_grid = np.linspace(x_min, x_max, samples)
    y_grid = np.linspace(y_min, y_max, samples)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)

    for i in range(len(x_grid)):
        x_val = x_grid[i]
        # Interpolate all level lines at x_val
        level_ys = [
            np.interp(x_val, x_coords, u_array[level, :]) for level in range(n_levels)
        ]
        for j in range(len(y_grid)):
            y_val = y_grid[j]
            # Find the highest level below y_val
            level = 0
            for k in range(n_levels):
                if level_ys[k] <= y_val:
                    level = k
            Z[j, i] = level * h0
    return X, Y, Z


def subtract_average_surface_slope(X, Y, Z):
    # fit plane Z = aX + bY + c
    A = np.c_[X.ravel(), Y.ravel(), np.ones(X.size)]
    C, _, _, _ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    a, b, c = C
    Z_detrended = Z - (a * X + b * Y + c)
    return Z_detrended
