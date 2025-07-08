import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from libs.checkpoints import ImportedCheckpoint


class CLI(Tap):
    input_checkpoint: Path  # Path to the input checkpoint file <checkpoint.npz>
    h0: int = 1  # Step height used during visualization

    def configure(self):
        self.add_argument(
            "input_checkpoint", type=Path, help="Path to the input checkpoint file."
        )


def create_3d_surface_from_level_lines(u_array, x_coords, h0):
    """
    Create a 3D surface from level lines.

    Parameters:
    -----------
    u_array : numpy.ndarray
        2D array where u_array[i, j] = u_i(x_j) represents the i-th level line
        at position x_j
    x_coords : numpy.ndarray
        1D array of x coordinates
    h0 : float
        Height difference between consecutive level lines
    Returns:
    --------
    X, Y, Z : numpy.ndarray
        Meshgrid arrays for 3D plotting
    """

    n_levels, n_x = u_array.shape

    # Create staircase surface
    # For each level line, create a horizontal plane at height i*h0

    # Create extended coordinates for the staircase
    x_extended = []
    y_extended = []
    z_extended = []

    for i in range(n_levels):
        # Height for this level
        height = i * h0

        # Get the level line values
        level_values = u_array[i, :]

        # For each segment of the level line
        for j in range(len(x_coords)):
            x_extended.append(x_coords[j])
            y_extended.append(level_values[j])
            z_extended.append(height)

    # Create a regular grid for surface plotting
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = u_array.min(), u_array.max()

    # Create grid
    x_grid = np.linspace(x_min, x_max, 300)
    y_grid = np.linspace(y_min, y_max, 300)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Initialize Z with zeros
    Z = np.zeros_like(X)

    # For each point in the grid, determine which level it belongs to
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            x_val = x_grid[i]
            y_val = y_grid[j]

            # Find the appropriate level by checking which level line this point is closest to
            min_dist = float("inf")
            best_level = 0

            for level in range(n_levels):
                # Interpolate the level line at x_val
                level_y = np.interp(x_val, x_coords, u_array[level, :])
                dist = abs(y_val - level_y)

                if dist < min_dist:
                    min_dist = dist
                    best_level = level

            # Set the height based on the closest level
            Z[j, i] = best_level * h0

    return X, Y, Z


def plot_3d_surface(
    X,
    Y,
    Z,
    level_lines_2d=None,
    x_coords=None,
    title="3D Surface from Level Lines",
    h0=1,
):
    """
    Plot the 3D surface with optional 2D level lines overlay.
    """
    fig = plt.figure(figsize=(15, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.view_init(elev=30, azim=-160)
    surf = ax1.plot_surface(
        X,
        Y,
        Z,
        color="black",
        alpha=0.5,
        linewidth=0,
        antialiased=False,
        rcount=50,
        ccount=50,
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z (Height)")
    ax1.set_title(title)

    # Add contour lines at exact step heights (isolines) for better edge visibility
    step_heights = np.arange(0, Z.max() + h0, h0)
    for height in step_heights:
        ax1.contour(X, Y, Z, levels=[height], colors="red", linewidths=2.0, alpha=0.9)

    # Set background color for better contrast
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(True, alpha=0.3)

    # 2D level lines plot (if provided)
    if level_lines_2d is not None and x_coords is not None:
        ax2 = fig.add_subplot(122)
        for i in range(level_lines_2d.shape[0]):
            ax2.plot(x_coords, level_lines_2d[i, :])
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title("Original Level Lines")
        ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    cli = CLI().parse_args()
    ch = ImportedCheckpoint.load_from_file(cli.input_checkpoint)

    X, Y, Z = create_3d_surface_from_level_lines(ch.U, ch.X, cli.h0)

    plot_3d_surface(
        X, Y, Z, ch.U, ch.X, "3D Staircase Surface from Level Lines", h0=cli.h0
    )


if __name__ == "__main__":
    main()
