from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from libs.checkpoints import Checkpoint


class CLI(Tap):
    input_checkpoint: Path  # Path to the input checkpoint file <checkpoint.npz>
    h0: int = 1  # Step height used during visualization
    save: bool = False  # Save the plot to a file
    save_dir: Path = Path("./3d")  # Directory to save the plot

    def configure(self):
        self.add_argument(
            "input_checkpoint", type=Path, help="Path to the input checkpoint file."
        )


def create_3d_surface_from_level_lines(u_array, x_coords, h0):
    n_levels, n_x = u_array.shape
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = u_array.min(), u_array.max()
    x_grid = np.linspace(x_min, x_max, 300)
    y_grid = np.linspace(y_min, y_max, 300)
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
    ax1.plot_surface(
        X,
        Y,
        Z,
        color="black",
        alpha=0.5,
        linewidth=0,
        rcount=500,
        ccount=500,
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z (Height)")
    ax1.set_title(title)

    # Add contour lines at exact step heights (isolines) for better edge visibility
    step_heights = np.arange(0, Z.max() + h0, h0)
    for height in step_heights:
        ax1.contour(X, Y, Z, levels=[height], colors="red", linewidths=0.7, alpha=0.3)

    # Set background color for better contrast
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.grid(False)

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


def main():
    cli = CLI().parse_args()
    ch = Checkpoint.load_from_file(cli.input_checkpoint)

    X, Y, Z = create_3d_surface_from_level_lines(ch.U - np.min(ch.U), ch.X, cli.h0)

    plot_3d_surface(
        X, Y, Z, ch.U, ch.X, "3D Staircase Surface from Level Lines", h0=cli.h0
    )
    if cli.save:
        plt.savefig(cli.save_dir / f"{cli.input_checkpoint.stem}.png", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
