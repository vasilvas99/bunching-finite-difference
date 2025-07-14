from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tap import Tap

from libs.checkpoints import Checkpoint


class CLI(Tap):
    checkpoints_dir: Path  # Directory containing the checkpoints
    sigma: float = 6.0  # Standard deviation for Gaussian smoothing

    def configure(self):
        self.add_argument(
            "checkpoints_dir", type=Path, help="Directory containing the checkpoints."
        )


def calculate_average_curvate(ch: Checkpoint, sigma=6.0) -> float:
    total_curvature = 0
    for i in range(ch.K):
        level_line = gaussian_filter1d(ch.U[i, :], sigma=sigma)  # Smooth the level line
        x = ch.X
        # calculate the curvature using finite diff and periodic boundary conditions
        dx = np.roll(x, -1) - x  # Periodic boundary conditions
        dy = np.roll(level_line, -1) - level_line  # Periodic boundary conditions
        d2x = np.roll(dx, -1) - dx  # Second derivative
        d2y = np.roll(dy, -1) - dy  # Second derivative
        kappa = np.abs(d2y * dx - dy * d2x) / (dx**2 + dy**2) ** (3 / 2)
        total_curvature += np.mean(kappa)  # Average curvature for this level line

    return total_curvature / ch.K


def main():
    cli = CLI().parse_args()
    checkpoints_dir = cli.checkpoints_dir
    datapoints = []
    dt = None
    gamma = None
    printed_ch = False
    for checkpoint_path in checkpoints_dir.glob("checkpoint_*.npz"):
        ch = Checkpoint.load_from_file(checkpoint_path)
        dt = ch.dt if dt is None else dt
        if not printed_ch:
            print(f"Loaded checkpoint from {checkpoint_path}")
            print(
                f"Parameters: K={ch.K}, M={ch.M}, L={ch.L}, T={ch.T}, D={ch.D}, c={ch.c}, dt={ch.dt}"
            )
            printed_ch = True
        gamma = ch.D if gamma is None else gamma
        avg_curvature = calculate_average_curvate(ch, sigma=cli.sigma)
        datapoints.append((ch.iter, avg_curvature))
        print(f"Iteration {ch.iter}: Average curvature = {avg_curvature:.4f}")
    sorted_datapoints = sorted(datapoints, key=lambda x: x[0])

    # plot the x_y pairs in sorted_datapoints
    iterations, curvatures = zip(*sorted_datapoints)
    iterations = np.array(iterations, dtype=float)
    iterations *= dt
    curvatures = np.array(curvatures)
    curvatures /= curvatures[0]
    plt.plot(iterations, curvatures)
    plt.title(f"Relative surface curvature over time ($\\gamma$={gamma})")
    plt.ylabel("Relative surface curvature")
    plt.xlabel("time (s)")
    plt.show()


if __name__ == "__main__":
    main()
