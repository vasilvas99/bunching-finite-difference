from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tap import Tap

from libs.checkpoints import Checkpoint


class CLI(Tap):
    checkpoints_dir: Path  # Directory containing the checkpoints
    sigma: float = 6.0  # Standard deviation for Gaussian smoothing
    parallelism: int = 4  # Threads to use for plotting

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
    print(f"Total curvature for checkpoint at iteration {ch.iter} is {total_curvature}")
    return total_curvature / ch.K


def map_fn(cp_path, sigma) -> tuple[int, float]:
    loaded_ch = Checkpoint.load_from_file(cp_path)
    avg_curv = calculate_average_curvate(loaded_ch, sigma=sigma)
    return loaded_ch.iter, avg_curv


def main():
    cli = CLI().parse_args()
    checkpoints_dir = cli.checkpoints_dir
    datapoints = []

    checkpoints = checkpoints_dir.glob("checkpoint_*.npz")
    try:
        first_checkpoint = Checkpoint.load_from_file(next(checkpoints))
    except StopIteration:
        print(f"No checkpoint files found in directory {checkpoints_dir}.")
        exit(1)

    print(
        f"Loaded checkpoints from {checkpoints_dir}. Parameters K={first_checkpoint.K}, M={first_checkpoint.M},"
        f" L={first_checkpoint.L}, T={first_checkpoint.T},"
        f" D={first_checkpoint.D}, c={first_checkpoint.c}, dt={first_checkpoint.dt}"
    )

    first_curvature = calculate_average_curvate(first_checkpoint, sigma=cli.sigma)
    datapoints.append((first_checkpoint.iter, first_curvature))

    partial_fn = partial(map_fn, sigma=cli.sigma)

    with Pool(cli.parallelism) as pool:
        results = pool.map(partial_fn, checkpoints)

    datapoints.extend(results)

    sorted_datapoints = sorted(datapoints, key=lambda x: x[0])

    iterations, curvatures = zip(*sorted_datapoints)
    iterations = np.array(iterations, dtype=float)
    iterations *= first_checkpoint.dt
    curvatures = np.array(curvatures)
    curvatures /= curvatures[0]
    plt.plot(iterations, curvatures)
    plt.title(f"Relative surface curvature over time ($\\gamma$={first_checkpoint.D})")
    plt.ylabel("Relative surface curvature")
    plt.xlabel("time (s)")
    plt.show()


if __name__ == "__main__":
    main()
