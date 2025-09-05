from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn import linear_model
from tap import Tap

from libs.checkpoints import Checkpoint


@dataclass
class BunchSlice:
    start_index: int
    end_index: int
    positions: np.ndarray
    l_min: float  # minimum distance in the bunch
    l_max: float  # maximum distance in the bunch
    l_avg: float  # average distance in the bunch
    bunch_size: int  # number of steps in the bunch


class CLI(Tap):
    input_checkpoint: Path  # Path to the input checkpoint file <checkpoint.npz>
    output_dir: Path = Path(
        "analysis-results"
    )  # Directory to save the analysis results

    def configure(self):
        self.add_argument("input_checkpoint")


def average_step_positions(ch: Checkpoint) -> np.ndarray:
    return np.mean(ch.U, axis=1)


def indentify_bunches(positions: np.ndarray, bunch_def: float) -> list[BunchSlice]:
    # bunch is defined as a group of steps that are closer than bunch_def
    # the bunch ends when the distance between two consecutive steps is larger than bunch_def
    distances = np.diff(positions)
    bunch_indices = np.where(distances > bunch_def)[0]
    res = np.where(distances > bunch_def)

    if len(bunch_indices) == 0:
        # all steps are in one bunch
        return [
            BunchSlice(
                0,
                len(positions) - 1,
                positions,
                np.min(distances),
                np.max(distances),
                float(np.mean(distances)),
                len(positions),
            )
        ]

    bunches = []
    for i in range(len(bunch_indices) + 1):
        if i == 0:
            start_index = 0
            end_index = bunch_indices[i]
        elif i == len(bunch_indices):
            start_index = bunch_indices[i - 1] + 1
            end_index = len(positions) - 1
        else:
            start_index = bunch_indices[i - 1] + 1
            end_index = bunch_indices[i]

        bunch_positions = positions[start_index : end_index + 1]
        if len(bunch_positions) > 1:
            bunch_distances = np.diff(bunch_positions)
            l_min = np.min(bunch_distances)
            l_max = np.max(bunch_distances)
            l_avg = float(np.mean(bunch_distances))
        else:
            l_min = 0.0
            l_avg = 0.0
            l_max = 0.0

        bunch_size = len(bunch_positions)
        bunches.append(
            BunchSlice(
                start_index, end_index, bunch_positions, l_min, l_max, l_avg, bunch_size
            )
        )

    return bunches


def plot_lmax_vs_lmin(bunches: list[BunchSlice]):
    l_maxes = np.array([bunch.l_max for bunch in bunches])
    l_mins = np.array([bunch.l_min for bunch in bunches])

    ransac_estimator = linear_model.RANSACRegressor(max_trials=1000)
    ransac_estimator.fit(
        np.array(l_maxes).reshape(-1, 1), np.array(l_mins).reshape(-1, 1)
    )
    fit = (
        ransac_estimator.estimator_.coef_[0][0],
        ransac_estimator.estimator_.intercept_[0],
    )
    l_min_calc = fit[0] * l_maxes + fit[1]

    plt.scatter(l_maxes, l_mins)
    plt.plot(
        l_maxes,
        l_min_calc,
        color="red",
        label=f"RANSAC Fit: y={fit[0]:.2f}x + {fit[1]:.2f}",
    )
    plt.title("Max vs Min distance in bunches")
    plt.xlabel("$l_{max}$")
    plt.ylabel("$l_{min}$")
    plt.legend()
    plt.show()


def plot_bunch_sizes_histogram(bunches: list[BunchSlice]):
    sizes = [bunch.bunch_size for bunch in bunches]
    plt.hist(sizes, bins=range(1, max(sizes) + 2), align="left", rwidth=0.8)
    plt.title("Bunch Size Distribution")
    plt.xlabel("Bunch Size (Number of Steps)")
    plt.ylabel("Frequency")
    plt.xticks(range(1, max(sizes) + 1))
    plt.show()


def main():
    args = CLI().parse_args()
    checkpoint = Checkpoint.load_from_file(args.input_checkpoint)

    # plot average  step positions
    avg_positions = average_step_positions(checkpoint)
    bunches = indentify_bunches(avg_positions, bunch_def=checkpoint.c)

    plot_lmax_vs_lmin(bunches)
    plot_bunch_sizes_histogram(bunches)


if __name__ == "__main__":
    main()
