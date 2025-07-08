from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from libs.checkpoints import ImportedCheckpoint


class CLI(Tap):
    input_checkpoint: Path  # Path to the input checkpoint file <checkpoint.npz>
    output_dir: Path = Path(
        "analysis-results"
    )  # Directory to save the analysis results

    def configure(self):
        self.add_argument("input_checkpoint")


def find_column_wise_distances(U: np.ndarray) -> np.ndarray:
    U_norm = U - np.min(
        U, axis=0
    )  # Normalize U by subtracting the minimum in each column
    U_rolled = np.roll(U_norm, shift=1, axis=0)
    diff = np.abs(U_norm - U_rolled)
    return diff


def count_bunches_per_column(distances_thr: np.ndarray):
    """
    Count the number of bunches in each column of the thresholded array.
    A bunch is defined as a contiguous sequence of 1s in a column.
    """
    bunch_counts = []
    for col in distances_thr.T:  # Iterate over columns
        count = 0
        in_bunch = False
        for value in col:
            if value == 1 and not in_bunch:
                count += 1
                in_bunch = True
            elif value == 0:
                in_bunch = False
        bunch_counts.append(count + 1)
    return np.array(bunch_counts)


def largest_bunch_per_column(distances_thr: np.ndarray):
    """
    Find the size of the largest bunch in each column of the thresholded array.
    A bunch is defined as a contiguous sequence of 1s in a column.
    """
    largest_bunches = []
    for col in distances_thr.T:  # Iterate over columns
        max_size = 0
        current_size = 0
        for value in col:
            if value == 1:
                current_size += 1
            else:
                max_size = max(max_size, current_size)
                current_size = 0
        max_size = max(max_size, current_size)  # Check at the end of the column
        largest_bunches.append(max_size + 1)
    return np.array(largest_bunches)


def main():
    cli = CLI().parse_args()
    ch = ImportedCheckpoint.load_from_file(cli.input_checkpoint)

    distances = find_column_wise_distances(ch.U)
    distances_thr = np.where(distances < ch.c, 1, 0)

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.2], height_ratios=[1, 1])

    # Plot 1: Column-wise distances
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(distances[1:, 0], label="Column-wise distances")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Distance")
    ax1.set_title("Column-wise distances")

    # Plot 2: Column-wise distances (thresholded)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(distances_thr[1:, 0], label="Column-wise distances")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Distance (thresholded)")
    ax2.set_title("Column-wise distances (thresholded)")

    # Plot 3: Bunch counts per column
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(count_bunches_per_column(distances_thr), label="Bunch counts per column")
    ax3.set_xlabel("Column index")
    ax3.set_ylabel("Number of bunches")
    ax3.set_title("Bunch counts per column")

    # Plot 4: Largest bunch per column
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(largest_bunch_per_column(distances_thr), label="Largest bunch per column")
    ax4.set_xlabel("Column index")
    ax4.set_ylabel("Largest bunch size")
    ax4.set_title("Largest bunch size per column")

    # Plot 5: Column-wise distances (2D)
    ax5 = fig.add_subplot(gs[1, 1:])
    im = ax5.imshow(distances[1:, :], aspect="auto", cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04, label="Distance")
    ax5.set_xlabel("Index")
    ax5.set_ylabel("Column")
    ax5.set_title("Column-wise distances (2D)")

    plt.tight_layout()

    # save high quality figures
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    input_file_name = cli.input_checkpoint.stem
    output_file = cli.output_dir / f"analysis_results_{input_file_name}.png"
    plt.savefig(output_file, dpi=300)


if __name__ == "__main__":
    main()
