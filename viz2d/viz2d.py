import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from libs.checkpoints import Checkpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class CLI(Tap):
    input: Path  # Path to an input checkpoint file or a directory containing checkpoint files
    parallelism: int = 4  # Threads to use  for plotting

    def configure(self):
        self.add_argument(
            "input",
            type=Path,
            help="Path to an input checkpoint file or a directory containing checkpoint files.",
        )


def plot_dir(input_dir: Path):
    checkpoint_files = list(input_dir.glob("*.npz"))
    output_dir = input_dir / "plots"
    os.makedirs(output_dir, exist_ok=True)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in directory {input_dir}.")

    partial_plot_single = partial(plot_single, output_dir=output_dir)

    with Pool(processes=CLI().parallelism) as pool:
        pool.map(partial_plot_single, checkpoint_files)


def plot_single(input_path: Path, output_dir: Path):
    logger.debug(f"Plotting checkpoint file: {input_path}")
    checkpoint = Checkpoint.load_from_file(input_path)
    step = checkpoint.iter
    U = checkpoint.U
    K = checkpoint.K
    x = checkpoint.X

    U_norm = U - np.min(U)
    U_max = 1.1 * np.max(U_norm)

    plt.figure()
    for i in range(K):
        plt.plot(x, U_norm[i, :], label=f"$u_{{" f"{i + 1}" f"}}$")

    plt.xlabel("x")
    plt.ylabel("$u_i - min$")
    plt.ylim(0, U_max)
    plt.title(f"Time step {step}")

    image_name = f"{input_path.stem}.png"
    plt.savefig(output_dir / image_name, dpi=200)
    logger.debug(f"Saved plot to {output_dir/image_name}")
    plt.close()


def main():
    cli = CLI().parse_args()

    if cli.input.is_dir():
        logger.info(f"Plotting all checkpoints in directory: {cli.input}")
        plot_dir(cli.input)
    elif cli.input.is_file():
        logger.info(f"Plotting single checkpoint file: {cli.input}")
        plot_single(cli.input, cli.input.parent)
    else:
        raise ValueError(f"Input {cli.input} is neither a file nor a directory.")
    logger.info(f"Finished plotting all checkpoints in directory: {cli.input}")


if __name__ == "__main__":
    main()
