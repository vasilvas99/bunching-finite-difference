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
    show: bool = False # Whether show or save
    publication: bool = False # Whether to use publication style for the plots

    def configure(self):
        self.add_argument(
            "input",
            type=Path,
            help="Path to an input checkpoint file or a directory containing checkpoint files.",
        )


def plot_dir(input_dir: Path, show = False, publication = False):
    checkpoint_files = list(input_dir.glob("*.npz"))
    output_dir = input_dir / "plots"
    os.makedirs(output_dir, exist_ok=True)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in directory {input_dir}.")

    partial_plot_single = partial(plot_single, output_dir=output_dir, show=show, publication=publication)

    with Pool(processes=CLI().parallelism) as pool:
        pool.map(partial_plot_single, checkpoint_files)


def plot_single(input_path: Path, output_dir: Path, show = False, publication = False):
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
        if publication:
            plt.plot(x, U_norm[i, :], label=f"$u_{{{i + 1}}}$", color="black")
        else:
            plt.plot(x, U_norm[i, :], label=f"$u_{{{i + 1}}}$")

    plt.xlabel("x")

    plt.ylim(0, U_max)
    if not publication:
        plt.ylabel("$u_i - min$")
    else:
        plt.ylabel("$u_n(x)$")

    if not publication:
        plt.title(f"Time step {step}")

    if publication:
        plt.tight_layout()

    image_name = f"{input_path.stem}.png"
    if not show:
        plt.savefig(output_dir / image_name, dpi=200)
        logger.debug(f"Saved plot to {output_dir/image_name}")
        plt.close()
    else:
        plt.show()


def main():
    cli = CLI().parse_args()

    if cli.input.is_dir():
        logger.info(f"Plotting all checkpoints in directory: {cli.input}")
        plot_dir(cli.input, show=cli.show, publication=cli.publication)
    elif cli.input.is_file():
        logger.info(f"Plotting single checkpoint file: {cli.input}")
        plot_single(cli.input, cli.input.parent, show=cli.show, publication=cli.publication)
    else:
        raise ValueError(f"Input {cli.input} is neither a file nor a directory.")
    logger.info(f"Finished plotting all checkpoints in directory: {cli.input}")


if __name__ == "__main__":
    main()
