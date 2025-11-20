import logging
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from libs.checkpoints import Checkpoint
from libs.data3d import create_3d_surface_from_level_lines, subtract_average_surface_slope

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
    afm: bool = False  # Produce AFM-like image instead of normal line plot

    def configure(self):
        self.add_argument(
            "input",
            type=Path,
            help="Path to an input checkpoint file or a directory containing checkpoint files.",
        )


def plot_dir(input_dir: Path, afm: bool = False):
    checkpoint_files = list(input_dir.glob("*.npz"))
    output_dir = input_dir / "plots"
    os.makedirs(output_dir, exist_ok=True)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in directory {input_dir}.")

    partial_plot_single = partial(plot_single, output_dir=output_dir, afm=afm)

    with Pool(processes=CLI().parallelism) as pool:
        pool.map(partial_plot_single, checkpoint_files)


def plot_single(input_path: Path, output_dir: Path, afm: bool = False):
    logger.debug(f"Plotting checkpoint file: {input_path}")
    checkpoint = Checkpoint.load_from_file(input_path)
    step = checkpoint.iter
    U = checkpoint.U
    K = checkpoint.K
    x = checkpoint.X

    if afm:
        plot_afm(U, x, step, input_path, output_dir)
    else:
        plot_normal(U, K, x, step, input_path, output_dir)


def plot_normal(U, K, x, step, input_path, output_dir):
    """Plot normal line plot visualization."""
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


def create_afm_hot_colormap():
    """Create the AFM hot color scheme from Mathematica.

    afmHot[z] = RGBColor[2z, 2z - 0.5, 2z - 1]
    Maps from black (z=0) -> red -> yellow -> white (z=1)
    """
    from matplotlib.colors import LinearSegmentedColormap

    # Create color map by evaluating the AFM hot function at different z values
    n_colors = 256
    z_vals = np.linspace(0, 1, n_colors)
    colors = []

    for z in z_vals:
        r = np.clip(2 * z, 0, 1)
        g = np.clip(2 * z - 0.5, 0, 1)
        b = np.clip(2 * z - 1, 0, 1)
        colors.append([r, g, b])

    return LinearSegmentedColormap.from_list("afm_hot", colors)


def plot_afm(U, x, step, input_path, output_dir):
    """Plot AFM-like image of the surface."""
    # Create 3D surface from level lines
    # Use spacing between levels as h0 (vertical spacing)
    h0 = 1.0  # Default height per level
    X, Y, Z = create_3d_surface_from_level_lines(U, x, h0, samples=300)

    # Detrend the surface (subtract average slope)
    Z_detrended = subtract_average_surface_slope(X, Y, Z)

    # Normalize Z to [0, 1] for color mapping
    Z_min = Z_detrended.min()
    Z_max = Z_detrended.max()
    Z_normalized = (Z_detrended - Z_min) / (Z_max - Z_min + 1e-10)

    # Create AFM-like image using height map with 90 degree rotation (swap x, y)
    plt.figure(figsize=(10, 8))

    # Transpose the Z matrix and swap extent coordinates to achieve 90 degree rotation
    im = plt.imshow(
        Z_normalized.T,  # Transpose for 90 degree rotation
        extent=[Y.min(), Y.max(), X.min(), X.max()],  # Swapped extent (rotated)
        origin="lower",
        cmap=create_afm_hot_colormap(),
        interpolation="bilinear",
        aspect="auto",
    )

    cbar = plt.colorbar(im, label="Height (normalized)")
    plt.xlabel("y (level)")
    plt.ylabel("x")
    plt.title(f"AFM Surface - Time step {step}")

    image_name = f"{input_path.stem}_afm.png"
    plt.savefig(output_dir / image_name, dpi=200, bbox_inches="tight")
    logger.debug(f"Saved AFM plot to {output_dir/image_name}")
    plt.close()


def main():
    cli = CLI().parse_args()

    if cli.input.is_dir():
        logger.info(f"Plotting all checkpoints in directory: {cli.input}")
        plot_dir(cli.input, afm=cli.afm)
    elif cli.input.is_file():
        logger.info(f"Plotting single checkpoint file: {cli.input}")
        plot_single(cli.input, cli.input.parent, afm=cli.afm)
    else:
        raise ValueError(f"Input {cli.input} is neither a file nor a directory.")
    logger.info(f"Finished plotting all checkpoints in directory: {cli.input}")


if __name__ == "__main__":
    main()
