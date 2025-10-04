import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate
from tap import Tap

from libs.checkpoints import Checkpoint
from libs.data3d import create_3d_surface_from_level_lines

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class CLI(Tap):
    input_checkpoint: Path
    h0: int = 1
    samples: int = 300

    def configure(self):
        self.add_argument(
            "input_checkpoint", type=Path, help="Path to the input checkpoint file."
        )


def estimate_bin_params(N: int):
    # target minimum per bin grows ~ sqrt(N)
    target_min_el_per_bin = max(10, int(np.sqrt(N)))
    n_bins = N // target_min_el_per_bin if target_min_el_per_bin > 0 else 1
    n_bins = int(np.clip(n_bins, 5, 200))

    # minimum count per bin at least 5, but also at least 40% of average bin count
    min_count = max(5, int(0.4 * N / max(n_bins, 1)))
    return n_bins, min_count


def quantile_bin(dist: np.ndarray, vals: np.ndarray, n_bins: int, min_count: int):
    mask = np.isfinite(dist) & np.isfinite(vals)
    d = dist[mask]
    v = vals[mask]
    if d.size == 0:
        return np.array([]), np.array([]), np.array([])

    edges = np.quantile(d, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)

    if edges.size < 2:
        return np.array([]), np.array([]), np.array([])

    # Assign bins
    idx = np.digitize(d, edges, right=True) - 1
    nb = edges.size - 1
    centers = []
    means = []
    counts = []

    for k in range(nb):
        sel = idx == k  # get all elements in bin k
        c = np.sum(sel)  # count elements in bin k

        if c < min_count:  # skip bins with too few elements
            continue

        centers.append(0.5 * (edges[k] + edges[k + 1]))
        means.append(v[sel].mean())
        counts.append(c)

    if not centers:
        return np.array([]), np.array([]), np.array([])

    return np.asarray(centers), np.asarray(means), np.asarray(counts)


def compute_variogram(Z: np.ndarray, x_step: float = 1.0, y_step: float = 1.0):
    """
    Let = i a multindex  i = (i_y, i_x) and h = (h_y, h_x)
    Variogram is
    V(h) = 1/deg(h) * sum_{i,j: j-i=h} (z_i - z_j)^2
     = 1/deg(h) * ( sum_i z_i^2 + sum_j z_j^2 - 2 sum_{i,j: j-i=h} z_i * z_j )
     = 1/deg(h) * ( A(h) + A(-h) - 2 B(h) )
     Let  Z_2 = Z_ij ^ 2, i.e. elementwise square, and I identity matrix with shape of Z_2
     A(h) = sum_i z_i^2 = Z_2 ⊗ I (I affects Z_2)
     A(-h) = sum_i z_i+h  = I ⊗ Z_2 (z_2 affects I)
     B(h) = sum_i z_i * z_{i+h} = Z ⊗ Z (auto-correlation)
     deg(h) = number of overlapped ordered pairs for lag h, used for normalization by the number of pairs in lag h

     Then total_sum T(h) = A(h) + A(-h) - 2 B(h)
     V(H) = T(h) / deg(h) # average per lag (averaged over all pairs with lag h)
    """

    ny, nx = Z.shape
    ones = np.ones_like(Z, dtype=float)
    Z2 = Z * Z

    B = correlate(Z, Z, mode="full", method="fft")  # B(h) = sum_i z_i * z_{i+h}
    logger.debug(f"B shape: {B.shape}, expected: {(2*ny-1, 2*nx-1)}")
    A = correlate(Z2, ones, mode="full", method="fft")  # A(h) = sum_i z_i^2
    logger.debug(f"A shape: {A.shape}, expected: {(2*ny-1, 2*nx-1)}")
    A_neg = correlate(ones, Z2, mode="full", method="fft")  # A(-h) = sum_i z_{i+h}^2
    logger.debug(f"A_neg shape: {A_neg.shape}, expected: {(2*ny-1, 2*nx-1)}")

    # deg(h) = number of overlapped ordered pairs for lag h
    deg = correlate(ones, ones, mode="full", method="fft")
    logger.debug(f"deg shape: {deg.shape}, expected: {(2*ny-1, 2*nx-1)}")

    # Total sum T(h) = A(h) + A(-h) - 2 B(h)
    T = A + A_neg - 2.0 * B

    with np.errstate(divide="ignore", invalid="ignore"):
        variogram = np.where(deg > 0, T / deg, np.nan)

    mask = deg > 0

    # distances for each discrete lag index
    ky = np.arange(2 * ny - 1) - (ny - 1)
    kx = np.arange(2 * nx - 1) - (nx - 1)
    dy_idx, dx_idx = np.meshgrid(ky, kx, indexing="ij")
    dist_mat = np.sqrt((dx_idx * x_step) ** 2 + (dy_idx * y_step) ** 2)

    # return only valid entries
    return dist_mat[mask].ravel(), variogram[mask].ravel()


def compute_1d_variogram(Z: np.ndarray, axis: int = 0, axis_step: float = 1.0):
    variogram = []
    distances = []
    n_points = Z.shape[axis]
    for lag in range(1, n_points):
        if axis == 0:
            diffs = Z[lag:, :] - Z[:-lag, :]
        else:
            diffs = Z[:, lag:] - Z[:, :-lag]
        sq_diffs = diffs**2
        variogram.append(np.nanmean(sq_diffs))
        distances.append(lag * axis_step)
    return np.array(distances), np.array(variogram)


def main():
    cli = CLI().parse_args()
    if not cli.input_checkpoint.exists() and cli.input_checkpoint.name.endswith(".npz"):
        raise FileNotFoundError(
            f"Checkpoint file {cli.input_checkpoint} does not exist or is not an npz file."
        )

    ch = Checkpoint.load_from_file(cli.input_checkpoint)

    X, Y, Z = create_3d_surface_from_level_lines(
        ch.U - np.min(ch.U), ch.X, cli.h0, samples=cli.samples
    )

    x_step = float(X[0, 1] - X[0, 0])
    y_step = float(Y[1, 0] - Y[0, 0])

    dist, variogram = compute_variogram(Z, x_step=x_step, y_step=y_step)
    logger.info(f"Computed x,y variogram with {dist.size} entries.")

    dist_y, variogram_y = compute_1d_variogram(Z, axis_step=y_step, axis=0)
    logger.info(f"Computed y-only variogram with {dist_y.size} entries.")

    dist_x, variogram_x = compute_1d_variogram(Z, axis_step=x_step, axis=1)
    logger.info(f"Computed x-only variogram with {dist_x.size} entries.")

    n_bins, min_count = estimate_bin_params(dist.size)
    logger.info(
        f"Using {n_bins} bins with minimum count {min_count} for x,y variogram."
    )

    n_bins_y, min_count_y = estimate_bin_params(dist_y.size)
    logger.info(
        f"Using {n_bins_y} bins with minimum count {min_count_y} for y-only variogram."
    )

    n_bin_x, min_count_x = estimate_bin_params(dist_x.size)
    logger.info(
        f"Using {n_bin_x} bins with minimum count {min_count_x} for x-only variogram."
    )

    centers, binned_mean, counts = quantile_bin(dist, variogram, n_bins, min_count)

    centers_y, binned_mean_y, counts_y = quantile_bin(
        dist_y, variogram_y, n_bins_y, min_count_y
    )

    centers_x, binned_mean_x, counts_x = quantile_bin(
        dist_x, variogram_x, n_bin_x, min_count_x
    )

    if centers.size == 0:
        logger.warning("No valid bins found for x,y variogram.")
    if centers_y.size == 0:
        logger.warning("No valid bins found for y-only variogram.")
    if centers_x.size == 0:
        logger.warning("No valid bins found for x-only variogram.")

    # plot setup
    fig = plt.figure(figsize=(11, 6))
    subfig_main, subfig_side = fig.subfigures(
        1, 2, width_ratios=[4.2, 1.1], wspace=0.08
    )
    ax = subfig_main.subplots(1, 1)
    ax_y, ax_x = subfig_side.subplots(2, 1)
    subfig_side.subplots_adjust(hspace=0.35)

    # Main plot
    ax.scatter(dist, variogram, s=4, alpha=0.10, color="gray", label="Raw $x,y$ data")
    ax.plot(centers, binned_mean, color="C1", lw=2, label="Equal population mean")
    ax.plot(dist_y, variogram_y, color="C2", lw=1.2, alpha=0.7, label="$y$-only mean")
    ax.plot(dist_x, variogram_x, color="C3", lw=1.2, alpha=0.7, label="$x$-only mean")

    ax.set_xlim(dist.min(), dist.max())
    ax.set_ylim(np.nanmin(variogram), np.nanmax(variogram))
    ax.set_xlabel(r"Lag distance $r = \left|r_i - r_j\right|$")
    ax.set_ylabel("Average $(z_i - z_j)^2$")
    ax.set_title("Surface Variograms")
    ax.legend()

    # Y variogram
    ax_y.plot(dist_y, variogram_y, color="C2", lw=1.5)
    ax_y.scatter(
        centers_y, binned_mean_y, s=14, color="C2", edgecolor="k", linewidth=0.4
    )

    ax_y.set_xlim(dist_y.min(), dist_y.max())
    ax_y.set_ylim(np.nanmin(variogram_y), np.nanmax(variogram_y))
    ax_y.set_title("Y variogram", fontsize=9)

    # X variogram
    ax_x.plot(dist_x, variogram_x, color="C3", lw=1.5)
    ax_x.scatter(
        centers_x, binned_mean_x, s=14, color="C3", edgecolor="k", linewidth=0.4
    )
    ax_x.set_xlim(dist_x.min(), dist_x.max())
    ax_x.set_ylim(np.nanmin(variogram_x), np.nanmax(variogram_x))

    ax_x.set_title("X variogram", fontsize=9)
    ax_x.tick_params(labelsize=8)
    ax_x.set_xlabel(r"Lag distance $r = \left|r_i - r_j\right|$", fontsize=9)

    plt.show()


if __name__ == "__main__":
    main()
