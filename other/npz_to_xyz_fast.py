#!/usr/bin/env python3
"""
Fast NPZ to XYZ converter for step bunching visualization.

Optimized version with hardcoded parameters:
- h0 = 1.0 (step height)
- step_edge_fill_rate = 1 (one atom per step height)
- Vectorized step edge generation
- Pre-allocated arrays for better memory efficiency
- Compression enabled by default

Converts npz checkpoint files to XYZ format for VMD/OVITO visualization.
"""

import gzip
import logging
from pathlib import Path

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

# Hardcoded constants for optimization
H0 = 1.0  # Step height
STEP_EDGE_FILL_RATE = 1  # One atom per step height
STEP_THRESHOLD = 0.9  # Threshold for detecting step edges (90% of h0)


class CLI(Tap):
    input: Path  # Path to input npz file or directory containing npz files
    output: Path = None  # Output xyz file path (for single file) or trajectory file (for directory)
    atoms_per_terrace: int = (
        500  # Number of atoms in y-direction (reduced from 1000 for better performance)
    )
    compress: bool = True  # Use gzip compression (.xyz.gz format) - enabled by default for smaller files
    trajectory: bool = True  # Create single trajectory file from multiple npz files (directory mode only)

    def configure(self):
        self.add_argument(
            "input",
            type=Path,
            help="Path to an input checkpoint file or a directory containing checkpoint files.",
        )


def create_xyz_from_checkpoint(checkpoint: Checkpoint, atoms_per_terrace: int):
    """
    Create XYZ atom coordinates from a checkpoint (optimized version).

    Optimizations:
    - Vectorized step edge detection
    - Pre-allocated arrays instead of dynamic lists
    - Hardcoded h0=1.0 and fill_rate=1

    Args:
        checkpoint: Checkpoint object containing U (step positions) and X (x coords)
        atoms_per_terrace: Number of atoms to place on each terrace (samples in y direction)

    Returns:
        atom_types: Array of atom types (all type 3)
        coordinates: Nx3 array of (x, y, z) coordinates
    """
    U = checkpoint.U  # Shape: (K, M) where K is number of steps, M is x-grid points
    X = checkpoint.X  # x coordinates
    K = checkpoint.K  # Number of steps
    M = len(X)

    # Subtract minimum to center surface
    U_norm = U - np.min(U)

    # Get ranges
    y_min = np.min(U_norm)
    y_max = np.max(U_norm)
    x_min = np.min(X)
    x_max = np.max(X)

    logger.debug(
        f"X range: [{x_min:.3f}, {x_max:.3f}], Y range: [{y_min:.3f}, {y_max:.3f}]"
    )
    logger.debug(f"Number of steps (K): {K}, Grid points (M): {M}")

    # Create grid
    n_x_samples = M
    n_y_samples = atoms_per_terrace

    x_grid = np.arange(n_x_samples, dtype=np.float32)
    y_grid = np.arange(n_y_samples, dtype=np.float32)

    # Physical coordinates
    x_physical = np.linspace(x_min, x_max, n_x_samples)
    y_physical = np.linspace(y_min, y_max, n_y_samples)

    # Pre-allocate arrays (terrace atoms + estimated step edge atoms)
    # Estimate: ~30% of y-samples will have step edges, with avg K steps per edge
    n_terrace_atoms = n_x_samples * n_y_samples
    estimated_step_atoms = int(n_y_samples * 0.3 * K * STEP_EDGE_FILL_RATE)
    estimated_total = n_terrace_atoms + estimated_step_atoms

    atom_types = np.full(estimated_total, 3, dtype=np.int32)
    coordinates = np.zeros((estimated_total, 3), dtype=np.float32)

    atom_idx = 0
    prev_height_array = None

    logger.debug("Generating atoms with vectorized step edge detection...")

    for j, (y_grid_coord, y_phys) in enumerate(zip(y_grid, y_physical)):
        # Count how many steps each x position has crossed (vectorized)
        terrace_levels = np.sum(U_norm <= y_phys, axis=0)  # Shape: (M,)
        height_array = terrace_levels * H0

        # Vectorized step edge detection
        if prev_height_array is not None:
            height_diff = height_array - prev_height_array

            # Find positions with significant height jumps (vectorized)
            step_mask = height_diff >= (H0 * STEP_THRESHOLD)
            step_indices = np.where(step_mask)[0]

            # Process step edges
            for i in step_indices:
                prev_h = prev_height_array[i]
                curr_h = height_array[i]
                diff = curr_h - prev_h

                # Calculate number of intermediate atoms
                num_intermediate = int(diff / H0) * STEP_EDGE_FILL_RATE

                if num_intermediate > 0:
                    # Check if we need to expand arrays
                    if atom_idx + num_intermediate >= len(atom_types):
                        # Expand by 50%
                        new_size = int(len(atom_types) * 1.5)
                        atom_types = np.pad(
                            atom_types,
                            (0, new_size - len(atom_types)),
                            constant_values=3,
                        )
                        coordinates = np.pad(
                            coordinates, ((0, new_size - len(coordinates)), (0, 0))
                        )

                    # Generate intermediate atoms (vectorized)
                    fractions = np.arange(1, num_intermediate + 1) / (
                        num_intermediate + 1
                    )
                    intermediate_heights = prev_h + fractions * diff

                    # Store step edge atoms
                    end_idx = atom_idx + num_intermediate
                    coordinates[atom_idx:end_idx, 0] = x_grid[i] + 1
                    coordinates[atom_idx:end_idx, 1] = y_grid_coord + 0.5
                    coordinates[atom_idx:end_idx, 2] = intermediate_heights
                    atom_idx = end_idx

        # Store terrace atoms for this y-slice
        # Check if we need to expand arrays
        if atom_idx + n_x_samples >= len(atom_types):
            new_size = int(len(atom_types) * 1.5)
            atom_types = np.pad(
                atom_types, (0, new_size - len(atom_types)), constant_values=3
            )
            coordinates = np.pad(
                coordinates, ((0, new_size - len(coordinates)), (0, 0))
            )

        end_idx = atom_idx + n_x_samples
        coordinates[atom_idx:end_idx, 0] = x_grid + 1
        coordinates[atom_idx:end_idx, 1] = y_grid_coord + 1
        coordinates[atom_idx:end_idx, 2] = height_array
        atom_idx = end_idx

        prev_height_array = height_array

    # Trim to actual size
    atom_types = atom_types[:atom_idx]
    coordinates = coordinates[:atom_idx]

    n_step_atoms = atom_idx - n_terrace_atoms
    logger.debug(f"Generated {atom_idx} total atoms ({n_step_atoms} step edge atoms)")

    return atom_types, coordinates


def write_xyz_file(output_path: Path, atom_types, coordinates, compress: bool = True):
    """
    Write XYZ file in VMD/OVITO compatible format.

    Args:
        output_path: Path to output file
        atom_types: Array of atom types
        coordinates: Nx3 array of coordinates
        compress: If True, write gzip compressed file (.xyz.gz)
    """
    n_atoms = len(atom_types)

    # Adjust output path for compression
    if compress and not str(output_path).endswith(".gz"):
        output_path = Path(str(output_path) + ".gz")

    open_func = gzip.open if compress else open
    mode = "wt" if compress else "w"

    with open_func(output_path, mode) as f:
        # Write header
        f.write(f"{n_atoms:>12}\n")
        f.write("\n")

        # Write atoms (optimized: batch format strings)
        lines = [
            f"{at:>12} {x:>11.0f} {y:>11.0f} {z:>11.0f}\n"
            for at, (x, y, z) in zip(atom_types, coordinates)
        ]
        f.writelines(lines)

    logger.debug(f"Wrote {n_atoms} atoms to {output_path}")


def append_frame_to_trajectory(
    file_handle, atom_types, coordinates, frame_number: int, iteration: int
):
    """
    Append a single frame to an open trajectory file.

    Args:
        file_handle: Open file handle to write to
        atom_types: Array of atom types
        coordinates: Nx3 array of coordinates
        frame_number: Frame index in trajectory
        iteration: Simulation iteration number
    """
    n_atoms = len(atom_types)

    # Write header for this frame
    file_handle.write(f"{n_atoms:>12}\n")
    file_handle.write(f"Frame {frame_number} (iteration {iteration})\n")

    # Write atoms (optimized: batch format strings)
    lines = [
        f"{at:>12} {x:>11.0f} {y:>11.0f} {z:>11.0f}\n"
        for at, (x, y, z) in zip(atom_types, coordinates)
    ]
    file_handle.writelines(lines)


def process_single_npz(
    input_path: Path,
    output_path: Path,
    atoms_per_terrace: int,
    compress: bool = True,
):
    """Process a single npz file and generate xyz output."""
    logger.info(f"Processing {input_path}")

    # Load checkpoint
    checkpoint = Checkpoint.load_from_file(input_path)

    # Create xyz data
    atom_types, coordinates = create_xyz_from_checkpoint(checkpoint, atoms_per_terrace)

    # Write xyz file
    write_xyz_file(output_path, atom_types, coordinates, compress)

    logger.info(f"Successfully created {output_path}")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    atoms_per_terrace: int,
    compress: bool = True,
):
    """Process all npz files in a directory."""
    npz_files = list(input_dir.glob("*.npz"))

    if not npz_files:
        raise ValueError(f"No .npz files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Found {len(npz_files)} npz files to process")

    for npz_file in npz_files:
        output_file = output_dir / f"{npz_file.stem}.xyz"
        process_single_npz(npz_file, output_file, atoms_per_terrace, compress)


def create_trajectory(
    input_dir: Path,
    output_path: Path,
    atoms_per_terrace: int,
    compress: bool = True,
):
    """
    Process all npz files in a directory and create a single trajectory file.

    Args:
        input_dir: Directory containing npz checkpoint files
        output_path: Path to output trajectory xyz file
        atoms_per_terrace: Number of atoms in y-direction
        compress: If True, write gzip compressed trajectory
    """
    # Find all npz files
    npz_files = list(input_dir.glob("*.npz"))

    if not npz_files:
        raise ValueError(f"No .npz files found in {input_dir}")

    # Sort by iteration number
    logger.info(f"Found {len(npz_files)} npz files, sorting by iteration...")

    file_iterations = []
    for npz_file in npz_files:
        checkpoint = Checkpoint.load_from_file(npz_file)
        file_iterations.append((npz_file, checkpoint.iter))

    # Sort by iteration number
    file_iterations.sort(key=lambda x: x[1])
    sorted_files = [f for f, _ in file_iterations]

    logger.info(
        f"Creating trajectory with {len(sorted_files)} frames from iteration "
        f"{file_iterations[0][1]} to {file_iterations[-1][1]}"
    )

    # Adjust output path for compression
    if compress and not str(output_path).endswith(".gz"):
        output_path = Path(str(output_path) + ".gz")

    open_func = gzip.open if compress else open
    mode = "wt" if compress else "w"

    # Create trajectory file
    with open_func(output_path, mode) as traj_file:
        for frame_num, npz_file in enumerate(sorted_files):
            logger.debug(f"Processing frame {frame_num}: {npz_file.name}")

            # Load checkpoint
            checkpoint = Checkpoint.load_from_file(npz_file)

            # Create xyz data
            atom_types, coordinates = create_xyz_from_checkpoint(
                checkpoint, atoms_per_terrace
            )

            # Append to trajectory
            append_frame_to_trajectory(
                traj_file, atom_types, coordinates, frame_num, checkpoint.iter
            )

    logger.info(f"Successfully created trajectory file: {output_path}")
    logger.info(
        f"Total frames: {len(sorted_files)}, Total atoms per frame: {len(atom_types)}"
    )


def main():
    cli = CLI().parse_args()

    if cli.input.is_file():
        # Single file processing
        if cli.output is None:
            output_path = cli.input.with_suffix(".xyz")
        else:
            output_path = cli.output

        process_single_npz(
            cli.input,
            output_path,
            cli.atoms_per_terrace,
            cli.compress,
        )

    elif cli.input.is_dir():
        # Directory processing
        if cli.trajectory:
            # Create single trajectory file
            if cli.output is None:
                output_path = cli.input / "trajectory.xyz"
            else:
                output_path = cli.output

            create_trajectory(
                cli.input,
                output_path,
                cli.atoms_per_terrace,
                cli.compress,
            )
        else:
            # Create separate files for each npz
            if cli.output is None:
                output_dir = cli.input / "xyz_output"
            else:
                output_dir = cli.output

            process_directory(
                cli.input,
                output_dir,
                cli.atoms_per_terrace,
                cli.compress,
            )
    else:
        raise ValueError(f"Input {cli.input} is neither a file nor a directory")

    logger.info("Conversion complete!")


if __name__ == "__main__":
    main()
