import logging
import os
from pathlib import Path

import joblib

os.environ["JAX_PLATFORMS"] = "cpu"

from tap import Tap

from libs.checkpoints import Checkpoint
from viz2d import plot_dir, plot_single

logger = logging.getLogger("ORB2CH_CONVERTER")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _convert_checkpoint(checkpoint_dir: Path, output_dir: Path) -> None:
    Checkpoint.load_from_orbax(checkpoint_dir).save_to_file(
        output_dir / checkpoint_dir.name
    )


class CLI(Tap):
    input_orb: Path  # Path to the input orbax checkpoint directory or a directory containing orbax checkpoints
    output_ch: Path  # Path to the output .npz file or a directory containing the converted .npz files
    parallelism: int = 1  # Number of parallel processes to use
    with_plot: bool = False  # Whether to plot the checkpoint with viz2d
    publication_plot: bool = False  # Whether to use publication style for the plots

    def configure(self):
        self.add_argument(
            "input_orb", type=Path, help="Path to the input orbax checkpoint directory."
        )
        self.add_argument("output_ch", type=Path, help="Path to the output .npz file.")


def is_base_ch_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    # A single checkpoint dir has state/ and extra_metadata/ inside it
    if (path / "state").is_dir() and (path / "extra_metadata").is_dir():
        return False
    # The base dir should have at least one numeric subdir
    return any(x.is_dir() and x.name.isdigit() for x in path.iterdir())


def main():
    cli = CLI().parse_args()

    if is_base_ch_dir(cli.input_orb):
        sub_dirs = [
            x for x in cli.input_orb.iterdir() if x.is_dir() and x.name.isdigit()
        ]
        cli.output_ch.mkdir(parents=True, exist_ok=True)

        with joblib.Parallel(n_jobs=cli.parallelism) as parallel:
            logger.info(f"Converting {len(sub_dirs)} checkpoints to .npz files...")

            parallel(
                joblib.delayed(_convert_checkpoint)(
                    checkpoint_dir=sub_dir, output_dir=cli.output_ch
                )
                for sub_dir in sub_dirs
            )

            if cli.with_plot:
                logger.info(f"Plotting {len(sub_dirs)} checkpoints.")
                plot_dir(
                    cli.output_ch,
                    show=False,
                    publication=cli.publication_plot,
                    parallelism=cli.parallelism,
                )
    else:
        checkpoint = Checkpoint.load_from_orbax(cli.input_orb)

        checkpoint.save_to_file(cli.output_ch)
        if cli.with_plot:
            logger.info(f"Plotting {cli.output_ch} checkpoint.")
            plot_single(
                cli.output_ch,
                output_dir=cli.output_ch.parent,
                show=False,
                publication=cli.publication_plot,
            )


if __name__ == "__main__":
    main()
