from pathlib import Path

from tap import Tap

from libs.checkpoints import Checkpoint


class CLI(Tap):
    input_checkpoint: Path

    def configure(self):
        self.add_argument(
            "input_checkpoint", type=Path, help="Path to the input checkpoint file."
        )


def main():
    cli = CLI()
    cli.parse_args()
    if not cli.input_checkpoint.exists() and cli.input_checkpoint.suffix != ".npz":
        raise FileNotFoundError(
            f"Checkpoint file {cli.input_checkpoint} not found or is not an .npz file."
        )
    ch = Checkpoint.load_from_file(cli.input_checkpoint)

    print(f"== Checkpoint Data ==")
    print(f"Number of steps: {ch.K}")
    print(f"Number of spatial points: {ch.M}")
    print(f"Step tangential length (x): {ch.L}")
    print(f"Initial step spacing: {ch.c}")
    print(f"RHS type: {ch.f_type.name}")
    print(f"Gamma {ch.D}")
    print(f"Time {ch.iter*ch.dt} (step {ch.iter})")
    print(f"Total time: {ch.T}")


if __name__ == "__main__":
    main()
