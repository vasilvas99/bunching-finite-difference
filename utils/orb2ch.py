from pathlib import Path

from tap import Tap

from libs.checkpoints import Checkpoint


class CLI(Tap):
    input_orb: Path  # Path to the input orbax checkpoint directory
    output_ch: Path  # Path to the output .npz file

    def configure(self):
        self.add_argument(
            "input_orb", type=Path, help="Path to the input orbax checkpoint directory."
        )
        self.add_argument("output_ch", type=Path, help="Path to the output .npz file.")


def main():
    cli = CLI().parse_args()
    checkpoint = Checkpoint.load_from_orbax(cli.input_orb)
    checkpoint.save_to_file(cli.output_ch)


if __name__ == "__main__":
    main()
