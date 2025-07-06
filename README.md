# 2D+t step Bunching Finite Difference Method

## Installing

1. Install poetry 2.0+
    ```bash
    $ pip3 install poetry
    ```
2. Install the dependencies
    ```bash
    $ poetry sync
    ```
3. Get help
    ```bash
    $ poetry run python -m bunching_finite_diff --help
    ```
4. Run simulation with defaults
    ```bash
    $ poetry run python -m bunching_finite_diff
    ``
5. **OR** Continue simulation from checkpoint
    ```bash
    $ poetry run python -m bunching_finite_diff checkpoint <checkpoint_file.npz>
    ```