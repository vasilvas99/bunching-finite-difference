# 2D+t step Bunching Finite Difference Method

## Installing

1. Install poetry 2.0+
    ```bash
    pip3 install uv
    ```
2. Install the dependencies
    ```bash
    uv sync
    ```
3. Get help
    ```bash
    uv run python3 -m bunching_finite_diff --help
    ```
4. Run simulation with defaults
    ```bash
    uv run python3 -m bunching_finite_diff
    ```
5. **OR** Continue simulation from checkpoint
    ```bash
    uv run python3 -m bunching_finite_diff checkpoint <checkpoint_file.npz>
    ```

6. **OPTIONAL** Install with GPU/TPU support
    ```bash
      uv sync --group <cuda13|cuda12|tpu>
    ```
