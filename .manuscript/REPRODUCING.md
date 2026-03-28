# Reproduction steps

Note that the following steps are expected to take a long time. Using machines with GPUs/high CPU-count is recommended.
The following steps below assume that your machine have at least 1 Nvidia GPU with CUDA12+ and fp64 capabilities and a
CPU with at least 8 cores.

If you need help, feel free to contact the author via one of the contacts provided on his orcid
page [here](https://orcid.org/0000-0002-7651-2462).

> Note: the current version is a single-host one, though it supports multiple GPUs per host. If you need a multi-host
> version, contact the author.

# General installation and simulation start

> Note: If you are running on an HPC cluster, where system python installation might be outdated, create a conda environment
> and run the commands below in that conda environment. It might be necessary to install jax and orbax in the virtual
> env from conda forge if the system glibc is too old.

1. Start by following the steps from the [README.md](../README.md) for installing all the required dependencies:

    ```bash
    uv sync && uv sync --extra cuda12
    ```

2. Run the simulation launcher with the provided config file:
    ```bash
    uv run -m simulation_launcher .manuscript/manuscript_runs.yaml.j2
    ```

3. Step **2.** will produce a directory called `manuscript` within the current working directory.

## Post-processing

Assuming that all the simulation runs have completed, run the following bash script to convert them to `.npz` checkpoints
and plot them:

```bash
for d in $PWD/manuscript_revised/run_*; do
    uv run -m utils.orb2ch \
        --parallelism 8 \
        --with_plot \
        --publication_plot \
        "$d" \
        "${d}_npz"
done
```

The `.npz` checkpoints are inputs for all other scripts and provide a compressed and flexible format to work with.
The handling logic for `.npz` checkpoints can be found in `libs/checkpoints.py` which allows you to load/save them
and presents you with a convenient dataclass to work with in code.

## Other

If you want to exploit a multicore CPU for calculations, instead of a GPU, note that 
from JAX's point-of-view, each CPU is treated like a separate device along which the matrices are sharded. This imposes
significant communication overhead for smaller systems and runs the risk of oversubscribing the CPU for larger ones,
which is undesirable. Still, one might want to experiment with that.

To do so, set the following environment variable, before running the launcher:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=$(nproc)"
```

And modify the sharding spec in the `.yaml.j2` file:
```
config_id: manuscript_sweep
device_layout:
  x_devices: 1 # <----- modify
  y_devices: 1 # <----- modify
```

Note that `x_devices` shards along the number-of-steps axis and `y_devices` shards along the number-of-nodes axis.
From experimentation, modifying just `x_devices` to equal the total number of devices and keeping `y_devices` fixed to `1`
produces the least communication overhead.

Still, experimenting with the sharding topology is recommended for a given problem setup.
