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
      uv sync --extra <cuda13|cuda12|tpu>
    ```

## Running a simulation plan with the launcher

This repository provides a simulation launcher that can run a series of simulations
and upload the results to a specified s3 bucket.

To run with a job plan use:

```bash
 uv run python3 -m simulation_launcher <plan_file.plan.yaml>
```

Currently, there are two kinds of allowed jobs:

```yaml
  - type: SimulationRunConfig
    job_name: "simulation_run_1"
    checkpoints_dir: "{{ WORKDIR }}/checkpoints/sim_run_1"
    step_count: 20
    step_nodes: 100
    step_length: 5.0
    time_max: 0.01
    time_step: 1e-4
    step_stiffness: 0.1
    rhs_type: "MM2_a50_r1_jax"
    initial_spacing: 1.0
    solver_tolerance: 1e-6
    checkpoint_interval: 50
```

and 

```yaml
  - type: UploadJobConfig
    s3_url: "https://s3.endpoint.mysite.com/bucket_name"
    access_key_id: "EXAMPLE ACCESS KEY ID HERE"
    secret_access_key: "EXAMPLE ACCESS KEY HERE"
    path: "{{ WORKDIR }}"
```

To see a full example config go to `simulation_launcher/example/example_config.plan.yaml`

The `.plan.yaml` format allows for Jinja templating. The following variables are available:

- `{{ PWD }}` - The current working directory
- `{{ DATE }}` - The current date in YYYYMMDD format
- `{{ WORKDIR }}` - The workdir as defined in the workdir field of the plan file.
- `{{ env.VAR_NAME }}` - The value of the environment variable VAR_NAME

Further, the launcher support the jinja ansible features that help with yaml handling.
List is here: [https://pypi.org/project/jinja2-ansible-filters/](https://pypi.org/project/jinja2-ansible-filters/)