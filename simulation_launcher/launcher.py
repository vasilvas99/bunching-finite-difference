import hashlib
import json
import logging
import os
import tarfile
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import boto3
import yaml
from botocore.config import Config
from jinja2 import Environment
from pydantic import (BaseModel, Discriminator, Field, HttpUrl, SecretStr,
                      computed_field, model_validator)
from tap import Tap

from bunching_finite_diff.sharded_solver import DeviceLayout, ShardedSolver
from libs.rhs import RHSType

JINJA_ENV = Environment(
    extensions=["jinja2_ansible_filters.AnsibleCoreFiltersExtension"]
)


class CLI(Tap):
    ...

    def configure(self):
        self.add_argument("CONFIG_PATH", type=Path, help="Path to the run config file.")


def stable_hash(obj) -> str:
    """Generate a stable SHA256 hash of a JSON-serializable object.

    This hash is consistent across Python sessions unlike the built-in hash().
    """
    # Convert object to JSON string for hashing
    if isinstance(obj, dict):
        json_str = json.dumps(obj, sort_keys=True, default=str)
    else:
        json_str = json.dumps(obj, sort_keys=True, default=str)

    # Return hex digest (first 16 chars for brevity)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class SimulationRunConfig(BaseModel):
    type: Literal["SimulationRunConfig"] = Field(
        default="SimulationRunConfig", frozen=True
    )
    job_name: str
    checkpoints_dir: Path | None = None
    step_count: int = Field(20, description="Number of steps in the system")
    step_nodes: int = Field(100, description="Number of nodes per step")
    step_length: float = Field(
        5.0, description="Length of each step in arbitrary units"
    )
    time_max: float = Field(
        10.0, description="Maximum simulation time in arbitrary units"
    )
    time_step: float = Field(
        0.01, description="Time step for the simulation in arbitrary units"
    )
    step_stiffness: float = Field(
        0.1, description="Stiffness of each step in arbitrary units"
    )
    rhs_type: RHSType = Field(
        RHSType.MM2_a50_r1_jax, description="Nonlinear coupling function"
    )
    initial_spacing: float = Field(
        1.0, description="Initial spacing between steps (u_i(0,x) = initial_spacing)"
    )
    solver_tolerance: float = Field(
        1e-6, description="Tolerance for the numerical solver"
    )
    checkpoint_interval: int = Field(
        10, description="Interval (in time steps) to save checkpoints"
    )

    @model_validator(mode="after")
    def compute_checkpoints_dir_if_missing(self):
        if self.checkpoints_dir is None:
            self.checkpoints_dir = Path(f"checkpoints/{self.job_name}")
        return self

    @computed_field
    @property
    def is_completed(self) -> bool:
        completion_marker = self.checkpoints_dir / ".SIMULATION_COMPLETED"
        if completion_marker.exists():
            return True
        return False

    def __hash__(self):
        # Use stable hash based on model dump
        config_data = {
            "type": self.type,
            "job_name": self.job_name,
            "checkpoints_dir": (
                self.checkpoints_dir.as_posix() if self.checkpoints_dir else None
            ),
            "step_count": self.step_count,
            "step_nodes": self.step_nodes,
            "step_length": self.step_length,
            "time_max": self.time_max,
            "time_step": self.time_step,
            "step_stiffness": self.step_stiffness,
            "rhs_type": str(self.rhs_type),
            "initial_spacing": self.initial_spacing,
            "solver_tolerance": self.solver_tolerance,
            "checkpoint_interval": self.checkpoint_interval,
        }
        return int(stable_hash(config_data), 16)


class UploadJobConfig(BaseModel):
    type: Literal["UploadJobConfig"] = Field(default="UploadJobConfig", frozen=True)
    s3_url: HttpUrl
    access_key_id: SecretStr
    secret_access_key: SecretStr
    path: Path

    @computed_field
    @property
    def job_name(self) -> str:
        return f"Upload to {self.s3_url}"

    def __hash__(self):
        config_data = {
            "type": self.type,
            "s3_url": str(self.s3_url),
            "path": self.path.as_posix(),
        }
        return int(stable_hash(config_data), 16)


class SimulationLauncherConfig(BaseModel):
    config_id: str = Field(
        ..., description="Unique identifier for the launcher configuration"
    )
    device_layout: DeviceLayout = Field(
        default_factory=lambda: DeviceLayout(x_devices=1, y_devices=1)
    )
    workdir: Path | None = Field(None, description="Working directory for the launcher")
    max_retry_count: int = Field(
        3, description="Maximum number of retries for failed simulation runs"
    )
    jobs: list[
        Annotated[SimulationRunConfig | UploadJobConfig, Discriminator("type")]
    ] = Field(
        default_factory=list,
        description="List of simulation runs and upload jobs to execute",
    )

    def __hash__(self):
        config_data = {
            "config_id": self.config_id,
            "device_mesh": str(self.device_layout),
            "jax_prealloc_gpu_mem_fract": self.jax_prealloc_gpu_mem_fract,
            "workdir": self.workdir.as_posix() if self.workdir else None,
            "max_retry_count": self.max_retry_count,
            "jobs": [str(hash(job)) for job in self.jobs],
        }
        return int(stable_hash(config_data), 16)


class BaseLog(BaseModel):
    job_hash: str
    job_name: str

    @property
    def is_successful(self) -> bool:
        raise NotImplementedError


class SuccessfulRunLog(BaseLog):
    pass

    @property
    def is_successful(self) -> bool:
        return True


class FailedRunLog(BaseLog):
    error_message: str
    retried_count: int

    @property
    def is_successful(self) -> bool:
        return False


class LauncherLog(BaseModel):
    successful_jobs: dict[str, SuccessfulRunLog] = Field(
        default_factory=dict,
        description="Dict of simulation/upload job hashes and info that completed successfully",
    )
    failed_jobs: dict[str, FailedRunLog] = Field(
        default_factory=dict,
        description="Dict of simulation/upload job hashes and info that failed",
    )

    @model_validator(mode="after")
    def check_no_overlap_success_failed(self):
        overlap = set(self.successful_jobs.keys()) & set(self.failed_jobs.keys())
        if overlap:
            raise ValueError(f"Overlap between successful and failed jobs: {overlap}")
        return self

    def get(self, job_hash: str, default: BaseLog | None = None):
        if job_hash in self.successful_jobs:
            return self.successful_jobs[job_hash]
        elif job_hash in self.failed_jobs:
            return self.failed_jobs[job_hash]
        else:
            return default

    def __contains__(self, job_hash: str) -> bool:
        return job_hash in self.successful_jobs or job_hash in self.failed_jobs

    def __getitem__(self, job_hash: str) -> SuccessfulRunLog | FailedRunLog:
        if job_hash in self.successful_jobs:
            return self.successful_jobs[job_hash]
        elif job_hash in self.failed_jobs:
            return self.failed_jobs[job_hash]
        else:
            raise KeyError(f"Job hash {job_hash} not found in log.")

    def __setitem__(self, job_hash: str, value: SuccessfulRunLog | FailedRunLog):
        if isinstance(value, SuccessfulRunLog):
            self.successful_jobs[job_hash] = value
            if job_hash in self.failed_jobs:
                del self.failed_jobs[job_hash]
        elif isinstance(value, FailedRunLog):
            self.failed_jobs[job_hash] = value
            if job_hash in self.successful_jobs:
                del self.successful_jobs[job_hash]
        else:
            raise ValueError("Value must be either SuccessfulRunLog or FailedRunLog.")


def load_launcher_config(config_path: Path) -> SimulationLauncherConfig:
    config_content = config_path.read_text()
    env_vars = {f"env.{key}": value for key, value in os.environ.items()}

    # First render: render Jinja with PWD, DATE and current env variables
    template_vars_initial = {
        "PWD": Path.cwd().as_posix(),
        "DATE": datetime.now().isoformat(),
        **env_vars,
    }

    template = JINJA_ENV.from_string(config_content)
    rendered_content_initial = template.render(**template_vars_initial)

    preliminary_config = yaml.safe_load(rendered_content_initial)
    workdir_value = (
        preliminary_config.get("workdir")
        if isinstance(preliminary_config, dict)
        else None
    )

    # Second render: render Jinja again with WORKDIR now available
    template_vars_final = {
        "PWD": Path.cwd().as_posix(),
        "DATE": datetime.now().isoformat(),
        "WORKDIR": workdir_value if workdir_value else "",
    }
    rendered_content_final = template.render(**template_vars_final)

    # Parse YAML after final Jinja rendering
    config_dict = yaml.safe_load(rendered_content_final)
    return SimulationLauncherConfig.model_validate(config_dict)


class Launcher:
    def __init__(self, config: SimulationLauncherConfig):
        self.config = config

        self.config.workdir = self.config.workdir.resolve()

        if not self.config.workdir.exists():
            self.config.workdir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.log_path = (
            Path(__file__).parent / f".launcher_log_{self.config.config_id}.json"
        )

        if self.log_path.exists():
            self.log = LauncherLog.model_validate_json(self.log_path.read_text())
            self.logger.info(f"Loaded launcher log from {self.log_path}")
        else:
            self.log = LauncherLog()
            self.logger.info("Starting with fresh launcher log")

        for job in self.config.jobs:
            job_hash = str(hash(job))
            if isinstance(job, SimulationRunConfig) and job.is_completed:
                self.log[job_hash] = SuccessfulRunLog(
                    job_hash=job_hash,
                    job_name=job.job_name,
                )
                self.logger.info(
                    f"Marking job '{job.job_name}' as successful (completed marker found)."
                )

        self._persist_log()

    def execute(self):
        simulation_failed = any(
            isinstance(job, SimulationRunConfig) and job_hash in self.log.failed_jobs
            for job in self.config.jobs
            for job_hash in [str(hash(job))]
        )

        for job in self.config.jobs:
            job_hash = str(hash(job))
            job_log: SuccessfulRunLog | FailedRunLog = self.log.get(job_hash, None)

            # Skip successful simulations (always)
            if (
                job_log is not None
                and job_log.is_successful
                and isinstance(job, SimulationRunConfig)
            ):
                self.logger.info(f"Skipping already successful job: {job.job_name}")
                continue

            # Skip successful uploads unless a simulation failed
            if (
                job_log is not None
                and job_log.is_successful
                and isinstance(job, UploadJobConfig)
                and not simulation_failed
            ):
                self.logger.info(f"Skipping already successful job: {job.job_name}")
                continue

            if job_log is not None and not job_log.is_successful:
                self.logger.info(
                    f"A previous attempt for job '{job.job_name}' failed. Retrying..."
                )
                if isinstance(job, SimulationRunConfig):
                    simulation_failed = True

            try:
                if isinstance(job, SimulationRunConfig):
                    self._execute_simulation_run(job)
                elif isinstance(job, UploadJobConfig):
                    self._execute_upload_job(job)
                else:
                    raise ValueError(f"Unknown job type: {type(job)}")

                self.log[job_hash] = SuccessfulRunLog(
                    job_hash=job_hash,
                    job_name=job.job_name,
                )
                self.logger.info(f"Job '{job.job_name}' completed successfully.")
            except Exception as e:
                self.log[job_hash] = FailedRunLog(
                    job_hash=job_hash,
                    job_name=job.job_name,
                    error_message=str(e),
                    retried_count=(
                        self.log[job_hash].retried_count + 1
                        if job_hash in self.log.failed_jobs
                        else 1
                    ),
                )
                self.logger.error(f"Job '{job.job_name}' failed with error: {e}")
                if isinstance(job, SimulationRunConfig):
                    simulation_failed = True
            finally:
                self._persist_log()

    def _persist_log(self):
        self.log_path.write_text(self.log.model_dump_json(indent=2))

    def _execute_simulation_run(self, job: SimulationRunConfig):
        ch_dir = self.config.workdir.resolve() / job.checkpoints_dir
        try:
            solver = ShardedSolver.load_from_last_checkpoint(
                ch_dir, device_layout=self.config.device_layout
            )
            logging.info(f"Resuming simulation '{job.job_name}' from last checkpoint.")
        except Exception:
            logging.info(f"Starting new simulation '{job.job_name}'.")
            solver = ShardedSolver(
                K=job.step_count,
                M=job.step_nodes,
                L=job.step_length,
                T=job.time_max,
                D=job.step_stiffness,
                f_type=job.rhs_type,
                c=job.initial_spacing,
                dt=job.time_step,
                device_layout=self.config.device_layout,
                checkpoint_interval=job.checkpoint_interval,
                output_dir=ch_dir,
            )
        solver.solve()

    def _execute_upload_job(self, job: UploadJobConfig):
        # tar and gzip the path
        archive_name = f"{job.path.name}_{self.config.config_id}.tar.gz"
        archive_path = job.path.parent / archive_name
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(job.path, arcname=job.path.name)
        self.logger.info(f"Created archive {archive_path} for upload.")

        self.logger.info(f"Uploading {archive_path} to {job.s3_url}...")
        bucket_name = job.s3_url.path.lstrip("/").split("/")[0]
        api_url = f"{job.s3_url.scheme}://{job.s3_url.host}"

        with closing(
            boto3.client(
                "s3",
                endpoint_url=api_url,
                aws_access_key_id=job.access_key_id.get_secret_value(),
                aws_secret_access_key=job.secret_access_key.get_secret_value(),
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
        ) as s3_client:
            s3_client.upload_file(archive_path.as_posix(), bucket_name, archive_name)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    cli = CLI().parse_args()
    config = load_launcher_config(cli.CONFIG_PATH)
    logger.info(f"Loaded launcher config with ID: {config.config_id}")
    launcher = Launcher(config)
    launcher.execute()


if __name__ == "__main__":
    main()
