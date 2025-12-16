from __future__ import annotations

import logging
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import lineax
import optimistix as optx
import orbax.checkpoint as ocp
from pydantic import BaseModel
from tap import Tap

from libs.rhs import RHSType

jax.config.update(
    "jax_enable_x64", True
)  # Pretty important, otherwise convergence is horrendous

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


class DeviceLayout(BaseModel):
    x_devices: int
    y_devices: int


class CheckPointCLi(Tap):
    tol: float = 1e-6  # Tolerance for the solver
    device_layout_x: int = 1  # Number of devices in the x direction
    device_layout_y: int = 1  # Number of devices in the y direction

    def configure(self):
        self.add_argument(
            "CHECKPOINT_PATH", type=Path, help="Path to the checkpoint file."
        )


class CLI(Tap):
    K: int = 20  # Number of steps in the system
    M: int = 100  # Number of spatial grid points
    L: float = 5.0  # Length of the spatial domain
    T: float = 10.0  # Final integration time
    D: float = 1e-1  # Diffusion coefficient (equiv. to line tension)
    f: RHSType = RHSType.MM2_a50_r1_jax  # Nonlinear coupling function
    c: float = 1.0  # Initial spacing constant (u_i(0,x) = i * c)
    dt: float = 1e-4  # Time step
    tol: float = 1e-6  # Tolerance for the solver
    device_layout_x: int = 1  # Number of devices in the x direction
    device_layout_y: int = 1  # Number of devices in the y direction
    checkpoint_interval: int = 50  # Save a plot every this many timesteps
    checkpoints_dir: Path = Path(
        "checkpoint_ocp"
    )  # Directory to save Orbax checkpoints

    def configure(self):
        self.add_subparser(
            "checkpoint",
            CheckPointCLi,
            help="Load from a checkpoint file and continue solving the system.",
        )


def P(mesh, *args):
    return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))


@partial(jax.jit, static_argnames=["f", "K", "r", "c", "dt"])
def _build_residual(U, U_prev, K, r, c, dt, f):
    P = K * c
    U_jp = jnp.roll(U, -1, axis=1)
    U_jm = jnp.roll(U, 1, axis=1)
    lap = r * (U_jp - 2 * U + U_jm)

    U_im_raw = jnp.roll(U, 1, axis=0)
    U_ip_raw = jnp.roll(U, -1, axis=0)

    i_idx = jnp.arange(K)
    im_wrap = (i_idx == 0)[:, None]
    ip_wrap = (i_idx == K - 1)[:, None]

    Uim = jnp.where(im_wrap, U_im_raw - P, U_im_raw)
    Uip = jnp.where(ip_wrap, U_ip_raw + P, U_ip_raw)

    nonl = dt * f(Uim, U, Uip)
    F = U - U_prev - lap - nonl
    return F


class ShardedSolver:
    def __init__(
        self,
        K: int,
        M: int,
        L: float,
        T: float,
        D: float,
        f_type: RHSType,
        c: float,
        dt: float,
        device_layout: DeviceLayout | None = None,
        checkpoint_interval: int = 10,
        output_dir: Path | str = Path("./checkpoint_ocp"),
    ):
        """
        K: number of equations (periodic in i)
        M: number of spatial grid points
        L: spatial domain length [0, L]
        T: final time
        D: diffusion coefficient
        f_type: Type of nonlinear coupling f(u_{i-1}, u_i, u_{i+1})
        c: initial spacing constant (u_i(0,x) = i * c)
        dt: time step
        checkpoint_interval: interval to save checkpoints
        """
        logger.info("Initializing JAX-based solver.")
        self.K = K
        self.M = M
        self.L = L
        self.T = T
        self.D = D
        self.f_type = f_type
        self.c = c
        self.dt = dt
        self.iter = 0
        self.dx = self.L / (M - 1)
        self.r = D * dt / self.dx**2

        if device_layout is None:
            device_layout = DeviceLayout(x_devices=1, y_devices=1)

        self.checkpoint_interval = checkpoint_interval
        self.checkpoints_dir = Path(output_dir)

        self.max_to_keep = max(
            1, int(jnp.ceil(self.T / (self.dt * self.checkpoint_interval)))
        )

        self.x = jnp.linspace(0, L, M)
        self.U = jnp.zeros((K, M))
        rand_seed = int.from_bytes(os.urandom(4), sys.byteorder)
        noise = jax.random.normal(jax.random.key(rand_seed), (K, M)) * (0.005 * c)
        self.U = jnp.arange(K)[:, None] * c + noise

        self.time_steps = int(jnp.ceil(T / dt))
        self.iter = 0

        self.mesh = jax.make_mesh(
            axis_shapes=(device_layout.x_devices, device_layout.y_devices),
            axis_names=("X", "Y"),
        )
        self.U = jax.device_put(self.U, P(self.mesh, "X", "Y"))
        self.build_residual = jax.jit(
            _build_residual,
            static_argnames=["f", "K", "r", "c", "dt"],
            out_shardings=P(self.mesh, "X", "Y"),
        )
        self.f = jax.jit(f_type.into_rhs(), out_shardings=P(self.mesh, "X", "Y"))

        self.make_dirs()

    @partial(jax.jit, static_argnames=["self", "f", "K", "r", "c", "dt"])
    def step(self, U_prev, K, r, c, dt, f, tol):
        solver = optx.Newton(
            atol=tol,
            rtol=tol,
            linear_solver=lineax.GMRES(atol=tol, rtol=tol),
            cauchy_termination=False,
        )

        result = optx.root_find(
            lambda x, _: self.build_residual(x, U_prev, K, r, c, dt, f),
            solver,
            U_prev,
            throw=False,
        )
        # hint to JAX to only shard the solution value
        jax.lax.with_sharding_constraint(result.value, P(self.mesh, "X", "Y"))
        return result.result, result.value

    def make_dirs(self):
        if self.checkpoints_dir.exists() and self.checkpoints_dir.is_dir():
            logger.warning(
                f"Output directory {self.checkpoints_dir} already exists!"
                f" This may overwrite previous checkpoints."
            )

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def prepare_checkpoint(self, iter: int):
        pytree = {
            "U": self.U,
            "X": self.x,
        }
        extra_metadata = {
            "K": self.K,
            "M": self.M,
            "L": self.L,
            "c": self.c,
            "T": self.T,
            "D": self.D,
            "f_type": str(self.f_type),
            "dt": self.dt,
            "dx": self.dx,
            "r": self.r,
            "save_interval": self.checkpoint_interval,
            "time_steps": self.time_steps,
            "iter": iter,
        }
        return extra_metadata, pytree

    @staticmethod
    def _extract_base_checkpoint_name(dir_name: str) -> str:
        pattern = re.compile(r"^(?P<base>.+?)_(\(\d+\))$")
        match = pattern.match(dir_name)
        return match.group("base") if match else dir_name

    @staticmethod
    def _find_matching_checkpoint_dirs(parent_dir: Path, base_name: str) -> List[Path]:
        dir_pattern = re.compile(rf"^{re.escape(base_name)}(_\(\d+\))?$")
        return [
            entry.resolve(strict=True)
            for entry in parent_dir.iterdir()
            if entry.is_dir() and dir_pattern.match(entry.name)
        ]

    @staticmethod
    def _extract_checkpoint_number(path: Path) -> int:
        pattern = re.compile(r"^(?P<base>.+?)_(\(\d+\))$")
        match = pattern.match(path.name)
        if match:
            return int(match.group(2).strip("()"))
        return 0

    @staticmethod
    def load_all_paths(checkpoints_dir: Path) -> List[Path]:
        if not checkpoints_dir.exists() or not checkpoints_dir.is_dir():
            raise ValueError(f"Checkpoint directory {checkpoints_dir} does not exist.")

        base_name = ShardedSolver._extract_base_checkpoint_name(checkpoints_dir.name)
        checkpoint_paths = ShardedSolver._find_matching_checkpoint_dirs(
            checkpoints_dir.parent, base_name
        )
        checkpoint_paths.sort(key=ShardedSolver._extract_checkpoint_number)
        return checkpoint_paths

    @staticmethod
    def load_from_last_checkpoint(
        checkpoints_dir: Path, device_layout: DeviceLayout | None = None
    ) -> ShardedSolver:
        all_checkpoint_dirs = ShardedSolver.load_all_paths(checkpoints_dir)
        next_checkpoint_num = len(all_checkpoint_dirs)
        base_name = ShardedSolver._extract_base_checkpoint_name(
            all_checkpoint_dirs[0].name
        )

        last_checkpoint_dir = all_checkpoint_dirs[-1]

        # check if last_checkpoint_dir is empty as a smoke test
        if not any(last_checkpoint_dir.iterdir()):
            raise RuntimeError(
                f"Last checkpoint directory {last_checkpoint_dir} is empty."
            )

        new_checkpoint_dir = (
            last_checkpoint_dir.parent / f"{base_name}_({next_checkpoint_num})"
        )
        logger.debug(
            f"Found last checkpoint: {last_checkpoint_dir}, new checkpoint: {new_checkpoint_dir}"
        )
        registry = ocp.handlers.DefaultCheckpointHandlerRegistry()
        registry.add("state", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
        registry.add("extra_metadata", ocp.args.JsonRestore, ocp.JsonCheckpointHandler)

        if device_layout is None:
            device_layout = DeviceLayout(x_devices=1, y_devices=1)

        # First pass: load metadata from the latest checkpoint
        with ocp.CheckpointManager(
            last_checkpoint_dir,
            handler_registry=registry,
        ) as mngr:
            latest_step = mngr.latest_step()
            ch = mngr.restore(
                latest_step,
                args=ocp.args.Composite(
                    extra_metadata=ocp.args.JsonRestore(),
                ),
            )
            metadata = ch.extra_metadata

        solver = ShardedSolver(
            K=metadata["K"],
            M=metadata["M"],
            L=metadata["L"],
            T=metadata["T"],
            D=metadata["D"],
            f_type=RHSType(metadata["f_type"]),
            c=metadata["c"],
            dt=metadata["dt"],
            device_layout=device_layout,
            checkpoint_interval=metadata["save_interval"],
            output_dir=new_checkpoint_dir,
        )

        pytree = {"U": solver.U, "X": solver.x}

        abstract_pytree = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, pytree
        )

        # Second pass: load state from the latest checkpoint
        with ocp.CheckpointManager(
            last_checkpoint_dir,
            handler_registry=registry,
        ) as mngr:
            ch = mngr.restore(
                mngr.latest_step(),
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_pytree)
                ),
            )
            solver.U = jax.device_put(ch.state["U"], P(solver.mesh, "X", "Y"))
            solver.x = ch.state["X"]
            solver.iter = metadata["iter"]

        return solver

    def solve(self):
        ocp_options = ocp.CheckpointManagerOptions(
            max_to_keep=self.max_to_keep,
            save_interval_steps=self.checkpoint_interval,
        )

        starting_step = self.iter
        with ocp.CheckpointManager(
            self.checkpoints_dir.resolve(True),
            options=ocp_options,
        ) as mngr:
            for step in range(starting_step, self.time_steps):
                self.iter = step
                extra_metadata, pytree = self.prepare_checkpoint(step)

                if jnp.any(jnp.isnan(self.U)):
                    logger.warning("There are NaNs in self.U!")

                mngr.save(
                    step,
                    args=ocp.args.Composite(
                        state=ocp.args.StandardSave(pytree),
                        extra_metadata=ocp.args.JsonSave(extra_metadata),
                    ),
                )

                logger.info(f"Starting iteration {step+1}/{self.time_steps}")

                status, solution = self.step(
                    self.U, self.K, self.r, self.c, self.dt, self.f, tol=1e-6
                )

                if status != optx.RESULTS.successful:
                    logger.error(
                        f"Solver failed at iteration {step+1} with result: {status}"
                    )
                    break
                self.U = solution

        completion_marker = self.checkpoints_dir / ".SIMULATION_COMPLETED"
        completion_marker.touch(exist_ok=True)
        logger.info(
            f"Simulation completed successfully. Completion marker created at {completion_marker}"
        )


def main():
    args = CLI().parse_args()
    device_layout = DeviceLayout(
        x_devices=args.device_layout_x,
        y_devices=args.device_layout_y,
    )
    if hasattr(args, "CHECKPOINT_PATH"):
        logger.info("Loading from checkpoint...")
        solver = ShardedSolver.load_from_last_checkpoint(
            args.CHECKPOINT_PATH, device_layout
        )
        logger.info(f"Loaded checkpoint from {args.CHECKPOINT_PATH}")
    else:
        solver = ShardedSolver(
            K=args.K,
            M=args.M,
            L=args.L,
            T=args.T,
            D=args.D,
            f_type=args.f,
            c=args.c,
            dt=args.dt,
            device_layout=device_layout,
            checkpoint_interval=args.checkpoint_interval,
            output_dir=args.checkpoints_dir,
        )

    solver.solve()


if __name__ == "__main__":
    main()
