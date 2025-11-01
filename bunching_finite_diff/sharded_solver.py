import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import lineax
import optimistix as optx
import orbax.checkpoint as ocp
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


class CLI(Tap):
    with_mesh: List[int]  # How the devices should be laid-out in a mesh
    K: int = 20  # Number of steps in the system
    M: int = 100  # Number of spatial grid points
    L: float = 5.0  # Length of the spatial domain
    T: float = 10.0  # Final integration time
    D: float = 1e-1  # Diffusion coefficient (equiv. to line tension)
    f: RHSType = RHSType.MM2_a50_r1_jax  # Nonlinear coupling function
    c: float = 1.0  # Initial spacing constant (u_i(0,x) = i * c)
    dt: float = 1e-4  # Time step
    tol: float = 1e-6  # Tolerance for the solver
    checkpoint_interval: int = 50  # Save a plot every this many timesteps
    checkpoints_dir: Path = Path(
        "checkpoint_ocp"
    )  # Directory to save Orbax checkpoints


args = CLI().parse_args()
if len(args.with_mesh) != 2:
    raise ValueError("with_mesh must have exactly two dimensions for X and Y.")
mesh = jax.make_mesh(axis_shapes=(tuple(args.with_mesh)), axis_names=("X", "Y"))


def P(*args):
    return jax.NamedSharding(mesh, jax.sharding.PartitionSpec(*args))


@partial(
    jax.jit, static_argnames=["f", "K", "M", "r", "c", "dt"], out_shardings=P("X", "Y")
)
def _build_residual(U, U_prev, K, M, r, c, dt, f):
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


@partial(jax.jit, static_argnames=["f", "K", "M", "r", "c", "dt"])
def _step(U_prev, K, M, r, c, dt, f, tol):
    solver = optx.Newton(
        atol=tol,
        rtol=tol,
        linear_solver=lineax.GMRES(atol=tol, rtol=tol),
        cauchy_termination=False,
    )

    result = optx.root_find(
        lambda x, _: _build_residual(x, U_prev, K, M, r, c, dt, f),
        solver,
        U_prev,
        throw=False,
    )
    # hint to JAX to only shard the solution value
    jax.lax.with_sharding_constraint(result.value, P("X", "Y"))
    return result.result, result.value


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
        self.number_format_width = (
            int(jnp.log10(self.time_steps)) + 2 if self.time_steps > 0 else 2
        )
        self.iter = 0

        self.U = jax.device_put(self.U, P("X", "Y"))
        self.f = jax.jit(f_type.into_rhs(), out_shardings=P("X", "Y"))

        self.make_dirs()

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

    def solve(self):
        ocp_options = ocp.CheckpointManagerOptions(
            max_to_keep=self.max_to_keep,
            save_interval_steps=self.checkpoint_interval,
        )

        with ocp.CheckpointManager(
            self.checkpoints_dir.resolve(True),
            options=ocp_options,
        ) as mngr:
            for step in range(self.time_steps):
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

                status, solution = _step(
                    self.U, self.K, self.M, self.r, self.c, self.dt, self.f, tol=1e-6
                )

                if status != optx.RESULTS.successful:
                    logger.error(
                        f"Solver failed at iteration {step+1} with result: {status}"
                    )
                    break

                self.U = solution


def main():
    solver = ShardedSolver(
        K=args.K,
        M=args.M,
        L=args.L,
        T=args.T,
        D=args.D,
        f_type=args.f,
        c=args.c,
        dt=args.dt,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.checkpoints_dir,
    )
    solver.solve()


if __name__ == "__main__":
    main()
