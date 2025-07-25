import logging
import os
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import lineax
import numpy as onp
import optimistix as optx
from tap import Tap

from libs.checkpoints import Checkpoint
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


class CheckPointCLi(Tap):
    checkpoints_dir: Path = Path("checkpoints")  # Directory to save checkpoints
    tol: float = 1e-6  # Tolerance for the solver

    def configure(self):
        self.add_argument(
            "CHECKPOINT_FILE", type=Path, help="Path to the checkpoint file."
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
    checkpoint_interval: int = 50  # Save a plot every this many timesteps
    checkpoints_dir: Path = Path("checkpoints")  # Directory to save checkpoints

    def configure(self):
        self.add_subparser(
            "checkpoint",
            CheckPointCLi,
            help="Load from a checkpoint file and continue solving the system.",
        )


@partial(jax.jit, static_argnames=["f", "K", "M", "r", "c", "dt"])
def _build_residual(U, U_prev, K, M, r, c, dt, f):
    P = K * c
    U = U.reshape((K, M))
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
    return jnp.reshape(F, (K * M,))


@partial(jax.jit, static_argnames=["f", "K", "M", "r", "c", "dt"])
def _step(U_flat_prev, K, M, r, c, dt, f, tol):
    solver = optx.Newton(
        atol=tol,
        rtol=tol,
        linear_solver=lineax.GMRES(atol=tol, rtol=tol),
        cauchy_termination=False,
    )

    return optx.root_find(
        lambda x, _: _build_residual(
            x, U_flat_prev.reshape((K, M)), K, M, r, c, dt, f
        ).flatten(),
        solver,
        U_flat_prev,
    ).value


class CoupledHeatSolver:
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
        output_dir: Path | str = Path("./checkpoints"),
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
        save_interval: save a plot every this many timesteps
        output_dir: directory to save PNG frames
        """
        logger.info("Initializing JAX-based solver.")
        self.K = K
        self.M = M
        self.L = L
        self.T = T
        self.D = D
        self.f_type = f_type
        self.f = f_type.into_rhs()
        self.c = c
        self.dt = dt
        self.iter = 0
        self.dx = self.L / (M - 1)
        self.r = D * dt / self.dx**2

        self.checkpoint_interval = checkpoint_interval
        self.checkpoints_dir = Path(output_dir)

        self.make_dirs()

        # spatial grid
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

    def make_dirs(self):
        if self.checkpoints_dir.exists() and not self.checkpoints_dir.is_dir():
            raise ValueError(
                f"Output directory {self.checkpoints_dir} exists and is not a directory."
            )

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self) -> Checkpoint:
        ch = Checkpoint(
            U=onp.array(self.U),
            X=onp.array(self.x),
            K=self.K,
            M=self.M,
            L=self.L,
            T=self.T,
            D=self.D,
            f_type=self.f_type,
            c=self.c,
            dt=self.dt,
            dx=self.dx,
            r=self.r,
            save_interval=self.checkpoint_interval,
            time_steps=self.time_steps,
            iter=self.iter,
        )
        ch.save_to_file(
            self.checkpoints_dir
            / f"checkpoint_{self.iter:0{self.number_format_width}}.npz"
        )
        return ch

    @staticmethod
    def load_checkpoint(
        filepath: Path,
        output_dir: Path = Path("images"),
    ):
        ch = Checkpoint.load_from_file(filepath)

        solver = CoupledHeatSolver(
            K=ch.K,
            M=ch.M,
            L=ch.L,
            T=ch.T,
            D=ch.D,
            f_type=ch.f_type,
            c=ch.c,
            dt=ch.dt,
            checkpoint_interval=ch.save_interval,
            output_dir=Path(output_dir),
        )
        solver.iter = ch.iter
        solver.U = onp.array(ch.U)
        solver.time_steps = ch.time_steps
        solver.dx = onp.array(ch.dx)
        solver.r = ch.r
        logger.info(
            f"Loaded checkpoint with {solver.K} equations and {solver.M} grid points."
        )
        return solver

    def get_solution_error(self, U_flat_new, U_flat_old):
        return jnp.linalg.norm(
            _build_residual(
                U_flat_new.reshape((self.K, self.M)),
                U_flat_old.reshape((self.K, self.M)),
                self.K,
                self.M,
                self.r,
                self.c,
                self.dt,
                self.f,
            )
        )

    def solve(self, tol=1e-6):
        for n in range(self.iter, self.time_steps):
            self.iter = n
            if self.iter % self.checkpoint_interval == 0:
                self.save_checkpoint()

            logger.info(f"Solving step {n + 1}/{self.time_steps}...")

            U_flat_old = self.U.flatten()
            U_flat_new = _step(
                U_flat_old, self.K, self.M, self.r, self.c, self.dt, self.f, tol
            )

            logger.debug(
                f"Residual: {self.get_solution_error(U_flat_new, U_flat_old):.6e}"
            )
            self.U = U_flat_new.reshape((self.K, self.M))
        return self.U


def main():
    cli = CLI().parse_args()
    logger.info(cli)
    if hasattr(cli, "CHECKPOINT_FILE"):
        solver = CoupledHeatSolver.load_checkpoint(
            cli.CHECKPOINT_FILE, cli.checkpoints_dir
        )
        logger.info(f"Loaded checkpoint from {cli.CHECKPOINT_FILE}")
    else:
        solver = CoupledHeatSolver(
            K=cli.K,
            M=cli.M,
            L=cli.L,
            T=cli.T,
            D=cli.D,
            f_type=cli.f,
            c=cli.c,
            dt=cli.dt,
            checkpoint_interval=cli.checkpoint_interval,
            output_dir=cli.checkpoints_dir,
        )

    solver.solve(tol=cli.tol)
    logger.info("Done")


if __name__ == "__main__":
    main()
