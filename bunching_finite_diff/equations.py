import enum
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from numba import njit
from tap import Tap


@njit
def MM2_a50_r1(a, b, c):
    f = (
        -50 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f


class RHSType(enum.StrEnum):
    MM2_a50_r1 = "MM2_a50_r1"

    def into_rhs(self):
        if self == RHSType.MM2_a50_r1:
            return MM2_a50_r1
        raise ValueError(f"Unknown RHS type: {self}")


class CheckPointCLi(Tap):
    output_dir: Path = Path("images")  # Directory to save PNG frames
    checkpoints_dir: Path = Path("images/checkpoints")  # Directory to save checkpoints
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
    f: RHSType = RHSType.MM2_a50_r1  # Nonlinear coupling function
    c: float = 1.0  # Initial spacing constant (u_i(0,x) = i * c)
    dt: float = 1e-4  # Time step
    tol: float = 1e-6  # Tolerance for the solver
    image_save_interval: int = 50  # Save a plot every this many timesteps
    output_dir: Path = Path("images")  # Directory to save PNG frames
    checkpoints_dir: Path = Path("images/checkpoints")  # Directory to save checkpoints

    def configure(self):
        self.add_subparser(
            "checkpoint",
            CheckPointCLi,
            help="Load from a checkpoint file and continue solving the system.",
        )


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@njit(parallel=True)
def build_residual(U, U_prev, r, c, dt, f, K, M):
    P = K * c  # periodic boundary condition offset
    # P = np.max(U) - np.min(U)
    N = K * M
    F = np.zeros(N)

    for i in range(K):
        for j in range(M):
            jm = (j - 1) % M
            jp = (j + 1) % M
            im = (i - 1) % K
            ip = (i + 1) % K

            Uim = U[im, j] - P if (i - 1) != im else U[im, j]
            Uip = U[ip, j] + P if (i + 1) != ip else U[ip, j]

            idx = i * M + j
            lap = r * (U[i, jp] - 2 * U[i, j] + U[i, jm])
            nonl = dt * f(Uim, U[i, j], Uip)

            F[idx] = U[i, j] - U_prev[i, j] - lap - nonl

    return F


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
        save_interval=10,
        output_dir=Path("./images"),
        checkpoints_dir=Path("./images/checkpoints"),
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

        self.save_interval = save_interval
        self.output_dir = output_dir
        self.checkpoints_dir = checkpoints_dir

        self.make_dirs()

        # spatial grid
        self.x = np.linspace(0, L, M)
        self.U = np.zeros((K, M))
        for i in range(K):
            noise = np.random.normal(0, 0.05 * c, M)
            self.U[i, :] = i * c + noise

        self.time_steps = int(np.ceil(T / dt))
        self.iter = 0

    def make_dirs(self):
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise ValueError(f"Output directory {self.output_dir} is not a directory.")
        if self.output_dir.exists() and not self.checkpoints_dir.is_dir():
            raise ValueError(
                f"Checkpoints directory {self.checkpoints_dir} is not a directory."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def build_residual(self, U_flat):
        return build_residual(
            U_flat.reshape((self.K, self.M)),
            self.U,
            self.r,
            self.c,
            self.dt,
            self.f,
            self.K,
            self.M,
        )

    def plot_frame(self):
        if self.iter % self.save_interval != 0:
            return

        U = self.U
        self.save_checkpoint()
        step = self.iter

        U_norm = U - np.min(U)
        # U_norm = U
        U_max = 1.1 * np.max(U_norm)

        plt.figure()
        for i in range(self.K):
            plt.plot(self.x, U_norm[i, :], label=f"$u_{{" f"{i+1}" f"}}$")

        plt.xlabel("x")
        plt.ylabel("$u_i - min$")
        plt.ylim(0, U_max)
        plt.title(f"Time step {step}")

        filename = f"{self.output_dir}/frame_{step:05d}.png"
        plt.savefig(filename)
        plt.close()

    def save_checkpoint(self):
        # saves all parameters and the current state of U in a compressed format
        checkpoint = {
            "K": self.K,
            "M": self.M,
            "L": self.L,
            "T": self.T,
            "D": self.D,
            "f_type": self.f_type,
            "c": self.c,
            "dt": self.dt,
            "dx": self.dx,
            "r": self.r,
            "U": self.U.copy(),
            "save_interval": self.save_interval,
            "time_steps": self.time_steps,
            "iter": self.iter,
        }
        np.savez_compressed(
            self.checkpoints_dir / f"checkpoint_{self.iter}.npz",
            **checkpoint,
        )

    @staticmethod
    def load_checkpoint(
        filepath: Path,
        output_dir: Path = Path("images"),
        checkpoints_dir: Path = Path("images/checkpoints"),
    ):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file {filepath} does not exist.")

        data = np.load(filepath, allow_pickle=True)
        solver = CoupledHeatSolver(
            K=int(data["K"]),
            M=int(data["M"]),
            L=float(data["L"]),
            T=float(data["T"]),
            D=float(data["D"]),
            f_type=RHSType(data["f_type"].item()),
            c=float(data["c"]),
            dt=float(data["dt"]),
            save_interval=int(data["save_interval"]),
            output_dir=Path(output_dir),
            checkpoints_dir=Path(checkpoints_dir),
        )
        solver.iter = int(data["iter"])
        solver.U = data["U"]
        solver.time_steps = int(data["time_steps"])
        solver.dx = float(data["dx"])
        solver.r = float(data["r"])
        logger.info(
            f"Loaded checkpoint with {solver.K} equations and {solver.M} grid points."
        )
        return solver

    def solve(self, tol=1e-6):
        U_flat = self.U.flatten()
        for n in range(self.iter, self.time_steps):
            self.iter = n
            self.plot_frame()
            logger.info(f"Solving step {n + 1}/{self.time_steps}...")
            U_flat = opt.newton_krylov(
                self.build_residual,
                U_flat,
                f_tol=tol,
            )

            self.U = U_flat.reshape((self.K, self.M))

        return self.U


def main():
    cli = CLI().parse_args()
    print(cli)
    if hasattr(cli, "CHECKPOINT_FILE"):
        # Load from checkpoint
        solver = CoupledHeatSolver.load_checkpoint(
            cli.CHECKPOINT_FILE, cli.output_dir, cli.checkpoints_dir
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
            save_interval=cli.image_save_interval,
            output_dir=cli.output_dir,
            checkpoints_dir=cli.checkpoints_dir,
        )

    solver.solve(tol=cli.tol)
    logger.info("Done")


if __name__ == "__main__":
    main()
