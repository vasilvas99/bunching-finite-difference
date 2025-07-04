import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numba import njit

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
        K,
        M,
        L,
        T,
        D,
        f,
        c,
        dt,
        save_interval=10,
        output_dir="./",
    ):
        """
        K: number of equations (periodic in i)
        M: number of spatial grid points
        L: spatial domain length [0, L]
        T: final time
        D: diffusion coefficient
        f: nonlinear coupling f(u_{i-1}, u_i, u_{i+1})
        c: initial spacing constant (u_i(0,x) = i * c)
        dt: time step
        save_interval: save a plot every this many timesteps
        output_dir: directory to save PNG frames
        """
        if os.path.exists(output_dir) and not os.path.isdir(output_dir):
            raise ValueError(f"Output directory {output_dir} is not a directory.")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.K = K
        self.M = M
        self.L = L
        self.T = T
        self.D = D
        self.f = f
        self.c = c
        self.dt = dt
        self.dx = self.L / (M - 1)
        self.r = D * dt / self.dx**2
        self.save_interval = save_interval
        self.output_dir = output_dir
        # spatial grid
        self.x = np.linspace(0, L, M)
        self.U = np.zeros((K, M))
        for i in range(K):
            noise = np.random.normal(0, 0.05 * c, M)
            self.U[i, :] = i * c + noise
            # amplitude = 0.8 * self.c
            # phase_shift = (i * self.L / K) % (2 * np.pi)
            # self.U[i, :] = (
            #     i * self.c
            #     + amplitude * np.cos(2 * np.pi * self.x / self.L)
            #     + phase_shift
            # )

        self.time_steps = int(np.ceil(T / dt))
        self.iter = 0

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
        step = self.iter + 1
        
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

    def solve(self, tol=1e-6):
        U_flat = self.U.flatten()
        for n in range(self.time_steps):
            self.iter = n
            self.plot_frame()
            logger.info(f"Solving step {n + 1}/{self.time_steps}...")
            U_flat = opt.newton_krylov(
                self.build_residual,
                U_flat,
                f_tol=tol,
            )

            noise =  np.random.normal(0, 0.005 * self.c, U_flat.shape)
            
            U_flat = U_flat + noise
            self.U = U_flat.reshape((self.K, self.M))

        return self.U


if __name__ == "__main__":

    @njit
    def f2(a, b, c):
        v = c - b
        return v * np.exp(1-v)
    
    @njit
    def f3(a, b, c):
        f = 0.0
        if np.abs(c-b) < 0.1 or np.abs(b-a) < 0.1:
            f =  0.0
        f = -100*((-a + b) ** -1) + (-b + c) ** -1 + 1 * ((-a + b) ** -3 -((-b + c) ** -3))
        # print("f = ", f)
        return f
    
    @njit
    def f(a, b, c):
        return -((-a + b) ** -3) + (-a + b) ** -1 + (-b + c) ** -3 - ((-b + c) ** -1)

    solver = CoupledHeatSolver(
        K=20,
        M=100,
        L=5,
        T=10,
        D=1,
        f=f3,
        c=1,
        dt=1e-4,
        save_interval=50,
        output_dir="images",
    )
    U_final = solver.solve(tol=1e-6)
