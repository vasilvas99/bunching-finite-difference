import logging
import os

import matplotlib.pyplot as plt
import matspy
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class CoupledHeatSolver:
    def __init__(
        self,
        K,
        M,
        L,
        T,
        D,
        f,
        df_u_prev,
        df_u,
        df_u_next,
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
        df_u_prev, df_u, df_u_next: partial derivatives of f wrt u_{i-1}, u_i, u_{i+1}
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
        self.df_u_prev = df_u_prev
        self.df_u = df_u
        self.df_u_next = df_u_next
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
            # noise = np.random.normal(0, 0.05 * c, M)
            # self.U[i, :] = i * c + noise
            amplitude = 0.8 * self.c
            phase_shift = (i * self.L / K) % (2 * np.pi)
            self.U[i, :] = i * self.c + amplitude * np.cos(2 * np.pi * self.x / self.L) + phase_shift

        self.time_steps = int(np.ceil(T / dt))

    def index(self, i, j):
        return i * self.M + j

    def build_residual_and_jacobian(self, U_flat, build_jacobian=True):
        K, M = self.K, self.M
        dt, r = self.dt, self.r
        f = self.f
        df_u_prev = self.df_u_prev
        df_u = self.df_u
        df_u_next = self.df_u_next

        U = U_flat.reshape((K, M))
        U_prev = self.U
        P = K * self.c  # periodic boundary condition offset
        # P = np.max(U) - np.min(U)
        N = K * M
        F = np.zeros(N)
        J = sp.lil_matrix((N, N))

        for i in range(K):
            for j in range(M):
                jm = (j - 1) % M
                jp = (j + 1) % M
                im = (i - 1) % K
                ip = (i + 1) % K

                Uim = U[im, j] - P if (i - 1) != im else U[im, j]
                Uip = U[ip, j] + P if (i + 1) != ip else U[ip, j]

                idx = self.index(i, j)
                lap = r * (U[i, jp] - 2 * U[i, j] + U[i, jm])
                nonl = dt * f(Uim, U[i, j], Uip)
                F[idx] = U[i, j] - U_prev[i, j] - lap - nonl

                if not build_jacobian:
                    continue

                J[idx, idx] += 1 + 2 * r - dt * df_u(Uim, U[i, j], Uip)
                J[idx, self.index(i, jm)] += -r
                J[idx, self.index(i, jp)] += -r
                J[idx, self.index(im, j)] += -dt * df_u_prev(Uim, U[i, j], Uip)
                J[idx, self.index(ip, j)] += -dt * df_u_next(Uim, U[i, j], Uip)

        if not build_jacobian:
            return F, None

        return F, J.tocsr()

    def plot_frame(self, U, step):
        # normalize by subtracting the minimum value over i and x
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
        # plt.legend()
        filename = f"{self.output_dir}/frame_{step:05d}.png"
        plt.savefig(filename)
        plt.close()

    def solve(self, tol=1e-6, maxiter=5000, reinit_jacobian_steps=10):
        U_flat = self.U.flatten()
        for n in range(self.time_steps):
            logger.info(f"Solving step {n + 1}/{self.time_steps}...")
            for k in range(maxiter):
                if k % reinit_jacobian_steps == 0:
                    F, J = self.build_residual_and_jacobian(U_flat)
                else:
                    F, _ = self.build_residual_and_jacobian(
                        U_flat, build_jacobian=False
                    )

                logger.debug(f"Resd: {np.linalg.norm(F, np.inf)} at step {n}, iter {k}")
                if np.linalg.norm(F, np.inf) < tol:
                    break

                delta = spla.spsolve(J, -F)
                U_flat += delta
            else:
                logger.warning(f"Newton did not converge at step {n}")

            self.U = U_flat.reshape((self.K, self.M))

            # save plot every save_interval steps
            if n % self.save_interval == 0:
                self.plot_frame(self.U, n)
                # matspy.spy(J)

        return self.U

    def solve_with_scipy(self):
        U_flat = self.U.flatten()
        
        for n in range(self.time_steps):
            # save plot every save_interval steps
            if n % self.save_interval == 0:
                self.plot_frame(self.U, n)
            
            logger.info(f"Solving step {n + 1}/{self.time_steps}...")
            U_flat = opt.newton_krylov(
                lambda u: self.build_residual_and_jacobian(u, build_jacobian=False)[0],
                U_flat,
                f_tol=1e-6,
            )
            self.U = U_flat.reshape((self.K, self.M))

        return self.U


if __name__ == "__main__":

    def f(a, b, c):
        v = c - b
        return v * np.exp(1 - v)

    def df_prev(a, b, c):
        return 0.0

    def df_b(a, b, c):
        v = c - b
        return -np.exp(1 - v) + v * np.exp(1 - v)

    def df_next(a, b, c):
        v = c - b
        return np.exp(1 - v) - v * np.exp(1 - v)

    solver = CoupledHeatSolver(
        K=5,
        M=1000,
        L=50,
        T=5,
        D=1e-4,
        f=f,
        df_u_prev=df_prev,
        df_u=df_b,
        df_u_next=df_next,
        c=1,
        dt=1e-4,
        save_interval=1,
        output_dir="images",
    )
    U_final = solver.solve_with_scipy()
