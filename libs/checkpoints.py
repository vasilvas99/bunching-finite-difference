from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from libs.rhs import RHSType


@dataclass
class Checkpoint:
    U: np.ndarray
    X: np.ndarray
    K: int
    M: int
    L: float
    c: float
    T: float
    D: float
    f_type: RHSType
    dt: float
    dx: float
    r: float
    save_interval: int
    time_steps: int
    iter: int

    @staticmethod
    def load_from_file(filepath: Path) -> Checkpoint:
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file {filepath} does not exist.")

        data = np.load(filepath, allow_pickle=True)
        U = data["U"]
        L = float(data["L"])
        K = int(data["K"])
        M = int(data["M"])
        c = float(data["c"])
        x = np.linspace(0, L, M)
        T = float(data["T"])
        D = float(data["D"])
        f_type = RHSType(data["f_type"].item())
        dt = float(data["dt"])
        dx = float(data["dx"])
        r = float(data["r"])
        save_interval = int(data["save_interval"])
        time_steps = int(data["time_steps"])
        iteration = int(data["iter"])

        return Checkpoint(
            U=U,
            X=x,
            K=K,
            M=M,
            L=L,
            c=c,
            T=T,
            D=D,
            f_type=f_type,
            dt=dt,
            dx=dx,
            r=r,
            save_interval=save_interval,
            time_steps=time_steps,
            iter=iteration,
        )

    def save_to_file(self, filepath: Path) -> None:
        checkpoint_data = {
            "U": self.U,
            "X": self.X,
            "K": self.K,
            "M": self.M,
            "L": self.L,
            "c": self.c,
            "T": self.T,
            "D": self.D,
            "f_type": self.f_type,
            "dt": self.dt,
            "dx": self.dx,
            "r": self.r,
            "save_interval": self.save_interval,
            "time_steps": self.time_steps,
            "iter": self.iter,
        }
        np.savez_compressed(filepath, **checkpoint_data)
