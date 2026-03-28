from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from libs.rhs import RHSType

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


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

    @staticmethod
    def load_from_orbax(checkpoint_path: Path) -> Checkpoint:
        import jax

        logger.debug(f"Loading checkpoint from {checkpoint_path}")

        with open(checkpoint_path / "extra_metadata" / "metadata", "r") as f:
            metadata = json.load(f)

        K = metadata["K"]
        M = metadata["M"]
        L = metadata["L"]

        U = jnp.zeros((K, M))
        X = jnp.linspace(0, L, M)
        pytree = {"U": U, "X": X}

        abstract_pytree = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, pytree
        )

        checkpointer = ocp.StandardCheckpointer()
        state = checkpointer.restore(
            checkpoint_path / "state",
            abstract_pytree,
        )

        logger.debug(f"Loaded checkpoint from {checkpoint_path}")

        return Checkpoint(
            U=np.array(state["U"]),
            X=np.array(state["X"]),
            K=metadata["K"],
            M=metadata["M"],
            L=metadata["L"],
            c=metadata["c"],
            T=metadata["T"],
            D=metadata["D"],
            f_type=RHSType(metadata["f_type"]),
            dt=metadata["dt"],
            dx=metadata["dx"],
            r=metadata["r"],
            save_interval=metadata["save_interval"],
            time_steps=metadata["time_steps"],
            iter=metadata["iter"],
        )
