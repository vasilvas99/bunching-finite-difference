from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ImportedCheckpoint:
    U: np.ndarray
    X: np.ndarray
    K: int
    M: int
    L: float
    c: float

    @staticmethod
    def load_from_file(filepath: Path) -> ImportedCheckpoint:
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint file {filepath} does not exist.")

        data = np.load(filepath, allow_pickle=True)
        U = data["U"]
        L = float(data["L"])
        K = int(data["K"])
        M = int(data["M"])
        c = float(data["c"])
        x = np.linspace(0, L, M)

        return ImportedCheckpoint(U=U, X=x, K=K, M=M, L=L, c=c)
