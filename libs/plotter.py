from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Union, cast

import matplotlib.pyplot as plt
import numpy as np

from libs.checkpoints import Checkpoint

STOP_SENTINEL = object()


@dataclass
class QueuedCheckpoint:
    save_path: Path
    checkpoint: Checkpoint


type QueueItem = QueuedCheckpoint | STOP_SENTINEL


class Plotter(Process):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = Queue(maxsize=10)

    def __send_internal(self, item: QueueItem):
        self.queue.put(item)

    def send_frame(self, save_path: Path | str, checkpoint: Checkpoint):
        self.__send_internal(QueuedCheckpoint(save_path, checkpoint))

    def run(self):
        while True:
            item = cast(Union[QueuedCheckpoint, STOP_SENTINEL], self.queue.get())

            if item is STOP_SENTINEL:
                return

            Plotter.__create_plot(item.save_path, item.checkpoint)

    @staticmethod
    def __create_plot(save_path, checkpoint: Checkpoint):
        step = checkpoint.iter
        U = checkpoint.U
        K = checkpoint.K
        x = checkpoint.X

        U_norm = U - np.min(U)
        U_max = 1.1 * np.max(U_norm)

        plt.figure()
        for i in range(K):
            plt.plot(x, U_norm[i, :], label=f"$u_{{" f"{i+1}" f"}}$")

        plt.xlabel("x")
        plt.ylabel("$u_i - min$")
        plt.ylim(0, U_max)
        plt.title(f"Time step {step}")

        plt.savefig(save_path)
        plt.close()

    def stop(self):
        self.__send_internal(STOP_SENTINEL)
