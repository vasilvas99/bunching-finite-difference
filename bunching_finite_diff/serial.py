import logging
import os
import sys
from functools import partial
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import jax
import jax.numpy as jnp
import lineax
import numpy as onp
import optimistix as optx
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tap import Tap

from libs.checkpoints import Checkpoint
from libs.rhs import RHSType

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnames=["f", "K", "M", "r", "c", "dt"])
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

    return optx.root_find(
        lambda x, _: _build_residual(x, U_prev, K, M, r, c, dt, f),
        solver,
        U_prev,
    ).value


# generate some mock data
K = 8 * 10
M = 8 * 3
r = 0.1
c = 1.0
dt = 0.01
U_prev = jnp.arange(0, K * M, dtype=jnp.float64).reshape((K, M))
U_init = jnp.arange(2, K * M + 2, dtype=jnp.float64).reshape((K, M))

mm2 = RHSType.MM2_a50_r1_jax
f_jit = jax.jit(mm2.into_rhs())

for i in range(10):
    print(f"Iter {i+1}")
    residual = _build_residual(U_init, U_prev, K, M, r, c, dt, f_jit)

print(jnp.sum(residual))

# final = residual.reshape(K*M, )
# print("Final residual shape:", final.shape)
# print(f"Final device mesh {final.addressable_shards}")

# print("Running test solve")
U_new = _step(U_prev, K, M, r, c, dt, f_jit, tol=1e-6)
print(jnp.sum(U_new))
# print("New U shape:", U_new.shape)
# print(f"New U device mesh {U_new.addressable_shards}")
