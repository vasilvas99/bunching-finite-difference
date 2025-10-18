import os
from functools import partial
from pathlib import Path

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=64 --xla_gpu_first_collective_call_warn_stuck_timeout_seconds=60"
)
import jax
import jax.numpy as jnp
import lineax
import optimistix as optx
import orbax.checkpoint as ocp

from libs.rhs import RHSType

jax.config.update("jax_enable_x64", True)

mesh = jax.make_mesh(axis_shapes=(8, 1), axis_names=("X", "Y"))


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


@partial(
    jax.jit, static_argnames=["f", "K", "M", "r", "c", "dt"], out_shardings=P("X", "Y")
)
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
        throw=False,
    )


# generate some mock data
K = 8 * 10
M = 8 * 3
r = 0.1
c = 1.0
dt = 0.01
U_prev = jnp.arange(0, K * M, dtype=jnp.float64).reshape((K, M))
U_prev = jax.device_put(U_prev, P("X", "Y"))
U_init = jnp.arange(2, K * M + 2, dtype=jnp.float64).reshape((K, M))
U_init = jax.device_put(U_init, P("X", "Y"))

mm2 = RHSType.MM2_a50_r1_jax
f_jit = jax.jit(mm2.into_rhs(), out_shardings=P("X", "Y"))

for i in range(10):
    print(f"Iter {i+1}")
    residual = _build_residual(U_init, U_prev, K, M, r, c, dt, f_jit)

print(jnp.sum(residual))


script_dir = Path(__file__).parent.resolve()
path = script_dir / "checkpoints_orbax"


# options = ocp.CheckpointManagerOptions()
# with ocp.CheckpointManager(
#   ocp.test_utils.erase_and_create_empty(path),
#   options=options,
# ) as mngr:
#     for i in range(10):
#         U_new = _step(U_prev, K, M, r, c, dt, f_jit, tol=1e-6)
#         pytree = {
#             'state': U_new,
#             'extra_params': {
#                 'K': K,
#                 'M': M,
#                 'r': r,
#                 'c': c,
#                 'dt': dt,
#             }
#         }
#         mngr.save(i, args=ocp.args.StandardSave(pytree))


U_new = _step(U_prev, K, M, r, c, dt, f_jit, tol=1e-6)
print(jnp.sum(U_new))
