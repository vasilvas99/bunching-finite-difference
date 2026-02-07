import enum

import jax
from numba import njit


@njit
def MM2_a50_r1(a, b, c):
    f = (
        -50 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f

@jax.jit
def MM2_a0p1_r1_jax(a, b, c):
    f = (
        -0.1 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f

@jax.jit
def MM2_a1_r1_jax(a, b, c):
    f = (
        -1 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f

@jax.jit
def MM2_a10_r1_jax(a, b, c):
    f = (
        -10 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f

@jax.jit
def MM2_a50_r1_jax(a, b, c):
    f = (
        -50 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f

@jax.jit
def MM2_a100_r1_jax(a, b, c):
    f = (
        -100 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f

@jax.jit
def MM2_a200_r1_jax(a, b, c):
    f = (
        -200 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f


class RHSType(enum.StrEnum):
    MM2_a50_r1 = "MM2_a50_r1"
    MM2_a0p1_r1_jax = "MM2_a0p1_r1_jax"
    MM2_a1_r1_jax = "MM2_a1_r1_jax"
    MM2_a10_r1_jax = "MM2_a10_r1_jax"
    MM2_a50_r1_jax = "MM2_a50_r1_jax"
    MM2_a100_r1_jax = "MM2_a100_r1_jax"
    MM2_a200_r1_jax = "MM2_a200_r1_jax"

    def into_rhs(self):
        if self == RHSType.MM2_a50_r1:
            return MM2_a50_r1
        elif self == RHSType.MM2_a0p1_r1_jax:
            return MM2_a0p1_r1_jax
        elif self == RHSType.MM2_a1_r1_jax:
            return MM2_a1_r1_jax
        elif self == RHSType.MM2_a10_r1_jax:
            return MM2_a10_r1_jax
        elif self == RHSType.MM2_a50_r1_jax:
            return MM2_a50_r1_jax
        elif self == RHSType.MM2_a100_r1_jax:
            return MM2_a100_r1_jax
        elif self == RHSType.MM2_a200_r1_jax:
            return MM2_a200_r1_jax
        raise ValueError(f"Unknown RHS type: {self}")
