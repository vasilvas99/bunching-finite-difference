"""
Minimal tests for the MM2 RHS functions.

These verify the core physical invariant: f(a, b, c) uses the back-terrace
convention Δu_n = b − a, Δu_{n+1} = c − b, and implements

    f = −κ_a * (Δu_n^{-p} − Δu_{n+1}^{-p}) + κ_r * (Δu_n^{-q} − Δu_{n+1}^{-q})

with p=1, q=3, κ_r=1.
"""

import jax
jax.config.update("jax_enable_x64", True)

import pytest
from libs.rhs import (
    MM2_a0p1_r1_jax,
    MM2_a1_r1_jax,
    MM2_a10_r1_jax,
    MM2_a50_r1_jax,
    MM2_a100_r1_jax,
    MM2_a200_r1_jax,
)

# (function, κ_a value)
VARIANTS = [
    (MM2_a0p1_r1_jax, 0.1),
    (MM2_a1_r1_jax, 1.0),
    (MM2_a10_r1_jax, 10.0),
    (MM2_a50_r1_jax, 50.0),
    (MM2_a100_r1_jax, 100.0),
    (MM2_a200_r1_jax, 200.0),
]


@pytest.mark.parametrize("f, ka", VARIANTS, ids=lambda v: str(v))
def test_equally_spaced_steps_give_zero_velocity(f, ka):
    """Equally spaced steps (Δu_n = Δu_{n+1}) must have zero velocity."""
    # a=0, b=1, c=2  →  Δu_n = 1, Δu_{n+1} = 1
    result = float(f(0.0, 1.0, 2.0))
    assert result == pytest.approx(0.0, abs=1e-12), (
        f"f(0,1,2) should be 0 for equal spacing, got {result}"
    )


@pytest.mark.parametrize("f, ka", VARIANTS, ids=lambda v: str(v))
def test_antisymmetry_under_terrace_swap(f, ka):
    """Swapping Δu_n and Δu_{n+1} should flip the sign of f.

    f(a, b, c) with Δu_n=d1, Δu_{n+1}=d2 should equal -f(a', b', c')
    where Δu_n=d2, Δu_{n+1}=d1.
    """
    # (a=0, b=1, c=3) → Δu_n=1, Δu_{n+1}=2
    v1 = float(f(0.0, 1.0, 3.0))
    # (a=0, b=2, c=3) → Δu_n=2, Δu_{n+1}=1
    v2 = float(f(0.0, 2.0, 3.0))
    assert v1 == pytest.approx(-v2, abs=1e-12), (
        f"f should be antisymmetric under Δu_n ↔ Δu_{{n+1}}, got {v1} vs {v2}"
    )


@pytest.mark.parametrize("f, ka", VARIANTS, ids=lambda v: str(v))
def test_matches_analytical_mm2_formula(f, ka):
    """Verify against the analytical MM2 formula with p=1, q=3, κ_r=1."""
    a, b, c = 0.0, 1.0, 4.0  # Δu_n=1, Δu_{n+1}=3
    delta_n = b - a
    delta_np1 = c - b

    expected = (
        -ka * (delta_n ** -1 - delta_np1 ** -1)
        + 1.0 * (delta_n ** -3 - delta_np1 ** -3)
    )
    result = float(f(a, b, c))
    assert result == pytest.approx(expected, abs=1e-12), (
        f"f({a},{b},{c}) = {result}, expected {expected}"
    )
