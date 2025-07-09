import enum

from numba import njit


@njit
def MM2_a50_r1(a, b, c):
    f = (
        -50 * ((-a + b) ** -1)
        + (-b + c) ** -1
        + 1 * ((-a + b) ** -3 - ((-b + c) ** -3))
    )
    return f


class RHSType(enum.StrEnum):
    MM2_a50_r1 = "MM2_a50_r1"

    def into_rhs(self):
        if self == RHSType.MM2_a50_r1:
            return MM2_a50_r1
        raise ValueError(f"Unknown RHS type: {self}")
