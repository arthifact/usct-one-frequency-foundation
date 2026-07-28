"""Named complex boundary-pressure patterns."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from usct.physics.domain import Domain

_FOURIER_PATTERN = re.compile(r"(cos|sin|exp):([0-9]+)\Z")


@dataclass(frozen=True)
class BoundaryForcing:
    """Boundary pressure columns, shape ``(boundary_dofs, patterns)``."""

    names: tuple[str, ...]
    values: NDArray[np.complex128]


def parse_pattern_name(name: str) -> tuple[str, int]:
    """Parse ``constant`` or an exact ``cos:m``, ``sin:m``, ``exp:m`` name."""

    if name == "constant":
        return "constant", 0
    match = _FOURIER_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(
            f"unsupported boundary pattern {name!r}; expected constant, cos:m, sin:m, or exp:m"
        )
    return match.group(1), int(match.group(2))


def _evaluate_pattern(kind: str, mode: int, theta: NDArray[np.float64]):
    if kind == "constant":
        return np.ones_like(theta)
    if kind == "cos":
        return np.cos(mode * theta)
    if kind == "sin":
        return np.sin(mode * theta)
    return np.exp(1j * mode * theta)


def build_boundary_forcing(
    domain: Domain,
    pattern_names: Sequence[str],
) -> BoundaryForcing:
    """Evaluate explicitly selected boundary pressure patterns in column order."""

    names = tuple(pattern_names)
    if not names:
        raise ValueError("at least one boundary forcing pattern is required")
    if len(set(names)) != len(names):
        raise ValueError("boundary forcing pattern names must not be repeated")
    parsed = [parse_pattern_name(name) for name in names]
    columns = [_evaluate_pattern(kind, mode, domain.boundary_theta) for kind, mode in parsed]
    values = np.asarray(np.column_stack(columns), dtype=np.complex128)
    return BoundaryForcing(names=names, values=values)
