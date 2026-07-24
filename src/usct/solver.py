"""Explicit Dirichlet elimination and batched sparse direct solution."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import splu

from usct.domain import Domain
from usct.forcing import BoundaryForcing
from usct.operator import HelmholtzOperator


@dataclass(frozen=True)
class SolveReport:
    """Diagnostics for one factorization and a multi-right-hand-side solve."""

    degrees_of_freedom: int
    interior_degrees_of_freedom: int
    boundary_degrees_of_freedom: int
    right_hand_sides: int
    factorizations: int
    factorization_time_seconds: float
    solve_time_seconds: float
    normalized_interior_residuals: NDArray[np.float64]
    maximum_boundary_value_error: float


@dataclass(frozen=True)
class FieldSolution:
    """Full complex pressure field, shape ``(dofs, forcing_patterns)``."""

    field: NDArray[np.complex128]
    report: SolveReport


def normalize_volume_rhs(
    volume_rhs: NDArray[np.complex128] | None,
    degrees_of_freedom: int,
    right_hand_sides: int,
) -> NDArray[np.complex128]:
    """Return an assembled complex volume RHS matrix with one column per pattern."""

    if volume_rhs is None:
        return np.zeros((degrees_of_freedom, right_hand_sides), dtype=np.complex128)
    rhs = np.asarray(volume_rhs, dtype=np.complex128)
    if rhs.shape == (degrees_of_freedom,):
        return np.repeat(rhs[:, None], right_hand_sides, axis=1)
    expected = (degrees_of_freedom, right_hand_sides)
    if rhs.shape != expected:
        raise ValueError(f"volume RHS shape must be {(degrees_of_freedom,)} or {expected}")
    if not np.all(np.isfinite(rhs)):
        raise ValueError("volume RHS contains a non-finite value")
    return rhs.copy()


def solve_dirichlet(
    domain: Domain,
    operator: HelmholtzOperator,
    forcing: BoundaryForcing,
    volume_rhs: NDArray[np.complex128] | None = None,
) -> FieldSolution:
    """Solve for pressure with prescribed boundary values using one batched LU."""

    boundary = domain.boundary_dofs
    interior = domain.interior_dofs
    count = forcing.values.shape[1]
    expected = (boundary.size, count)
    if forcing.values.shape != expected:
        raise ValueError(f"forcing values must have shape {expected}")
    if operator.matrix.shape != (domain.basis.N, domain.basis.N):
        raise ValueError("operator shape does not match domain")
    q = normalize_volume_rhs(volume_rhs, domain.basis.N, count)
    matrix_ii = operator.matrix[interior][:, interior].tocsc()
    matrix_ib = operator.matrix[interior][:, boundary]
    rhs = np.asarray(q[interior] - matrix_ib @ forcing.values, dtype=np.complex128)

    factor_started = perf_counter()
    factor = splu(matrix_ii)
    factorization_time = perf_counter() - factor_started
    solve_started = perf_counter()
    interior_field = np.asarray(factor.solve(rhs), dtype=np.complex128)
    solve_time = perf_counter() - solve_started
    if interior_field.ndim == 1:
        interior_field = interior_field[:, None]

    field = np.empty((domain.basis.N, count), dtype=np.complex128)
    field[interior] = interior_field
    field[boundary] = forcing.values
    residual = matrix_ii @ interior_field - rhs
    denominator = np.maximum(
        np.linalg.norm(rhs, axis=0),
        np.finfo(np.float64).eps,
    )
    normalized_residuals = np.asarray(
        np.linalg.norm(residual, axis=0) / denominator,
        dtype=np.float64,
    )
    boundary_error = float(np.max(np.abs(field[boundary] - forcing.values)))
    report = SolveReport(
        degrees_of_freedom=int(domain.basis.N),
        interior_degrees_of_freedom=int(interior.size),
        boundary_degrees_of_freedom=int(boundary.size),
        right_hand_sides=int(count),
        factorizations=1,
        factorization_time_seconds=factorization_time,
        solve_time_seconds=solve_time,
        normalized_interior_residuals=normalized_residuals,
        maximum_boundary_value_error=boundary_error,
    )
    return FieldSolution(field=field, report=report)
