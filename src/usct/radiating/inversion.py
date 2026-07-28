"""Adjoint-state gradient for the radiating-boundary sim (velocity-in/pressure-out).

The inversion half of sim 2. The complex-L2 misfit is on the BOUNDARY PRESSURE
observable, and the adjoint reuses the forward factorization (`trans="H"`) exactly
as the DtN inversion (`usct.dtn.inversion`) does for its flux observable. Shares
the adjoint load form with the DtN sim via `usct.physics.operator`.

Verified against central finite differences in `tests/test_radiating.py`.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu
from skfem import asm

from usct.physics.boundary import boundary_mass_matrix
from usct.physics.domain import Domain
from usct.physics.forcing import BoundaryForcing
from usct.physics.operator import _gradient_load_form, _stiffness_form
from usct.radiating.forward import (
    _boundary_source,
    _medium_from_speed,
    assemble_radiating_matrix,
    full_boundary_mass,
)


@dataclass(frozen=True)
class RadiatingObjectiveResult:
    """One complex-L2 boundary-pressure objective with its adjoint gradient."""

    misfit: float
    data_misfit: float
    regularization: float
    gradient: NDArray[np.float64]  # dJ/dc, nodal
    predicted_pressure: NDArray[np.complex128]  # (boundary_dofs, patterns)


def radiating_objective_and_gradient(
    domain: Domain,
    sound_speed: NDArray[np.float64],
    forcing: BoundaryForcing,
    omega: float,
    loss: float,
    boundary_speed: float,
    observed_pressure: NDArray[np.complex128],
    *,
    alpha: float = 0.0,
    background_speed: float | None = None,
    boundary_mass_full: sparse.csc_matrix | None = None,
    boundary_mass: sparse.csc_matrix | None = None,
    stiffness: sparse.csc_matrix | None = None,
) -> RadiatingObjectiveResult:
    """Complex-L2 boundary-pressure misfit and its adjoint-state gradient wrt ``c``.

    The source and boundary term depend only on the known bath ``k_b``, not on the
    interior sound speed, so ``dA/dm = -omega^2 (1 + i eta) T^{(j)}`` and the
    gradient assembly is identical to the DtN adjoint -- only the observable
    (boundary pressure) and its adjoint source differ. Verified against central
    finite differences in ``tests/test_radiating.py``.
    """

    speed = np.asarray(sound_speed, dtype=np.float64)
    boundary = domain.boundary_dofs
    pattern_count = forcing.values.shape[1]
    if observed_pressure.shape != (boundary.size, pattern_count):
        raise ValueError(
            f"observed_pressure must have shape {(boundary.size, pattern_count)}; "
            f"got {observed_pressure.shape}"
        )

    if boundary_mass_full is None:
        boundary_mass_full = full_boundary_mass(domain)
    matrix, _ = assemble_radiating_matrix(
        domain, _medium_from_speed(speed), omega, loss, boundary_speed, boundary_mass_full
    )
    factor = splu(matrix)

    rhs = _boundary_source(domain, forcing, boundary_mass_full)
    field = np.asarray(factor.solve(rhs), dtype=np.complex128)
    if field.ndim == 1:
        field = field[:, None]
    predicted = field[boundary]

    residual = predicted - observed_pressure
    if boundary_mass is None:
        boundary_mass = boundary_mass_matrix(domain)
    boundary_mass_complex = boundary_mass.astype(np.complex128)
    weighted_residual = boundary_mass_complex @ residual
    data_misfit = 0.5 * float(np.real(np.sum(np.conj(residual) * weighted_residual)))

    # Adjoint: A^H lambda = R^T M_b r  (source scattered onto boundary DOFs only).
    adjoint_source = np.zeros((domain.basis.N, pattern_count), dtype=np.complex128)
    adjoint_source[boundary] = weighted_residual
    adjoint = np.asarray(factor.solve(adjoint_source, trans="H"), dtype=np.complex128)
    if adjoint.ndim == 1:
        adjoint = adjoint[:, None]

    load = np.zeros(domain.basis.N, dtype=np.complex128)
    for pattern in range(pattern_count):
        lam_conj = domain.basis.interpolate(np.conj(adjoint[:, pattern]))
        u_fwd = domain.basis.interpolate(field[:, pattern])
        load += np.asarray(
            asm(_gradient_load_form, domain.basis, lam_conj=lam_conj, u_fwd=u_fwd),
            dtype=np.complex128,
        )
    gradient_m = (omega**2) * np.real((1.0 + 1j * loss) * load)
    gradient = (-2.0 / speed**3) * gradient_m

    regularization = 0.0
    if alpha > 0.0:
        if stiffness is None:
            stiffness = asm(_stiffness_form, domain.basis).tocsc()
        stiffness_c = np.asarray(stiffness @ speed, dtype=np.float64)
        regularization = 0.5 * alpha * float(speed @ stiffness_c)
        gradient = gradient + alpha * stiffness_c

    if background_speed is not None:
        gradient[boundary] = 0.0

    return RadiatingObjectiveResult(
        misfit=data_misfit + regularization,
        data_misfit=data_misfit,
        regularization=regularization,
        gradient=gradient,
        predicted_pressure=predicted,
    )
