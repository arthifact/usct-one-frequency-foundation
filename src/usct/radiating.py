"""Radiating (impedance) boundary forward: an open medium, not a closed cavity.

Physics-fidelity foundation slice. The verified DtN forward solves the disk as a
HARD-WALLED cavity (`u = g` on the boundary), which has interior resonances at
the Bessel zeros -- a pathology a real open water tank does not have, and the
reason the lossless model needs an `eta` stabilizer.

This module is the physically faithful alternative: the transducer ring sits in
an OPEN medium, so outgoing waves radiate away. The hard wall is replaced by a
first-order absorbing / impedance (Robin) boundary condition with a transducer
drive `g`:

    du/dn - i k_b u = g        on the boundary,   k_b = omega / c_bath

The `-i k_b u` term is the first-order Sommerfeld radiation condition for the
`exp(-i omega t)` convention (matched impedance: outgoing waves leave without
reflection); `g` is the transducer drive (a velocity-like source). The natural
observable is then the BOUNDARY PRESSURE `u` itself -- exactly what a receiving
transducer reads. This is the velocity-in / pressure-out experiment, the physical
counterpart of the pressure-in / flux-out DtN map.

Because the radiation term makes the operator non-singular for every `omega`, the
cavity resonances are simply GONE, not damped (see `research/radiating_experiment`).

Weak form (multiply by `v`, integrate by parts, substitute the Robin condition):

    integral grad(u).grad(v) - integral k^2 u v - i k_b integral_bnd u v
        = integral_bnd g v

so the operator is  ``A_rad = A_volume - i k_b M_b_full``  with
``A_volume = K - omega^2 (1 + i eta) M(1/c^2)`` reused verbatim from
:func:`usct.operator.assemble_helmholtz`, and the load is ``M_b_full @ g``. All
degrees of freedom (including the boundary) are unknowns -- a full solve, no
Dirichlet elimination. ``A_rad`` stays complex-symmetric, so the adjoint reuses
the forward factorization (``trans="H"``), exactly as :mod:`usct.inversion`.

This is a NEW forward beside the verified DtN (AGENTS.md sec 4); it does not
modify any verified numerics.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.special import jv, jvp
from skfem import BilinearForm, FacetBasis, asm

from usct.domain import Domain
from usct.forcing import BoundaryForcing
from usct.inversion import _gradient_load_form, _stiffness_form
from usct.measurement import boundary_mass_matrix
from usct.medium import Medium
from usct.operator import assemble_helmholtz


@BilinearForm
def _boundary_mass_form(u, v, _):
    return u * v


def full_boundary_mass(domain: Domain) -> sparse.csc_matrix:
    """Assemble the boundary mass matrix on the *full* DOF space (zero interior)."""

    facet_basis = FacetBasis(
        domain.mesh, domain.basis.elem, facets=domain.mesh.boundary_facets()
    )
    return asm(_boundary_mass_form, facet_basis).tocsc()


@dataclass(frozen=True)
class RadiatingSolution:
    """Full complex field and the boundary-pressure observable, boundary-DOF order."""

    matrix: sparse.csc_matrix
    field: NDArray[np.complex128]  # (dofs, patterns)
    boundary_pressure: NDArray[np.complex128]  # (boundary_dofs, patterns)
    theta: NDArray[np.float64]  # (boundary_dofs,), matching boundary-DOF order
    k_boundary: float


@dataclass(frozen=True)
class RadiatingObjectiveResult:
    """One complex-L2 boundary-pressure objective with its adjoint gradient."""

    misfit: float
    data_misfit: float
    regularization: float
    gradient: NDArray[np.float64]  # dJ/dc, nodal
    predicted_pressure: NDArray[np.complex128]  # (boundary_dofs, patterns)


def _medium_from_speed(sound_speed: NDArray[np.float64]) -> Medium:
    return Medium(
        sound_speed=np.asarray(sound_speed, dtype=np.float64).copy(),
        description="radiating candidate sound speed",
    )


def analytic_boundary_pressure(
    mode: int, k_boundary: float, radius: float, theta: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Exact homogeneous-disk boundary pressure for drive ``g = exp(i m theta)``.

    Regular interior solution ``u = A_m J_m(k r) exp(i m theta)`` with the
    radiation condition ``du/dn - i k u = g`` at ``r = R`` gives
    ``A_m = 1 / (k [J_m'(kR) - i J_m(kR)])``. The denominator never vanishes for
    real ``kR`` (``J_m`` and ``J_m'`` share no zeros), so there is no resonance.
    """

    argument = k_boundary * radius
    amplitude = jv(mode, argument) / (
        k_boundary * (jvp(mode, argument) - 1j * jv(mode, argument))
    )
    return (amplitude * np.exp(1j * mode * np.asarray(theta))).astype(np.complex128)


def assemble_radiating_matrix(
    domain: Domain,
    medium: Medium,
    omega: float,
    loss: float,
    boundary_speed: float,
    boundary_mass_full: sparse.csc_matrix | None = None,
) -> tuple[sparse.csc_matrix, float]:
    """Return ``A_rad = A_volume - i k_b M_b`` and the boundary wavenumber ``k_b``."""

    volume = assemble_helmholtz(domain, medium, omega, loss)
    k_boundary = omega / boundary_speed
    if boundary_mass_full is None:
        boundary_mass_full = full_boundary_mass(domain)
    matrix = (volume.matrix - 1j * k_boundary * boundary_mass_full).tocsc()
    return matrix, float(k_boundary)


def _boundary_source(
    domain: Domain, forcing: BoundaryForcing, boundary_mass_full: sparse.csc_matrix
) -> NDArray[np.complex128]:
    """Assemble the load ``integral_bnd g v = M_b_full @ g`` (g on boundary DOFs)."""

    g_full = np.zeros((domain.basis.N, forcing.values.shape[1]), dtype=np.complex128)
    g_full[domain.boundary_dofs] = forcing.values
    return np.asarray(boundary_mass_full @ g_full, dtype=np.complex128)


def solve_radiating(
    domain: Domain,
    sound_speed: NDArray[np.float64],
    forcing: BoundaryForcing,
    omega: float,
    loss: float,
    boundary_speed: float,
) -> RadiatingSolution:
    """Solve the radiating-boundary forward problem for all drive patterns."""

    boundary_mass_full = full_boundary_mass(domain)
    matrix, k_boundary = assemble_radiating_matrix(
        domain, _medium_from_speed(sound_speed), omega, loss, boundary_speed, boundary_mass_full
    )
    rhs = _boundary_source(domain, forcing, boundary_mass_full)
    field = np.asarray(splu(matrix).solve(rhs), dtype=np.complex128)
    if field.ndim == 1:
        field = field[:, None]
    boundary = domain.boundary_dofs
    return RadiatingSolution(
        matrix=matrix,
        field=field,
        boundary_pressure=field[boundary].copy(),
        theta=domain.boundary_theta.copy(),
        k_boundary=k_boundary,
    )


def forward_boundary_pressure(
    domain: Domain,
    sound_speed: NDArray[np.float64],
    forcing: BoundaryForcing,
    omega: float,
    loss: float,
    boundary_speed: float,
) -> NDArray[np.complex128]:
    """Predicted boundary pressure ``(boundary_dofs, patterns)``, boundary-DOF order."""

    solution = solve_radiating(domain, sound_speed, forcing, omega, loss, boundary_speed)
    return solution.boundary_pressure


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
