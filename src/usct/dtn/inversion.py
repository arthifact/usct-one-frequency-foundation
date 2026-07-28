"""Adjoint-state gradient and complex-L2 FWI objective for sound-speed inversion.

This is *research* code layered on top of the verified forward foundation. It
does not modify any verified numerics. It re-uses :func:`assemble_helmholtz`, the
same explicit interior/boundary Dirichlet elimination proven in
:mod:`usct.dtn.solver`, and the same weak Neumann trace as :mod:`usct.physics.boundary`,
and adds three things the foundation deliberately excluded:

1. the complex-L2 boundary-flux misfit in the natural ``L2(boundary)`` inner
   product (weighted by the boundary mass matrix ``M_b``);
2. its **adjoint-state gradient** with respect to nodal sound speed, computed by
   *re-using the single interior LU factorization* -- the forward solve and the
   adjoint solve share one ``splu`` object (the adjoint is the conjugate
   transpose, ``trans="H"``), so the gradient for every boundary pattern is
   nearly free once the forward field exists;
3. optional H1 (Tikhonov) smoothness regularization.

Derivation (discretely consistent with the verified forward + measurement)
--------------------------------------------------------------------------
The operator is ``A(m) = K - omega^2 (1 + i eta) M(m)`` with nodal squared
slowness ``m = 1/c^2`` entering ``M(m)`` linearly, so
``dA/dm_j = -omega^2 (1 + i eta) T^{(j)}`` where
``T^{(j)}_{ab} = integral phi_j phi_a phi_b``.

With Dirichlet data ``u_B = g`` fixed and interior solve
``A_II u_I = -A_IB g``, the observable is ``D = M_b^{-1} (A u)_B`` and the misfit
``J = 1/2 (D - D_obs)^H M_b (D - D_obs)``. The adjoint field ``lambda`` solves

    lambda_B = D - D_obs                         (boundary source is the residual)
    A_II^H lambda_I = -A_BI^H lambda_B           (re-uses the forward factor, trans="H")

and the gradient is

    dJ/dm_j = -omega^2 * Re[ (1 + i eta) * integral phi_j conj(lambda) u ]
    dJ/dc_j = (-2 / c_j^3) * dJ/dm_j .

The adjoint is verified against central finite differences and a second-order
Taylor test in ``tests/test_adjoint.py``. See ``AGENTS.md`` sections 8-9.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import SuperLU, splu
from skfem import asm

from usct.physics.boundary import boundary_mass_matrix
from usct.physics.domain import Domain
from usct.physics.forcing import BoundaryForcing
from usct.physics.medium import Medium
from usct.physics.operator import _gradient_load_form, _stiffness_form, assemble_helmholtz


@dataclass(frozen=True)
class ForwardFlux:
    """Predicted complex boundary flux in ascending boundary-DOF order."""

    flux: NDArray[np.complex128]  # shape (boundary_dofs, patterns)
    field: NDArray[np.complex128]  # full complex field, shape (dofs, patterns)


@dataclass(frozen=True)
class ObjectiveResult:
    """One complex-L2 FWI objective evaluation with its adjoint gradient."""

    misfit: float
    data_misfit: float
    regularization: float
    gradient: NDArray[np.float64]  # dJ/dc, nodal, shape (dofs,)
    predicted_flux: NDArray[np.complex128]  # shape (boundary_dofs, patterns)


def _medium_from_speed(sound_speed: NDArray[np.float64]) -> Medium:
    return Medium(
        sound_speed=np.asarray(sound_speed, dtype=np.float64).copy(),
        description="inversion candidate sound speed",
    )


def _partition_blocks(
    matrix: sparse.csc_matrix,
    interior: NDArray[np.int64],
    boundary: NDArray[np.int64],
) -> tuple[sparse.csc_matrix, sparse.csc_matrix, sparse.csc_matrix]:
    """Return ``A_II`` (csc), ``A_IB``, and ``A_BI`` blocks."""

    a_ii = matrix[interior][:, interior].tocsc()
    a_ib = matrix[interior][:, boundary]
    a_bi = matrix[boundary][:, interior]
    return a_ii, a_ib, a_bi


def _solve_forward(
    domain: Domain,
    matrix: sparse.csc_matrix,
    forcing_values: NDArray[np.complex128],
    factor: SuperLU,
    a_ib: sparse.csc_matrix,
) -> NDArray[np.complex128]:
    interior = domain.interior_dofs
    boundary = domain.boundary_dofs
    rhs = np.asarray(-(a_ib @ forcing_values), dtype=np.complex128)
    interior_field = np.asarray(factor.solve(rhs), dtype=np.complex128)
    if interior_field.ndim == 1:
        interior_field = interior_field[:, None]
    field = np.empty((domain.basis.N, forcing_values.shape[1]), dtype=np.complex128)
    field[interior] = interior_field
    field[boundary] = forcing_values
    return field


def forward_flux(
    domain: Domain,
    sound_speed: NDArray[np.float64],
    forcing: BoundaryForcing,
    omega: float,
    loss: float,
) -> ForwardFlux:
    """Predicted boundary flux ``D = M_b^{-1} (A u)_B`` in ascending boundary-DOF order.

    This is the same observable as :func:`usct.physics.boundary.project_neumann_trace`
    but returned *unsorted* (in ascending boundary-DOF order) so it aligns with
    the boundary mass matrix used by the misfit. Used both to synthesize observed
    data and inside the objective.
    """

    operator = assemble_helmholtz(domain, _medium_from_speed(sound_speed), omega, loss)
    matrix = operator.matrix.tocsc()
    a_ii, a_ib, _ = _partition_blocks(matrix, domain.interior_dofs, domain.boundary_dofs)
    factor = splu(a_ii)
    field = _solve_forward(domain, matrix, forcing.values, factor, a_ib)
    reaction = np.asarray((matrix @ field)[domain.boundary_dofs], dtype=np.complex128)
    boundary_mass = boundary_mass_matrix(domain).astype(np.complex128)
    flux = np.asarray(splu(boundary_mass).solve(reaction), dtype=np.complex128)
    if flux.ndim == 1:
        flux = flux[:, None]
    return ForwardFlux(flux=flux, field=field)


def objective_and_gradient(
    domain: Domain,
    sound_speed: NDArray[np.float64],
    forcing: BoundaryForcing,
    omega: float,
    loss: float,
    observed_flux: NDArray[np.complex128],
    *,
    alpha: float = 0.0,
    background_speed: float | None = None,
    boundary_mass: sparse.csc_matrix | None = None,
    stiffness: sparse.csc_matrix | None = None,
) -> ObjectiveResult:
    """Evaluate the complex-L2 FWI objective and its adjoint-state gradient.

    Parameters
    ----------
    observed_flux
        Target complex flux, shape ``(boundary_dofs, patterns)``, in the same
        ascending boundary-DOF order as :func:`forward_flux`.
    alpha
        H1 (Tikhonov) smoothness weight. ``0`` disables regularization.
    background_speed
        If given, the boundary sound speed is treated as *known* (the immersion
        medium): the gradient is zeroed on boundary DOFs so the optimizer leaves
        them at the background value.
    boundary_mass, stiffness
        Optional pre-assembled ``M_b`` and real stiffness ``K`` (they do not
        depend on ``c``), passed in to avoid re-assembly across optimizer steps.
    """

    speed = np.asarray(sound_speed, dtype=np.float64)
    interior = domain.interior_dofs
    boundary = domain.boundary_dofs
    forcing_values = forcing.values
    pattern_count = forcing_values.shape[1]
    if observed_flux.shape != (boundary.size, pattern_count):
        raise ValueError(
            f"observed_flux must have shape {(boundary.size, pattern_count)}; "
            f"got {observed_flux.shape}"
        )

    operator = assemble_helmholtz(domain, _medium_from_speed(speed), omega, loss)
    matrix = operator.matrix.tocsc()
    a_ii, a_ib, a_bi = _partition_blocks(matrix, interior, boundary)

    # One factorization serves the forward solve AND the adjoint solve.
    factor = splu(a_ii)
    field = _solve_forward(domain, matrix, forcing_values, factor, a_ib)

    reaction = np.asarray((matrix @ field)[boundary], dtype=np.complex128)
    if boundary_mass is None:
        boundary_mass = boundary_mass_matrix(domain)
    boundary_mass_complex = boundary_mass.astype(np.complex128)
    predicted = np.asarray(splu(boundary_mass_complex).solve(reaction), dtype=np.complex128)
    if predicted.ndim == 1:
        predicted = predicted[:, None]

    residual = predicted - observed_flux  # (nB, P); also the adjoint boundary source
    weighted_residual = boundary_mass_complex @ residual
    data_misfit = 0.5 * float(np.real(np.sum(np.conj(residual) * weighted_residual)))

    # Adjoint solve: A_II^H lambda_I = -A_BI^H lambda_B, lambda_B = residual.
    adjoint_rhs = np.asarray(-(a_bi.conj().T @ residual), dtype=np.complex128)
    interior_adjoint = np.asarray(factor.solve(adjoint_rhs, trans="H"), dtype=np.complex128)
    if interior_adjoint.ndim == 1:
        interior_adjoint = interior_adjoint[:, None]
    adjoint = np.empty((domain.basis.N, pattern_count), dtype=np.complex128)
    adjoint[interior] = interior_adjoint
    adjoint[boundary] = residual

    # Gradient w.r.t. nodal squared slowness m = 1/c^2, summed over patterns.
    gradient_m = np.zeros(domain.basis.N, dtype=np.float64)
    damping = 1.0 + 1j * loss
    for pattern in range(pattern_count):
        lam_conj = domain.basis.interpolate(np.conj(adjoint[:, pattern]))
        u_fwd = domain.basis.interpolate(field[:, pattern])
        load = asm(_gradient_load_form, domain.basis, lam_conj=lam_conj, u_fwd=u_fwd)
        gradient_m += np.real(damping * np.asarray(load, dtype=np.complex128))
    gradient_m *= -(omega**2)

    # Chain rule to sound speed: dm/dc = -2 / c^3.
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

    return ObjectiveResult(
        misfit=data_misfit + regularization,
        data_misfit=data_misfit,
        regularization=regularization,
        gradient=gradient,
        predicted_flux=predicted,
    )
