"""Spatially-varying attenuation: complex-medium forward + joint (c, eta) adjoint.

Physics-fidelity research layered on the verified foundation, without modifying
any verified numerics. Two gaps from `docs/inversion_hypothesis.md` are addressed
here:

* attenuation as a **physical, spatially-varying field** rather than the single
  scalar `eta` numerical stabilizer, and
* the adjoint gradient with respect to that attenuation field, so it can be
  *reconstructed* alongside sound speed.

Model. The operator becomes `A = K - omega^2 M(s)` with a nodal **complex**
slowness

    s(x) = (1 / c(x)^2) * (1 + i eta(x)),

where `eta(x) >= 0` is the loss tangent. It maps to a physical attenuation
coefficient by `eta = 2 c alpha / omega` (alpha in nepers per unit length), so a
frequency-linear tissue attenuation `alpha ~ f` gives a nearly frequency-flat
`eta`. Distributed loss also moves the interior Dirichlet eigenvalues off the
real axis, damping the cavity resonances the lossless DtN model suffers.

Adjoint. `A` stays complex-symmetric (`A^T = A`), so the same trick as
`usct.dtn.inversion` holds: the forward interior LU factorization is re-used for the
adjoint solve (`trans="H"`). A single assembled load `L_j = integral phi_j
conj(lambda) u` gives *both* gradients:

    dJ/dm_j   = -omega^2 * Re[(1 + i eta_j) L_j]     (-> sound speed)
    dJ/deta_j =  omega^2 * m_j * Im[L_j]              (-> attenuation)

with `m = 1/c^2`. The scalar-`eta`, speed-only case reduces exactly to
`usct.dtn.inversion.objective_and_gradient` (cross-checked in tests). Both gradients
are verified against finite differences and a Taylor test in
`tests/test_attenuation.py`.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import splu
from skfem import BilinearForm, asm

from usct.physics.boundary import boundary_mass_matrix
from usct.physics.domain import Domain
from usct.physics.forcing import BoundaryForcing
from usct.physics.operator import _gradient_load_form, _stiffness_form


@BilinearForm(dtype=np.complex128)
def _complex_slowness_mass_form(u, v, w):
    return w.slowness * u * v


@dataclass(frozen=True)
class JointObjectiveResult:
    """Objective value with adjoint gradients wrt sound speed and attenuation."""

    misfit: float
    data_misfit: float
    regularization: float
    speed_gradient: NDArray[np.float64]  # dJ/dc, nodal
    loss_gradient: NDArray[np.float64]  # dJ/d eta, nodal
    predicted_flux: NDArray[np.complex128]


def complex_slowness(
    sound_speed: NDArray[np.float64], loss: NDArray[np.float64] | float
) -> NDArray[np.complex128]:
    """Nodal complex slowness ``s = (1/c^2)(1 + i eta)``."""

    speed = np.asarray(sound_speed, dtype=np.float64)
    reciprocal = 1.0 / np.square(speed)
    return (reciprocal * (1.0 + 1j * np.asarray(loss, dtype=np.float64))).astype(np.complex128)


def assemble_complex_matrix(
    domain: Domain, comp_slowness: NDArray[np.complex128], omega: float
) -> sparse.csc_matrix:
    """Assemble ``A = K - omega^2 M(s)`` for a nodal complex slowness field."""

    stiffness = asm(_stiffness_form, domain.basis).astype(np.complex128)
    interpolated = domain.basis.interpolate(np.asarray(comp_slowness, dtype=np.complex128))
    wave_mass = asm(_complex_slowness_mass_form, domain.basis, slowness=interpolated)
    return (stiffness - (omega**2) * wave_mass).tocsc()


def _solve_forward(domain, matrix, forcing_values, factor, a_ib):
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


def forward_flux_complex(
    domain: Domain,
    sound_speed: NDArray[np.float64],
    loss: NDArray[np.float64] | float,
    forcing: BoundaryForcing,
    omega: float,
) -> NDArray[np.complex128]:
    """Predicted boundary flux for a complex (attenuating) medium, ascending order."""

    matrix = assemble_complex_matrix(domain, complex_slowness(sound_speed, loss), omega)
    interior, boundary = domain.interior_dofs, domain.boundary_dofs
    a_ii = matrix[interior][:, interior].tocsc()
    a_ib = matrix[interior][:, boundary]
    factor = splu(a_ii)
    field = _solve_forward(domain, matrix, forcing.values, factor, a_ib)
    reaction = np.asarray((matrix @ field)[boundary], dtype=np.complex128)
    boundary_mass = boundary_mass_matrix(domain).astype(np.complex128)
    flux = np.asarray(splu(boundary_mass).solve(reaction), dtype=np.complex128)
    if flux.ndim == 1:
        flux = flux[:, None]
    return flux


def joint_objective_and_gradient(
    domain: Domain,
    sound_speed: NDArray[np.float64],
    loss: NDArray[np.float64],
    forcing: BoundaryForcing,
    omega: float,
    observed_flux: NDArray[np.complex128],
    *,
    alpha_speed: float = 0.0,
    alpha_loss: float = 0.0,
    background_speed: float | None = None,
    background_loss: float | None = None,
    boundary_mass: sparse.csc_matrix | None = None,
    stiffness: sparse.csc_matrix | None = None,
) -> JointObjectiveResult:
    """Complex-L2 misfit with adjoint gradients wrt both ``c`` and ``eta`` fields."""

    speed = np.asarray(sound_speed, dtype=np.float64)
    eta = np.asarray(loss, dtype=np.float64)
    reciprocal_squared = 1.0 / np.square(speed)
    interior, boundary = domain.interior_dofs, domain.boundary_dofs
    forcing_values = forcing.values
    pattern_count = forcing_values.shape[1]
    if observed_flux.shape != (boundary.size, pattern_count):
        raise ValueError(
            f"observed_flux must have shape {(boundary.size, pattern_count)}; "
            f"got {observed_flux.shape}"
        )

    matrix = assemble_complex_matrix(domain, complex_slowness(speed, eta), omega)
    a_ii = matrix[interior][:, interior].tocsc()
    a_ib = matrix[interior][:, boundary]
    a_bi = matrix[boundary][:, interior]

    factor = splu(a_ii)
    field = _solve_forward(domain, matrix, forcing_values, factor, a_ib)

    reaction = np.asarray((matrix @ field)[boundary], dtype=np.complex128)
    if boundary_mass is None:
        boundary_mass = boundary_mass_matrix(domain)
    boundary_mass_complex = boundary_mass.astype(np.complex128)
    predicted = np.asarray(splu(boundary_mass_complex).solve(reaction), dtype=np.complex128)
    if predicted.ndim == 1:
        predicted = predicted[:, None]

    residual = predicted - observed_flux
    weighted_residual = boundary_mass_complex @ residual
    data_misfit = 0.5 * float(np.real(np.sum(np.conj(residual) * weighted_residual)))

    adjoint_rhs = np.asarray(-(a_bi.conj().T @ residual), dtype=np.complex128)
    interior_adjoint = np.asarray(factor.solve(adjoint_rhs, trans="H"), dtype=np.complex128)
    if interior_adjoint.ndim == 1:
        interior_adjoint = interior_adjoint[:, None]
    adjoint = np.empty((domain.basis.N, pattern_count), dtype=np.complex128)
    adjoint[interior] = interior_adjoint
    adjoint[boundary] = residual

    # One complex load per pattern feeds BOTH gradients.
    load = np.zeros(domain.basis.N, dtype=np.complex128)
    for pattern in range(pattern_count):
        lam_conj = domain.basis.interpolate(np.conj(adjoint[:, pattern]))
        u_fwd = domain.basis.interpolate(field[:, pattern])
        load += np.asarray(
            asm(_gradient_load_form, domain.basis, lam_conj=lam_conj, u_fwd=u_fwd),
            dtype=np.complex128,
        )

    gradient_m = -(omega**2) * np.real((1.0 + 1j * eta) * load)
    speed_gradient = (-2.0 / speed**3) * gradient_m
    loss_gradient = (omega**2) * reciprocal_squared * np.imag(load)

    regularization = 0.0
    if alpha_speed > 0.0 or alpha_loss > 0.0:
        if stiffness is None:
            stiffness = asm(_stiffness_form, domain.basis).tocsc()
        if alpha_speed > 0.0:
            stiffness_c = np.asarray(stiffness @ speed, dtype=np.float64)
            regularization += 0.5 * alpha_speed * float(speed @ stiffness_c)
            speed_gradient = speed_gradient + alpha_speed * stiffness_c
        if alpha_loss > 0.0:
            stiffness_eta = np.asarray(stiffness @ eta, dtype=np.float64)
            regularization += 0.5 * alpha_loss * float(eta @ stiffness_eta)
            loss_gradient = loss_gradient + alpha_loss * stiffness_eta

    if background_speed is not None:
        speed_gradient[boundary] = 0.0
    if background_loss is not None:
        loss_gradient[boundary] = 0.0

    return JointObjectiveResult(
        misfit=data_misfit + regularization,
        data_misfit=data_misfit,
        regularization=regularization,
        speed_gradient=speed_gradient,
        loss_gradient=loss_gradient,
        predicted_flux=predicted,
    )
