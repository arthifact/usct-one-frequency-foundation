"""DtN measurement: weak L2 projection of the outward Neumann boundary trace.

The observable for sim 1 (Dirichlet-to-Neumann): given a solved pressure field,
recover the outward normal flux ``D = I + iQ`` on the boundary by an L2 projection
against the shared boundary mass matrix. The radiating sim's observable (boundary
pressure) lives in :mod:`usct.radiating.forward` instead.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import splu

from usct.dtn.solver import FieldSolution, normalize_volume_rhs
from usct.physics.boundary import boundary_mass_matrix
from usct.physics.domain import Domain
from usct.physics.operator import HelmholtzOperator


@dataclass(frozen=True)
class BoundaryResponse:
    """Angle-sorted complex outward flux ``D = I + iQ`` on the boundary."""

    theta: NDArray[np.float64]
    boundary_dofs: NDArray[np.int64]
    complex_response: NDArray[np.complex128]

    @property
    def i(self) -> NDArray[np.float64]:
        """Real I channel."""

        return self.complex_response.real

    @property
    def q(self) -> NDArray[np.float64]:
        """Imaginary Q channel."""

        return self.complex_response.imag

    @property
    def in_phase(self) -> NDArray[np.float64]:
        """Real (I) channel."""

        return self.i

    @property
    def quadrature(self) -> NDArray[np.float64]:
        """Imaginary (Q) channel."""

        return self.q

    @property
    def amplitude(self) -> NDArray[np.float64]:
        """Complex-response magnitude."""

        return np.abs(self.complex_response)

    @property
    def phase(self) -> NDArray[np.float64]:
        """Complex-response phase in radians."""

        return np.angle(self.complex_response)


def project_neumann_trace(
    domain: Domain,
    operator: HelmholtzOperator,
    solution: FieldSolution,
    volume_rhs: NDArray[np.complex128] | None = None,
) -> BoundaryResponse:
    """Recover the outward normal derivative by a weak boundary-mass projection."""

    fields = np.asarray(solution.field, dtype=np.complex128)
    if fields.ndim != 2 or fields.shape[0] != domain.basis.N:
        raise ValueError("solution field shape does not match domain")
    count = fields.shape[1]
    q = normalize_volume_rhs(volume_rhs, domain.basis.N, count)
    reaction = operator.matrix @ fields - q
    boundary = domain.boundary_dofs
    boundary_reaction = np.asarray(reaction[boundary], dtype=np.complex128)
    mass = boundary_mass_matrix(domain).astype(np.complex128)
    projected = np.asarray(splu(mass).solve(boundary_reaction), dtype=np.complex128)
    if projected.ndim == 1:
        projected = projected[:, None]

    order = np.argsort(domain.boundary_theta, kind="stable")
    return BoundaryResponse(
        theta=domain.boundary_theta[order].copy(),
        boundary_dofs=boundary[order].copy(),
        complex_response=projected[order].copy(),
    )
