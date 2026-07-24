"""Weak L2 projection of the outward Neumann boundary trace."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import splu
from skfem import BilinearForm, FacetBasis, asm

from usct.domain import Domain
from usct.operator import HelmholtzOperator
from usct.solver import FieldSolution, normalize_volume_rhs


@BilinearForm
def _boundary_mass_form(u, v, _):
    return u * v


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


def boundary_mass_matrix(domain: Domain):
    """Assemble the real boundary mass matrix restricted to boundary degrees of freedom."""

    facets = domain.mesh.boundary_facets()
    facet_basis = FacetBasis(domain.mesh, domain.basis.elem, facets=facets)
    full_mass = asm(_boundary_mass_form, facet_basis).tocsc()
    boundary = domain.boundary_dofs
    return full_mass[boundary][:, boundary].tocsc()


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
