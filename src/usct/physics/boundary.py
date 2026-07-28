"""Shared boundary tooling used by both sims: mass matrix and flux resampling.

These are sim-independent primitives (no forward-model coupling): the L2 boundary
mass matrix that weights every boundary-misfit inner product, and periodic
resampling of boundary data between two meshes' angular samples. The DtN flux
observable itself lives in :mod:`usct.dtn.measurement`; the radiating pressure
observable in :mod:`usct.radiating.forward`.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from skfem import BilinearForm, FacetBasis, asm

from usct.physics.domain import Domain


@BilinearForm
def _boundary_mass_form(u, v, _):
    return u * v


def boundary_mass_matrix(domain: Domain) -> sparse.csc_matrix:
    """Assemble the real boundary mass matrix restricted to boundary degrees of freedom."""

    facets = domain.mesh.boundary_facets()
    facet_basis = FacetBasis(domain.mesh, domain.basis.elem, facets=facets)
    full_mass = asm(_boundary_mass_form, facet_basis).tocsc()
    boundary = domain.boundary_dofs
    return full_mass[boundary][:, boundary].tocsc()


def resample_flux_to(
    target_theta: NDArray[np.float64],
    source_theta: NDArray[np.float64],
    source_flux: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Periodically interpolate boundary data from one mesh's angles onto another.

    Used to compare data generated on a fine "observation" mesh against a
    prediction on a coarser inversion mesh without committing an inverse crime
    (the two meshes never share nodes).
    """

    order = np.argsort(source_theta)
    sorted_theta = source_theta[order]
    sorted_flux = source_flux[order]
    two_pi = 2.0 * np.pi
    resampled = np.empty((target_theta.size, source_flux.shape[1]), dtype=np.complex128)
    for pattern in range(source_flux.shape[1]):
        real = np.interp(target_theta, sorted_theta, sorted_flux[:, pattern].real, period=two_pi)
        imag = np.interp(target_theta, sorted_theta, sorted_flux[:, pattern].imag, period=two_pi)
        resampled[:, pattern] = real + 1j * imag
    return resampled
