"""Assembly of the inspectable complex Helmholtz operator."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from skfem import BilinearForm, asm
from skfem.helpers import dot, grad

from usct.domain import Domain
from usct.medium import Medium


@BilinearForm
def _stiffness_form(u, v, _):
    return dot(grad(u), grad(v))


@BilinearForm
def _slowness_mass_form(u, v, w):
    return w.slowness_squared * u * v


@dataclass(frozen=True)
class HelmholtzOperator:
    """Complex matrix for ``-Laplacian(u) - k_squared(x) u``."""

    omega: float
    loss: float
    stiffness: sparse.csc_matrix
    wave_mass: sparse.csc_matrix
    matrix: sparse.csc_matrix


def assemble_helmholtz(
    domain: Domain,
    medium: Medium,
    omega: float,
    loss: float,
) -> HelmholtzOperator:
    """Assemble complex128 ``K``, ``H``, and ``A = K - H`` at one angular frequency."""

    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError(f"angular frequency must be positive and finite; got {omega!r}")
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError(f"loss must be nonnegative and finite; got {loss!r}")
    if medium.sound_speed.shape != (domain.basis.N,):
        raise ValueError("medium sound speed is not defined on this domain")

    stiffness_real = asm(_stiffness_form, domain.basis).tocsc()
    slowness = domain.basis.interpolate(1.0 / np.square(medium.sound_speed))
    slowness_mass = asm(
        _slowness_mass_form,
        domain.basis,
        slowness_squared=slowness,
    ).tocsc()
    factor = np.complex128(omega**2 * (1.0 + 1j * loss))
    stiffness = stiffness_real.astype(np.complex128)
    wave_mass = (factor * slowness_mass).astype(np.complex128).tocsc()
    matrix = (stiffness - wave_mass).astype(np.complex128).tocsc()

    if not all(np.all(np.isfinite(item.data)) for item in (stiffness, wave_mass, matrix)):
        raise ValueError("assembled operator contains a non-finite entry")
    return HelmholtzOperator(
        omega=float(omega),
        loss=float(loss),
        stiffness=stiffness,
        wave_mass=wave_mass,
        matrix=matrix,
    )
