"""Complex Helmholtz operator assembly tests."""

import numpy as np
from scipy.sparse.linalg import norm

from usct.physics.domain import Domain
from usct.physics.medium import Medium, constant_speed
from usct.physics.operator import assemble_helmholtz


def test_operator_shape_dtype_finiteness_and_symmetry(
    small_domain: Domain,
    small_medium: Medium,
) -> None:
    operator = assemble_helmholtz(small_domain, small_medium, omega=2.0, loss=0.03)
    assert operator.matrix.shape == (small_domain.basis.N, small_domain.basis.N)
    assert operator.matrix.dtype == np.dtype(np.complex128)
    assert np.all(np.isfinite(operator.matrix.data))
    transpose_error = norm(operator.matrix - operator.matrix.T) / norm(operator.matrix)
    assert transpose_error < 1e-14
    hermitian_error = norm(operator.matrix - operator.matrix.conj().T) / norm(operator.matrix)
    assert hermitian_error > 0.0


def test_frequency_and_speed_change_wave_term_not_stiffness(small_domain: Domain) -> None:
    medium_one = constant_speed(small_domain, 1.0)
    medium_two = constant_speed(small_domain, 1.3)
    base = assemble_helmholtz(small_domain, medium_one, omega=2.0, loss=0.0)
    frequency_changed = assemble_helmholtz(
        small_domain,
        medium_one,
        omega=2.5,
        loss=0.0,
    )
    speed_changed = assemble_helmholtz(small_domain, medium_two, omega=2.0, loss=0.0)
    assert norm(base.stiffness - frequency_changed.stiffness) == 0.0
    assert norm(base.stiffness - speed_changed.stiffness) == 0.0
    assert norm(base.wave_mass - frequency_changed.wave_mass) > 0.0
    assert norm(base.wave_mass - speed_changed.wave_mass) > 0.0


def test_loss_is_only_present_when_requested(
    small_domain: Domain,
    small_medium: Medium,
) -> None:
    lossless = assemble_helmholtz(small_domain, small_medium, omega=2.0, loss=0.0)
    lossy = assemble_helmholtz(small_domain, small_medium, omega=2.0, loss=0.02)
    assert np.max(np.abs(lossless.wave_mass.data.imag)) == 0.0
    assert np.max(np.abs(lossy.wave_mass.data.imag)) > 0.0
    np.testing.assert_allclose(lossy.wave_mass.data.real, lossless.wave_mass.data.real)
