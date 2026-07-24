"""Circular mesh topology and geometry tests."""

import numpy as np

from usct.domain import build_disk


def test_every_dof_is_exactly_interior_or_boundary() -> None:
    domain = build_disk(radius=0.7, refinements=3)
    assert np.intersect1d(domain.boundary_dofs, domain.interior_dofs).size == 0
    partition = np.sort(np.concatenate((domain.boundary_dofs, domain.interior_dofs)))
    np.testing.assert_array_equal(partition, np.arange(domain.basis.N))


def test_boundary_radii_match_disk_radius() -> None:
    radius = 0.7
    domain = build_disk(radius=radius, refinements=3)
    boundary_radii = np.linalg.norm(domain.coordinates[domain.boundary_dofs], axis=1)
    np.testing.assert_allclose(boundary_radii, radius, rtol=0.0, atol=1e-12)


def test_boundary_order_and_angles_are_deterministic() -> None:
    first = build_disk(radius=1.0, refinements=3)
    second = build_disk(radius=1.0, refinements=3)
    np.testing.assert_array_equal(first.boundary_dofs, second.boundary_dofs)
    np.testing.assert_array_equal(first.boundary_theta, second.boundary_theta)
    assert np.all(np.diff(first.boundary_dofs) > 0)


def test_refinement_decreases_h_max() -> None:
    h_values = [build_disk(1.0, refinement).h_max for refinement in (2, 3, 4)]
    assert h_values[1] < h_values[0]
    assert h_values[2] < h_values[1]
