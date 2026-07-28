"""Boundary-pattern parsing and numerical evaluation tests."""

import numpy as np

from usct.physics.domain import Domain
from usct.physics.forcing import build_boundary_forcing, parse_pattern_name


def test_exact_pattern_grammar() -> None:
    assert parse_pattern_name("constant") == ("constant", 0)
    assert parse_pattern_name("cos:12") == ("cos", 12)
    assert parse_pattern_name("sin:3") == ("sin", 3)
    assert parse_pattern_name("exp:0") == ("exp", 0)


def test_pattern_values_use_boundary_angles(small_domain: Domain) -> None:
    forcing = build_boundary_forcing(
        small_domain,
        ("constant", "cos:2", "sin:2", "exp:2"),
    )
    theta = small_domain.boundary_theta
    np.testing.assert_array_equal(forcing.values[:, 0], np.ones(theta.size))
    np.testing.assert_allclose(forcing.values[:, 1], np.cos(2.0 * theta))
    np.testing.assert_allclose(forcing.values[:, 2], np.sin(2.0 * theta))
    np.testing.assert_allclose(forcing.values[:, 3], np.exp(2j * theta))
    assert forcing.values.dtype == np.complex128

