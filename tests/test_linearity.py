"""Complex superposition and Fourier-pattern identity tests."""

import numpy as np

from usct.dtn.measurement import project_neumann_trace
from usct.dtn.solver import solve_dirichlet
from usct.physics.domain import Domain
from usct.physics.forcing import BoundaryForcing, build_boundary_forcing
from usct.physics.operator import HelmholtzOperator


def _relative_error(numerical, reference) -> float:
    return float(np.linalg.norm(numerical - reference) / np.linalg.norm(reference))


def test_complex_linearity_for_field_and_boundary_response(
    small_domain: Domain,
    small_operator: HelmholtzOperator,
) -> None:
    forcing = build_boundary_forcing(small_domain, ("cos:1", "sin:2"))
    basis_solution = solve_dirichlet(small_domain, small_operator, forcing)
    basis_response = project_neumann_trace(small_domain, small_operator, basis_solution)
    scalar_a = 0.4 + 0.7j
    scalar_b = -0.2 + 0.3j
    combined_values = scalar_a * forcing.values[:, [0]] + scalar_b * forcing.values[:, [1]]
    combined_forcing = BoundaryForcing(("combined",), combined_values)
    combined_solution = solve_dirichlet(small_domain, small_operator, combined_forcing)
    combined_response = project_neumann_trace(
        small_domain,
        small_operator,
        combined_solution,
    )
    expected_field = (
        scalar_a * basis_solution.field[:, [0]] + scalar_b * basis_solution.field[:, [1]]
    )
    expected_response = (
        scalar_a * basis_response.complex_response[:, [0]]
        + scalar_b * basis_response.complex_response[:, [1]]
    )
    assert _relative_error(combined_solution.field, expected_field) < 1e-10
    assert _relative_error(combined_response.complex_response, expected_response) < 1e-10


def test_complex_exponential_equals_cosine_plus_i_sine(
    small_domain: Domain,
    small_operator: HelmholtzOperator,
) -> None:
    forcing = build_boundary_forcing(small_domain, ("cos:2", "sin:2", "exp:2"))
    solution = solve_dirichlet(small_domain, small_operator, forcing)
    response = project_neumann_trace(small_domain, small_operator, solution)
    assert (
        _relative_error(
            solution.field[:, 2],
            solution.field[:, 0] + 1j * solution.field[:, 1],
        )
        < 1e-10
    )
    assert (
        _relative_error(
            response.complex_response[:, 2],
            response.complex_response[:, 0] + 1j * response.complex_response[:, 1],
        )
        < 1e-10
    )
