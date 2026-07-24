"""Dirichlet elimination and multi-RHS LU tests."""

import numpy as np

from usct.domain import Domain
from usct.forcing import BoundaryForcing, build_boundary_forcing
from usct.operator import HelmholtzOperator
from usct.solver import solve_dirichlet


def test_boundary_values_and_interior_residual(
    small_domain: Domain,
    small_operator: HelmholtzOperator,
) -> None:
    forcing = build_boundary_forcing(small_domain, ("constant", "cos:1", "exp:2"))
    solution = solve_dirichlet(small_domain, small_operator, forcing)
    np.testing.assert_allclose(
        solution.field[small_domain.boundary_dofs],
        forcing.values,
        rtol=0.0,
        atol=0.0,
    )
    assert max(solution.report.normalized_interior_residuals) < 1e-10
    assert solution.report.maximum_boundary_value_error == 0.0
    assert solution.report.factorizations == 1


def test_multi_rhs_matches_separate_solves(
    small_domain: Domain,
    small_operator: HelmholtzOperator,
) -> None:
    forcing = build_boundary_forcing(small_domain, ("constant", "cos:1", "exp:2"))
    multi = solve_dirichlet(small_domain, small_operator, forcing)
    separate_columns = []
    for column, name in enumerate(forcing.names):
        single = BoundaryForcing((name,), forcing.values[:, [column]])
        separate_columns.append(solve_dirichlet(small_domain, small_operator, single).field)
    separate = np.column_stack(separate_columns)
    np.testing.assert_allclose(multi.field, separate, rtol=1e-12, atol=1e-12)


def test_one_factorization_handles_all_columns(
    monkeypatch,
    small_domain: Domain,
    small_operator: HelmholtzOperator,
) -> None:
    import usct.solver as solver_module

    calls = 0
    original = solver_module.splu

    def counted_splu(matrix):
        nonlocal calls
        calls += 1
        return original(matrix)

    monkeypatch.setattr(solver_module, "splu", counted_splu)
    forcing = build_boundary_forcing(small_domain, ("constant", "cos:1", "sin:1", "exp:2"))
    solve_dirichlet(small_domain, small_operator, forcing)
    assert calls == 1
