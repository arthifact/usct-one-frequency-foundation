"""Adjoint-state gradient trust tests: central finite differences + Taylor test.

These are the correctness gate for inversion, the analog of the analytic Bessel
test for the forward. If the discrete adjoint does not match a finite-difference
directional derivative to several digits, no reconstruction result can be
trusted, so these run on every commit.
"""

import numpy as np
import pytest

from usct.domain import build_disk
from usct.forcing import build_boundary_forcing
from usct.inversion import forward_flux, objective_and_gradient


def _smooth_speed(domain, amplitude, center, width):
    x = domain.coordinates[:, 0]
    y = domain.coordinates[:, 1]
    bump = amplitude * np.exp(-(((x - center[0]) ** 2 + (y - center[1]) ** 2) / width))
    return 1.0 + bump


@pytest.fixture(scope="module")
def adjoint_case():
    """A small, non-resonant, non-trivial inverse setup with nonzero residual."""

    domain = build_disk(radius=1.0, refinements=3)
    forcing = build_boundary_forcing(domain, ("cos:1", "sin:2"))
    omega = 2.5  # kR ~ 2.5 on the unit disk, away from Bessel resonances
    # Observed data comes from a genuinely different medium so the residual,
    # and therefore the gradient, is nonzero.
    truth = _smooth_speed(domain, amplitude=0.18, center=(0.15, -0.1), width=0.15)
    return domain, forcing, omega, truth


def _directional_fd(evaluate_misfit, base, direction, step):
    plus = evaluate_misfit(base + step * direction)
    minus = evaluate_misfit(base - step * direction)
    return (plus - minus) / (2.0 * step)


@pytest.mark.parametrize("loss", [0.0, 0.05])
@pytest.mark.parametrize("alpha", [0.0, 0.01])
def test_adjoint_matches_central_difference(adjoint_case, loss, alpha):
    domain, forcing, omega, truth = adjoint_case
    observed = forward_flux(domain, truth, forcing, omega, loss).flux

    base = _smooth_speed(domain, amplitude=0.05, center=(-0.1, 0.05), width=0.2)
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(domain.basis.N)
    direction /= np.linalg.norm(direction)

    result = objective_and_gradient(
        domain, base, forcing, omega, loss, observed, alpha=alpha
    )
    analytic = float(result.gradient @ direction)

    def evaluate_misfit(speed):
        return objective_and_gradient(
            domain, speed, forcing, omega, loss, observed, alpha=alpha
        ).misfit

    fd = _directional_fd(evaluate_misfit, base, direction, step=1e-4)
    relative_error = abs(analytic - fd) / max(abs(fd), 1e-30)
    assert relative_error < 1e-5, f"adjoint vs FD relative error {relative_error:.3e}"


def test_taylor_remainder_is_second_order(adjoint_case):
    """|J(c+t d) - J(c) - t g.d| must shrink like t^2 (slope ~2 in log-log)."""

    domain, forcing, omega, truth = adjoint_case
    loss = 0.02
    observed = forward_flux(domain, truth, forcing, omega, loss).flux

    base = _smooth_speed(domain, amplitude=0.05, center=(-0.1, 0.05), width=0.2)
    rng = np.random.default_rng(1)
    direction = rng.standard_normal(domain.basis.N)
    direction /= np.linalg.norm(direction)

    result = objective_and_gradient(domain, base, forcing, omega, loss, observed)
    base_misfit = result.misfit
    slope = float(result.gradient @ direction)

    steps = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4])
    remainders = []
    for step in steps:
        perturbed = objective_and_gradient(
            domain, base + step * direction, forcing, omega, loss, observed
        ).misfit
        remainders.append(abs(perturbed - base_misfit - step * slope))
    remainders = np.array(remainders)

    # A correct gradient gives an O(t^2) remainder (order -> 2 as t -> 0); a
    # wrong gradient leaves an O(t) remainder pinned near order 1. The estimated
    # order must climb into the second-order regime, cleanly above 1.
    orders = np.log(remainders[:-1] / remainders[1:]) / np.log(steps[:-1] / steps[1:])
    assert np.all(np.diff(remainders) < 0), f"remainder not shrinking: {remainders}"
    assert orders[-1] > 1.85, f"finest-step Taylor order {orders[-1]:.3f}; orders {orders}"
    assert np.median(orders) > 1.5, f"Taylor remainder orders {orders}"
