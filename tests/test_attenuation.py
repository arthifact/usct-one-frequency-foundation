"""Trust tests for the joint (sound speed, attenuation) adjoint gradient.

Same correctness bar as the speed-only adjoint: each gradient component must
match a central finite difference, and the complex-medium assembly must reduce
exactly to the verified speed-only path when attenuation is zero.
"""

import numpy as np
import pytest

from usct.dtn.attenuation import (
    forward_flux_complex,
    joint_objective_and_gradient,
)
from usct.dtn.inversion import objective_and_gradient
from usct.physics.domain import build_disk
from usct.physics.forcing import build_boundary_forcing


def _bump(domain, amplitude, center, width, floor=0.0):
    x = domain.coordinates[:, 0]
    y = domain.coordinates[:, 1]
    return floor + amplitude * np.exp(-(((x - center[0]) ** 2 + (y - center[1]) ** 2) / width))


@pytest.fixture(scope="module")
def joint_case():
    domain = build_disk(radius=1.0, refinements=3)
    forcing = build_boundary_forcing(domain, ("cos:1", "sin:2"))
    omega = 2.5
    speed_truth = _bump(domain, 0.18, (0.15, -0.1), 0.15, floor=1.0)
    loss_truth = _bump(domain, 0.05, (-0.1, 0.2), 0.2, floor=0.01)
    observed = forward_flux_complex(domain, speed_truth, loss_truth, forcing, omega)
    speed = _bump(domain, 0.05, (-0.1, 0.05), 0.2, floor=1.0)
    loss = _bump(domain, 0.02, (0.1, 0.1), 0.25, floor=0.01)
    return domain, forcing, omega, observed, speed, loss


def _misfit(domain, speed, loss, forcing, omega, observed):
    return joint_objective_and_gradient(
        domain, speed, loss, forcing, omega, observed
    ).misfit


def test_speed_gradient_matches_central_difference(joint_case):
    domain, forcing, omega, observed, speed, loss = joint_case
    result = joint_objective_and_gradient(domain, speed, loss, forcing, omega, observed)

    rng = np.random.default_rng(0)
    direction = rng.standard_normal(domain.basis.N)
    direction /= np.linalg.norm(direction)
    analytic = float(result.speed_gradient @ direction)

    step = 1e-4
    plus = _misfit(domain, speed + step * direction, loss, forcing, omega, observed)
    minus = _misfit(domain, speed - step * direction, loss, forcing, omega, observed)
    fd = (plus - minus) / (2.0 * step)
    assert abs(analytic - fd) / max(abs(fd), 1e-30) < 1e-5


def test_loss_gradient_matches_central_difference(joint_case):
    domain, forcing, omega, observed, speed, loss = joint_case
    result = joint_objective_and_gradient(domain, speed, loss, forcing, omega, observed)

    rng = np.random.default_rng(1)
    direction = rng.standard_normal(domain.basis.N)
    direction /= np.linalg.norm(direction)
    analytic = float(result.loss_gradient @ direction)

    step = 1e-5
    plus = _misfit(domain, speed, loss + step * direction, forcing, omega, observed)
    minus = _misfit(domain, speed, loss - step * direction, forcing, omega, observed)
    fd = (plus - minus) / (2.0 * step)
    assert abs(analytic - fd) / max(abs(fd), 1e-30) < 1e-5


def test_reduces_to_speed_only_adjoint_at_zero_loss(joint_case):
    """With eta = 0 the complex path must reproduce the verified speed-only path."""

    domain, forcing, omega, _observed, speed, _loss = joint_case
    zero_loss = np.zeros(domain.basis.N)
    # Independent observed data for this check, generated losslessly.
    truth = _bump(domain, 0.2, (0.2, 0.0), 0.15, floor=1.0)
    observed = forward_flux_complex(domain, truth, zero_loss, forcing, omega)

    joint = joint_objective_and_gradient(domain, speed, zero_loss, forcing, omega, observed)
    speed_only = objective_and_gradient(domain, speed, forcing, omega, 0.0, observed)

    assert abs(joint.misfit - speed_only.misfit) < 1e-10 * max(abs(speed_only.misfit), 1e-30)
    assert np.allclose(joint.speed_gradient, speed_only.gradient, rtol=1e-8, atol=1e-10)
