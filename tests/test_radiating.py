"""Trust tests for the radiating (impedance) boundary forward.

Same bar as the DtN foundation: an analytic homogeneous-disk check (the Bessel
analog for boundary pressure), a demonstration that the interior resonances are
gone, and a finite-difference gate on the adjoint gradient.
"""

import numpy as np
import pytest
from scipy.special import jn_zeros

from usct.dtn.solver import solve_dirichlet
from usct.physics.boundary import boundary_mass_matrix
from usct.physics.domain import build_disk
from usct.physics.forcing import build_boundary_forcing
from usct.physics.medium import constant_speed
from usct.physics.operator import assemble_helmholtz
from usct.radiating.forward import (
    analytic_boundary_pressure,
    forward_boundary_pressure,
    solve_radiating,
)
from usct.radiating.inversion import radiating_objective_and_gradient


def _boundary_relative_error(numerical, exact, mass):
    error = numerical - exact
    num = float(np.real(np.vdot(error, mass @ error)))
    den = float(np.real(np.vdot(exact, mass @ exact)))
    return float(np.sqrt(num / den))


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_analytic_boundary_pressure(mode):
    """Homogeneous disk boundary pressure matches the exact Bessel solution."""

    domain = build_disk(1.0, refinements=6)
    speed = 1.0
    omega = 4.0  # kR = 4, non-resonant for these modes
    forcing = build_boundary_forcing(domain, (f"exp:{mode}",))
    predicted = forward_boundary_pressure(
        domain, np.full(domain.basis.N, speed), forcing, omega, 0.0, speed
    )[:, 0]
    exact = analytic_boundary_pressure(mode, omega / speed, 1.0, domain.boundary_theta)
    mass = boundary_mass_matrix(domain).astype(np.complex128)
    assert _boundary_relative_error(predicted, exact, mass) < 0.02


def test_no_interior_resonance_vs_dtn_cavity():
    """At a DtN resonance (Bessel zero) the cavity blows up; radiating stays bounded."""

    domain = build_disk(1.0, refinements=6)
    mode = 2
    resonant_kr = float(jn_zeros(mode, 1)[0])  # J_2(kR) = 0 -> DtN cavity resonance
    speed = np.ones(domain.basis.N)
    forcing = build_boundary_forcing(domain, (f"exp:{mode}",))

    # DtN cavity (hard Dirichlet, lossless): interior solve is near-singular here.
    operator = assemble_helmholtz(domain, constant_speed(domain, 1.0), resonant_kr, 0.0)
    dtn_field = solve_dirichlet(domain, operator, forcing).field
    dtn_norm = float(np.max(np.abs(dtn_field)))

    # Radiating boundary: bounded response at the very same kR.
    radiating = solve_radiating(domain, speed, forcing, resonant_kr, 0.0, 1.0)
    radiating_norm = float(np.max(np.abs(radiating.field)))

    assert radiating_norm < 10.0  # O(1), no blow-up
    assert dtn_norm > 50.0 * radiating_norm  # the cavity, by contrast, explodes


@pytest.mark.parametrize("alpha", [0.0, 0.01])
def test_speed_gradient_matches_central_difference(alpha):
    domain = build_disk(1.0, refinements=3)
    forcing = build_boundary_forcing(domain, ("cos:1", "sin:2"))
    omega = 3.0
    speed_bath = 1.0

    x = domain.coordinates[:, 0]
    y = domain.coordinates[:, 1]
    truth = 1.0 + 0.15 * np.exp(-((x - 0.15) ** 2 + (y + 0.1) ** 2) / 0.15)
    observed = forward_boundary_pressure(domain, truth, forcing, omega, 0.0, speed_bath)

    base = 1.0 + 0.05 * np.exp(-((x + 0.1) ** 2 + (y - 0.05) ** 2) / 0.2)
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(domain.basis.N)
    direction /= np.linalg.norm(direction)

    result = radiating_objective_and_gradient(
        domain, base, forcing, omega, 0.0, speed_bath, observed, alpha=alpha
    )
    analytic = float(result.gradient @ direction)

    def misfit(candidate):
        return radiating_objective_and_gradient(
            domain, candidate, forcing, omega, 0.0, speed_bath, observed, alpha=alpha
        ).misfit

    step = 1e-4
    fd = (misfit(base + step * direction) - misfit(base - step * direction)) / (2.0 * step)
    assert abs(analytic - fd) / max(abs(fd), 1e-30) < 1e-5


def test_si_scale_invariance():
    """SI and dimensionless runs at the same kR and contrast agree up to length scale.

    Scaling coordinates by L and speeds by c0 (holding kR and every speed ratio
    fixed) leaves the radiating operator invariant and scales the boundary
    pressure by exactly L. This is the guard against a dimensional bug: if SI
    units were mishandled, the ratio would not be the pure real constant L.
    """

    length = 0.10  # metres
    reference_speed = 1500.0  # m/s
    modes = ("cos:1", "sin:2")

    unit_domain = build_disk(1.0, refinements=5)
    x = unit_domain.coordinates[:, 0]
    y = unit_domain.coordinates[:, 1]
    contrast = 0.03 * np.exp(-((x - 0.2) ** 2 + (y + 0.1) ** 2) / 0.1)  # up to +3%

    forcing_unit = build_boundary_forcing(unit_domain, modes)
    omega_unit = 6.0
    pressure_unit = forward_boundary_pressure(
        unit_domain, 1.0 + contrast, forcing_unit, omega_unit, 0.0, 1.0
    )

    si_domain = build_disk(length, refinements=5)  # same topology, scaled coordinates
    forcing_si = build_boundary_forcing(si_domain, modes)
    omega_si = omega_unit * reference_speed / length  # preserves kR
    pressure_si = forward_boundary_pressure(
        si_domain, reference_speed * (1.0 + contrast), forcing_si, omega_si, 0.0, reference_speed
    )

    ratio = pressure_si / (length * pressure_unit)
    assert np.allclose(ratio, 1.0, rtol=1e-6, atol=1e-8)
