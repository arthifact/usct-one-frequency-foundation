"""Smoke test for the frequency-continuation FWI driver.

The adjoint correctness gate lives in ``test_adjoint.py``; this only checks that
the driver wires together (bounds, warm-start, boundary freeze) and actually
reduces the boundary-flux misfit on a tiny disjoint-mesh problem.
"""

import numpy as np

from usct.dtn.inversion import forward_flux
from usct.dtn.reconstruction import Stage, adaptive_patterns, reconstruct
from usct.physics.boundary import resample_flux_to
from usct.physics.domain import build_disk
from usct.physics.forcing import build_boundary_forcing
from usct.physics.medium import feature_speed


def test_reconstruction_reduces_misfit_and_finds_inclusion():
    observation_domain = build_disk(1.0, refinements=5)
    inversion_domain = build_disk(1.0, refinements=4)

    features = ({"type": "disk", "label": "blob", "center": (0.2, -0.15),
                 "radius": 0.28, "contrast": 0.12},)
    truth = feature_speed(observation_domain, 1.0, 20.0, features).sound_speed

    frequency = 3.0 / (2 * np.pi)
    stage = Stage(
        frequency=frequency,
        patterns=adaptive_patterns(frequency),
        loss=1e-3,
        alpha=1e-4,
        max_iterations=25,
    )
    forcing = build_boundary_forcing(observation_domain, stage.patterns)
    clean = forward_flux(observation_domain, truth, forcing, 2 * np.pi * frequency, stage.loss).flux
    observed = {
        0: resample_flux_to(
            inversion_domain.boundary_theta, observation_domain.boundary_theta, clean
        )
    }

    initial = np.full(inversion_domain.basis.N, 1.0)
    result = reconstruct(
        inversion_domain, observed, (stage,),
        initial_speed=initial, speed_bounds=(0.85, 1.25), background_speed=1.0,
    )

    report = result.reports[0]
    # The optimizer must make real progress on the data misfit.
    assert report.final_data_misfit < 0.2 * report.initial_misfit
    # And the recovered field must show a positive inclusion near the true center.
    interior = inversion_domain.interior_dofs
    truth_inversion = feature_speed(inversion_domain, 1.0, 20.0, features).sound_speed
    error = result.sound_speed[interior] - truth_inversion[interior]
    homogeneous_error = 1.0 - truth_inversion[interior]
    assert np.sqrt(np.mean(error**2)) < 0.6 * np.sqrt(np.mean(homogeneous_error**2))
    assert result.sound_speed.max() > 1.03  # detected the faster inclusion
