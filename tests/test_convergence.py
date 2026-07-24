"""Three-level analytic mesh-convergence tests."""

import numpy as np

from usct.verification import run_analytic_disk


def test_field_and_flux_errors_decrease_substantially() -> None:
    metrics = [
        run_analytic_disk(refinement, enforce_wavelength_guard=False)[1] for refinement in (3, 4, 5)
    ]
    field_errors = np.asarray([max(item.field_errors) for item in metrics])
    flux_errors = np.asarray([max(item.flux_errors) for item in metrics])
    assert np.all(np.diff(field_errors) < 0.0)
    assert np.all(np.diff(flux_errors) < 0.0)
    assert field_errors[-1] / field_errors[0] < 0.60
    assert flux_errors[-1] / flux_errors[0] < 0.60
